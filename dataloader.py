import datetime
import json
import logging
import os.path
from typing import Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import tqdm
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

import utils
from preprocess import SrsRANLteRecordPHY, AmariNSARecord, SrsRANLteSample, SrsRANLteLogFile, AmariNSALogFile


class SrsRANLteHybridEncoder:
    def __init__(self):
        self.channels: List[str] = utils.srsRANLte_channels
        self.channels_minmax_columns: List[List[str]] = [[] for _ in self.channels]
        self.channels_onehot_columns: List[List[str]] = [[] for _ in self.channels]
        self.channels_minmax_scalers: List[utils.AdvancedMinMaxScaler] = []
        self.channels_onehot_encoders: List[utils.AdvancedOneHotEncoder] = []
        self.got_metadata: bool = False
        self.is_fit: bool = False
        self._reset_estimators()

    def _raise_raw_type_error(self):
        raise TypeError(
            "Input raw should be one of following formats: list of pkl paths, single pkl path, list of "
            "SrsRANLteLogfile objects, or single SrsRANLteLogFile object."
        )

    def _collect_columns_metadata_from_logfile(self, logfile: SrsRANLteLogFile):
        for channel_idx, channel in enumerate(self.channels):
            records_channel = [record for record in logfile.records if record.basic_info["channel"] == channel]
            df_raw = pd.DataFrame([record.message for record in records_channel])
            df_raw = df_raw.fillna("-1")
            for column in df_raw.columns:
                try:
                    df_raw[column] = df_raw[column].apply(eval)
                    if column not in (
                            self.channels_minmax_columns[channel_idx] +
                            self.channels_onehot_columns[channel_idx]
                    ):
                        self.channels_minmax_columns[channel_idx].append(column)
                except (NameError, TypeError, SyntaxError) as _:
                    if column in self.channels_minmax_columns[channel_idx]:
                        self.channels_minmax_columns[channel_idx].remove(column)
                        self.channels_onehot_columns[channel_idx].append(column)
                    elif column not in self.channels_onehot_columns[channel_idx]:
                        self.channels_onehot_columns[channel_idx].append(column)

    def collect_columns_metadata(self, raw: List[str] or str or List[SrsRANLteLogFile] or SrsRANLteLogFile):
        """Get column names of each channel and datatype of each column."""
        if raw:
            self.got_metadata = True
            if isinstance(raw, list) and isinstance(raw[0], str):
                for read_pkl_path in (t := tqdm.tqdm(raw)):
                    t.set_postfix({"step": "get_metadata", "read_path": read_pkl_path})
                    with open(read_pkl_path, "rb") as f:
                        logfile = pickle.load(f)
                    self._collect_columns_metadata_from_logfile(logfile)
            elif isinstance(raw, str):
                with open(raw, "rb") as f:
                    logfile = pickle.load(f)
                self._collect_columns_metadata_from_logfile(logfile)
            elif isinstance(raw, list) and isinstance(raw[0], SrsRANLteLogFile):
                for logfile in tqdm.tqdm(raw, postfix={"step": "get_metadata"}):
                    self._collect_columns_metadata_from_logfile(logfile)
            elif isinstance(raw, SrsRANLteLogFile):
                self._collect_columns_metadata_from_logfile(raw)
            else:
                self._raise_raw_type_error()

    def save_columns_metadata(self, save_json_path: str):
        if self.got_metadata:
            with open(save_json_path, "w") as f:
                json_dict = {
                    "channels_minmax_columns": self.channels_minmax_columns,
                    "channels_onehot_columns": self.channels_onehot_columns
                }
                json.dump(json_dict, f)

    def load_columns_metadata(self, read_json_path: str):
        if os.path.isfile(read_json_path):
            self.got_metadata = True
            with open(read_json_path, "r") as f:
                json_dict = json.load(f)
            self.channels_minmax_columns = json_dict["channels_minmax_columns"]
            self.channels_onehot_columns = json_dict["channels_onehot_columns"]

    def get_channels_features_num(self) -> List[int]:
        channels_features_num: List[int] = []
        for channel_idx in range(len(self.channels)):
            channel_minmax_features_num = len(self.channels_minmax_columns[channel_idx])
            channel_onehot_features_num = self.channels_onehot_encoders[channel_idx].get_features_num()
            channels_features_num.append(channel_minmax_features_num + channel_onehot_features_num)
        return channels_features_num

    def _reset_estimators(self):
        self.channels_minmax_scalers = [utils.AdvancedMinMaxScaler(clip=False) for _ in self.channels]
        self.channels_onehot_encoders = [
            utils.AdvancedOneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            for _ in self.channels
        ]
        self.is_fit = False

    def _form_raw_dataframe(self, logfile: SrsRANLteLogFile, channel_idx: int) -> pd.DataFrame:
        records_channel = [
            record
            for record in logfile.records
            if record.basic_info["channel"] == self.channels[channel_idx]
        ]
        df_raw = pd.DataFrame([record.message for record in records_channel])
        df_raw = df_raw.fillna("-1")
        for column in self.channels_minmax_columns[channel_idx]:
            if column not in df_raw.columns:
                df_raw.insert(loc=0, column=column, value=-1)
            else:
                df_raw[column] = df_raw[column].apply(eval)
        for column in self.channels_onehot_columns[channel_idx]:
            if column not in df_raw.columns:
                df_raw.insert(loc=0, column=column, value="-1")
        return df_raw

    def _fit_logfile(self, logfile: SrsRANLteLogFile):
        for channel_idx, channel in enumerate(self.channels):
            df_raw = self._form_raw_dataframe(logfile, channel_idx)
            if self.channels_minmax_columns[channel_idx]:
                self.channels_minmax_scalers[channel_idx].fit(df_raw[self.channels_minmax_columns[channel_idx]])
            if self.channels_onehot_columns[channel_idx]:
                self.channels_onehot_encoders[channel_idx].fit(
                    df_raw[self.channels_onehot_columns[channel_idx]]
                )

    def fit(self, raw: List[str] or str or List[SrsRANLteLogFile] or SrsRANLteLogFile):
        self._reset_estimators()
        if not self.got_metadata:
            self.collect_columns_metadata(raw)
            self.got_metadata = True
        if raw:
            self.is_fit = True
            if isinstance(raw, list) and isinstance(raw[0], str):
                for read_pkl_path in (t := tqdm.tqdm(raw)):
                    t.set_postfix({"step": "fit_estimators", "read_path": read_pkl_path})
                    with open(read_pkl_path, "rb") as f:
                        logfile = pickle.load(f)
                    self._fit_logfile(logfile)
            elif isinstance(raw, str):
                with open(raw, "rb") as f:
                    logfile = pickle.load(f)
                self._fit_logfile(logfile)
            elif isinstance(raw, list) and isinstance(raw[0], SrsRANLteLogFile):
                for logfile in tqdm.tqdm(raw, postfix={"step": "fit_estimators"}):
                    self._fit_logfile(logfile)
            elif isinstance(raw, SrsRANLteLogFile):
                self._fit_logfile(raw)
            else:
                self._raise_raw_type_error()

    def _transform_logfile(self, logfile: SrsRANLteLogFile) -> Tuple[np.ndarray, np.ndarray]:
        for channel_idx, channel in enumerate(self.channels):
            records_channel = [record for record in logfile.records if record.basic_info["channel"] == channel]
            df_raw = self._form_raw_dataframe(logfile, channel_idx)
            df_embedded = pd.DataFrame()
            if self.channels_minmax_columns[channel_idx]:
                scaled = pd.DataFrame(
                    self.channels_minmax_scalers[channel_idx].transform(
                        df_raw[self.channels_minmax_columns[channel_idx]]
                    )
                )
                df_embedded = pd.concat([df_embedded, scaled], axis=1)
            if self.channels_onehot_columns[channel_idx]:
                encoded = pd.DataFrame(
                    self.channels_onehot_encoders[channel_idx].transform(
                        df_raw[self.channels_onehot_columns[channel_idx]]
                    )
                )
                df_embedded = pd.concat([df_embedded, encoded], axis=1)
            df_embedded = df_embedded.to_numpy()
            for record_idx, record in enumerate(records_channel):
                record.embedded_message = df_embedded[record_idx]
        channels_columns_num = self.get_channels_features_num()
        logfile.form_sample_xs(channels_columns_num)
        logfile_X = np.array([sample.x for sample in logfile.samples])
        logfile_labels = np.array([sample.label for sample in logfile.samples])
        return logfile_X, logfile_labels

    def _transform_pkl_path_and_overwrite(self, read_pkl_path: str):
        with open(read_pkl_path, "rb") as f:
            logfile = pickle.load(f)
        logfile_X, logfile_labels = self._transform_logfile(logfile)
        with open(read_pkl_path, "wb") as f:
            pickle.dump(logfile, f)
        np.savez(os.path.splitext(read_pkl_path)[0] + ".npz", X=logfile_X, labels=logfile_labels)

    def _transform_logfile_and_save(
            self,
            logfile: SrsRANLteLogFile,
            save_pkl_path: str = None,
            save_npz_path: str = None
    ):
        logfile_X, logfile_labels = self._transform_logfile(logfile)
        if save_pkl_path:
            with open(save_pkl_path, "wb") as f:
                pickle.dump(logfile, f)
        if save_npz_path:
            np.savez(save_npz_path, X=logfile_X, labels=logfile_labels)

    def transform(
            self,
            raw: List[str] or str or List[SrsRANLteLogFile] or SrsRANLteLogFile,
            save_pkl_paths: List[str] or str = None,
            save_npz_paths: List[str] or str = None
    ):
        """
        Compute `embedded_message` for each record and `x` for each sample in all logfiles.
        Save structured X with labels in new npz files and overwrite original pkl files if input is in format pkl
        path(s) or if input is in format logfile(s) and save paths are given.
        """
        if not self.is_fit:
            raise NotFittedError(
                "This SrsRANLteHybridEncoder instance is not fit yet. "
                "Call 'fit' at least once before using this estimator."
            )
        if raw:
            if isinstance(raw, list) and isinstance(raw[0], str):
                for read_pkl_path in (t := tqdm.tqdm(raw)):
                    t.set_postfix({"step": "transform_samples", "read_path": read_pkl_path})
                    self._transform_pkl_path_and_overwrite(read_pkl_path)
            elif isinstance(raw, str):
                self._transform_pkl_path_and_overwrite(raw)
            elif isinstance(raw, list) and isinstance(raw[0], SrsRANLteLogFile):
                for logfile_idx in tqdm.tqdm(range(len(raw)), postfix={"step": "transform_samples"}):
                    self._transform_logfile_and_save(
                        raw[logfile_idx],
                        save_pkl_paths[logfile_idx] if save_pkl_paths else None,
                        save_npz_paths[logfile_idx] if save_npz_paths else None
                    )
            elif isinstance(raw, SrsRANLteLogFile):
                self._transform_logfile_and_save(raw, save_pkl_paths, save_npz_paths)
            else:
                self._raise_raw_type_error()


class SrsRANLteDataset(Dataset):
    def __init__(
            self,
            params: utils.HyperParams,
            read_npz_paths: List[str],
            label_mapping: Dict[str, str] = None
    ):
        self.X: np.ndarray = np.empty(0)
        self.labels: np.ndarray = np.empty(0)
        for npz_path_idx in range(len(read_npz_paths)):
            npz_dict = np.load(read_npz_paths[npz_path_idx])
            logfile_X: np.ndarray = npz_dict["X"]
            logfile_labels: np.ndarray = npz_dict["labels"]
            if not npz_path_idx:
                self.X = logfile_X
                self.labels = logfile_labels
            else:
                self.X = np.concatenate([self.X, npz_dict["X"]], axis=0)
                self.labels = np.concatenate([self.labels, npz_dict["labels"]], axis=0)
        if label_mapping:
            self.labels = np.array([label_mapping[label] for label in self.labels])
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.labels)

    def save_label_encoder(self, save_dir: str):
        with open(os.path.join(save_dir, "label_encoder.json"), "w") as f:
            json.dump(self.label_encoder.__dict__, f)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.tensor(self.X[idx], dtype=torch.float32), int(self.y[idx])


class AmariNSADataset(Dataset):
    def __init__(
            self,
            params: utils.HyperParams,
            feature_path: str,
            read_log_paths: List[str] = None,
            timetables: List[List[Tuple[Tuple[datetime.time, datetime.time], str]]] = None,
            save_path: str = None,
            read_npz_path: str = None
    ):
        """
        Read log from multiple files and generate generalized dataset (X,y) for ML/DL models,
        feature_path param DEPRECATEd
        """
        # TODO: turn some unsafe attributes to private, this and many other classes
        if not params.re_preprocess and read_npz_path and os.path.isfile(read_npz_path):
            self.re_preprocessed: bool = False
            Xy = np.load(read_npz_path)
            self.X: np.ndarray = Xy["X"]
            self.y: np.ndarray = Xy["y"]
        elif read_log_paths and timetables:
            self.re_preprocessed: bool = True
            self.logfiles: List[AmariNSALogFile] = AmariNSADataset._construct_logfiles(
                params, read_log_paths, timetables
            )
            self._embed_features(params)
            self.label_encoder = LabelEncoder()
            self.X: np.ndarray = self._form_dataset_X()
            self.y: np.ndarray = self._form_dataset_y()
            self._save_Xy(save_path)
        else:
            raise TypeError("Failed to load AmariNSADataset from npz file or log files")

    @staticmethod
    def _construct_logfiles(
            params: utils.HyperParams,
            read_paths: List[str],
            timetables: List[List[Tuple[Tuple[datetime.time, datetime.time], str]]]
    ) -> List[AmariNSALogFile]:
        """Read all logfiles from paths in the given list"""
        logfiles: List[AmariNSALogFile] = []
        for idx in (t := tqdm.trange(len(read_paths))):
            t.set_postfix({"read_path": "\""+read_paths[idx]+"\""})
            logfiles.append(AmariNSALogFile(
                    read_paths[idx],
                    {},
                    timetables[idx],
                    params.window_size,
                    params.pca_n_components,
                    params.tb_len_threshold
            ))
        return logfiles

    def _embed_features_naive(self):
        """Processing key_info vector to pure numeric, DEPRECATED"""
        for logfile in self.logfiles:
            for sample in logfile.samples:
                for record in sample.records:
                    for i in range(len(record.key_info)):
                        try:
                            record.key_info[i] = eval(record.key_info[i])
                        except (NameError, TypeError, SyntaxError) as _:
                            try:
                                record.key_info[i] = eval("".join([str(ord(c)) for c in record.key_info[i]]))
                            except TypeError as _:
                                pass

    def _embed_features(self, params: utils.HyperParams):
        """Embedding key_info vector to pure numeric, rescale features and extract principal components"""
        cell_channel_tuples = [
            (cell_id, channel) for cell_id in utils.amariNSA_channels.keys() for channel in utils.amariNSA_channels[cell_id]
        ]
        for cell_id, channel in (t := tqdm.tqdm(cell_channel_tuples)):
            t.set_postfix({"cell_id": cell_id, "channel": channel})
            # dataframe from record.message
            records_channel = [
                record for logfile in self.logfiles for record in logfile.records
                if record.basic_info["cell_id"] == cell_id and record.basic_info["channel"] == channel
            ]
            # embed
            df_raw = pd.DataFrame([record.message for record in records_channel])
            df_raw.fillna(-1)
            df_embedded = pd.DataFrame()
            columns_minmax: List[str] = []
            columns_onehot: List[str] = []
            for column in df_raw.columns:
                try:
                    df_raw[column] = pd[column].apply(eval)
                    columns_minmax.append(column)
                except (NameError, TypeError, SyntaxError) as _:
                    columns_onehot.append(column)
            if columns_minmax:
                scaled = pd.DataFrame(MinMaxScaler().fit_transform(df_raw[columns_minmax]))
                df_embedded = pd.concat([df_embedded, scaled])
            if columns_onehot:
                encoded = pd.DataFrame(OneHotEncoder(sparse_output=False).fit_transform(df_raw[columns_onehot]))
                df_embedded = pd.concat([df_embedded, encoded])
            # pca
            pca = PCA(n_components=params.pca_n_components)
            summarized = pca.fit_transform(df_embedded.to_numpy())
            for index, record in enumerate(records_channel):
                record.embedded_message = summarized[index]

    def _form_dataset_X(self) -> np.ndarray:
        """Assemble combined vector for each sample as input to ML/DL models"""
        raw_X: List[np.ndarray] = []
        for logfile in self.logfiles:
            for sample in logfile.samples:
                raw_X.append(sample.form_sample_X())  # TODO CONFIG HERE, better approaches？
        return np.array(raw_X)

    def _form_dataset_y(self) -> np.ndarray:
        """Assemble ground-truth labels as input to ML/DL models"""
        raw_y: List[str] = []
        for logfile in self.logfiles:
            for sample in logfile.samples:
                raw_y.append(sample.label)
        self.label_encoder.fit(raw_y)
        return self.label_encoder.transform(raw_y)

    def _save_Xy(self, save_path: str):
        """Write preprocessed X and y to file for further usage"""
        if save_path:
            np.savez(save_path, X=self.X, y=self.y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.tensor(self.X[idx], dtype=torch.float32), int(self.y[idx])

    def plot_channel_statistics(self):
        """Plot bar chart of channel statistics in labelled timezone before sampling, CONFIG ONLY"""
        if not self.re_preprocessed:
            warnings.warn("plot_channel_statistics failed as preprocessing bypassed")
            return
        channel_stat: Dict[str, int] = {}
        for logfile in self.logfiles:
            for record in logfile.records:
                if record.basic_info["channel"] in channel_stat.keys():
                    channel_stat[record.basic_info["channel"]] += 1
                else:
                    channel_stat[record.basic_info["channel"]] = 1
        plt.bar(channel_stat.keys(), channel_stat.values())
        plt.title("PHY Records of Different Channels in Dataset (total {} records)".format(sum(channel_stat.values())))
        plt.ylabel("# records")
        plt.show()

    def plot_tb_len_statistics(self):
        """Plot sum(tb_len) statistics after regroup and threshold filtering, CONFIG ONLY"""
        if not self.re_preprocessed:
            warnings.warn("plot_tb_len_statistics failed as preprocessing bypassed")
            return
        tb_lens_stat: Dict[str, List[int]] = {}
        for logfile in self.logfiles:
            for sample in logfile.samples:
                if sample.label in tb_lens_stat.keys():
                    tb_lens_stat[sample.label].append(sample.tb_len)
                else:
                    tb_lens_stat[sample.label] = [sample.tb_len]
        plt.hist(tb_lens_stat.values(), density=False, histtype='bar', stacked=False, label=list(tb_lens_stat.keys()))
        plt.yscale('log')
        plt.title("Samples with Different sum(tb_len) After Threshold (total {} samples)".format(
            sum(len(list_) for list_ in tb_lens_stat.values())
        ))
        plt.ylabel("# samples")
        plt.xlabel("sum(tb_len)")
        plt.legend()
        plt.show()

    def count_feature_combinations(self):
        """Count different combinations of features for each physical channel for feature selection, CONFIG ONLY"""
        if not self.re_preprocessed:
            warnings.warn("count_feature_combinations failed as preprocessing bypassed")
            return
        # TODO: better solution? minor details
        for channel in utils.amariNSA_channels["03"]:
            print(">>", channel)
            combinations: Dict[str, int] = {}
            for logfile in self.logfiles:
                for sample in logfile.samples:
                    for record in sample.records:
                        if record.basic_info["channel"] == channel:
                            combination_list = list(record.basic_info.keys())
                            combination_list.extend(list(record.short_message.keys()))
                            combination_list.extend(list(record.long_message.keys()))
                            combination = str(sorted(combination_list))
                            if combination not in combinations.keys():
                                combinations[combination] = 1
                            else:
                                combinations[combination] += 1
            all_features = sorted(list(
                set().union(*[json.loads(key.replace("'", "\"")) for key in combinations.keys()])
            ))
            blanked_combinations: Dict[str, int] = {}
            for combination, nb_appearance in combinations.items():
                blanked_combination_list = all_features.copy()
                for idx, feature in enumerate(blanked_combination_list):
                    if ("'" + str(feature) + "'") not in combination:
                        blanked_combination_list[idx] = " " * len(blanked_combination_list[idx])
                blanked_combinations[str(blanked_combination_list)] = nb_appearance
            for blanked_combination, nb_appearance in blanked_combinations.items():
                print(
                    "{:>10}\t".format(int(nb_appearance)),
                    ' '.join(json.loads(blanked_combination.replace("'", "\"")))
                )
        print("\n")


class SrsRANLteDataLoaders:
    def __init__(
            self,
            params: utils.HyperParams,
            read_log_paths: List[str] = None,
            labels: List[str] = None,
            hybrid_encoder: SrsRANLteHybridEncoder = SrsRANLteHybridEncoder(),
            label_encoder: LabelEncoder = LabelEncoder(),
            save_path: str = None,
            read_npz_paths: List[str] = None,
            split_percentages: List[float] = None
    ):
        """
        Split percentages given in params can be overwritten by passing percentages to `split_percentages`.
        """
        self.dataset = SrsRANLteDataset(
            params,
            read_log_paths,
            labels,
            hybrid_encoder,
            label_encoder,
            save_path,
            read_npz_paths
        )
        if not split_percentages:
            split_percentages = [
                (1 - params.split_val_percentage - params.split_test_percentage),
                params.split_val_percentage,
                params.split_test_percentage
            ]
        split_datasets = random_split(
            self.dataset,
            lengths=split_percentages,
            generator=torch.Generator().manual_seed(params.random_seed)
        )
        # TODO: maybe move this to Dataset so that functions in ml.py can use it directly but not calculate again
        if split_percentages[0] > 0:
            self.train = DataLoader(split_datasets[0], params.batch_size, shuffle=True)
        if split_percentages[1] > 0:
            self.val = DataLoader(split_datasets[1], params.batch_size, shuffle=False)
        if split_percentages[2] > 0:
            self.test = DataLoader(split_datasets[2], params.batch_size, shuffle=False)


class AmariNSADataLoaders:
    def __init__(
            self,
            params: utils.HyperParams,
            feature_path: str,
            read_log_paths: List[str] = None,
            timetables: List[List[Tuple[Tuple[datetime.time, datetime.time], str]]] = None,
            save_path: str = None,
            read_npz_path: str = None
    ):
        """Get train, validation and test dataloader"""
        self.dataset = AmariNSADataset(params, feature_path, read_log_paths, timetables, save_path, read_npz_path)
        split_datasets = random_split(
            self.dataset,
            lengths=[
                (1 - params.split_val_percentage - params.split_test_percentage),
                params.split_val_percentage,
                params.split_test_percentage
            ],
            generator=torch.Generator().manual_seed(params.random_seed)
        )
        self.num_features: int = params.pca_n_components * sum([
            len(channels) for channels in utils.amariNSA_channels.values()
        ])
        self.num_classes: int = len(set(self.dataset.y))
        # TODO: maybe move this to Dataset so that functions in ml.py can use it directly but not calculate again
        self.train = DataLoader(split_datasets[0], params.batch_size, shuffle=True)
        self.val = DataLoader(split_datasets[1], params.batch_size, shuffle=False)
        self.test = DataLoader(split_datasets[2], params.batch_size, shuffle=False)


if __name__ == "__main__":
    data_folder = "data/srsRAN/srsenb0219"
    # hybrid_encoder = SrsRANLteHybridEncoder()
    # hybrid_encoder.collect_columns_metadata(data_folder)
    # hybrid_encoder.load_columns_metadata("data/srsRAN/srsenb0219/columns_metadata.json")
    # hybrid_encoder.save_columns_metadata(data_folder)
    # hybrid_encoder.fit(data_folder)
    # with open(os.path.join(data_folder, "hybrid_encoder.pickle"), "wb") as file:
    #     pickle.dump(hybrid_encoder, file)
    with open("data/srsRAN/srsenb0219/hybrid_encoder.pickle", "rb") as file:
        hybrid_encoder = pickle.load(file)
    hybrid_encoder.transform(data_folder)

    # with open("tmp/hybrid_encoder.pkl", "rb") as file:
    #     hybrid_encoder = pickle.load(file)
    # hybrid_encoder.transform(data_folder)

    # """Unit test of AmariNSADataset"""
    # params = utils.HyperParams(json_path="../experiments/base/params.json")
    # params.re_preprocess = True
    # dl = AmariNSADataLoaders(
    #     params=params,
    #     feature_path="../experiments/base/features.json",
    #     read_log_paths=["../data/NR/1st-example/gnb0.log"],
    #     timetables=[[
    #         ((datetime.time(9, 48, 20), datetime.time(9, 58, 40)), "navigation_web"),
    #         ((datetime.time(10, 1, 40), datetime.time(10, 13, 20)), "streaming_youtube")
    #     ]],
    #     save_path="../data/NR/1st-example/dataset_Xy.npz")
    # dl.dataset.plot_channel_statistics()
    # dl.dataset.plot_tb_len_statistics()
    # dl.dataset.count_feature_combinations()

    # """Unit test of AmariNSADataLoaders"""
    # params = utils.HyperParams(json_path="experiments/base/params.json")
    # dataloaders = AmariNSADataLoaders(
    #     params=params,
    #     feature_path="",
    #     read_log_paths=["data/NR/1st-example/gnb0.log"],
    #     timetables=[[
    #         ((datetime.time(9, 48, 20), datetime.time(9, 58, 40)), "navigation_web"),
    #         ((datetime.time(10, 1, 40), datetime.time(10, 13, 20)), "streaming_youtube")
    #     ]],
    #     save_path="",
    #     read_npz_path=""
    # )

    # # Unit test of SrsRANLteDataset
    # params = utils.HyperParams(json_path="experiments/base/params.json")
    # hybrid_encoder = SrsRANLteHybridEncoder()
    # label_encoder = LabelEncoder()
    # # 6550
    # dataset = SrsRANLteDataset(
    #     params=params,
    #     read_log_paths=[
    #         "data/srsRAN/srsenb1020/tmeeting_video_6550.log",
    #         "data/srsRAN/srsenb1020/tmeeting_audio_6550.log",
    #         "data/srsRAN/srsenb1020/fastping_1721601_6550.log",
    #         "data/srsRAN/srsenb1020/zhihu_browse_6550.log",
    #         "data/srsRAN/srsenb1020/qqmusic_standard_6550.log",
    #         "data/srsRAN/srsenb1020/bilibili_1080p_6550.log",
    #         "data/srsRAN/srsenb1020/bilibili_live_6550.log",
    #         "data/srsRAN/srsenb1020/tiktok_browse_6550.log",
    #         "data/srsRAN/srsenb1020/wget_anaconda_6550.log",
    #         "data/srsRAN/srsenb1022/netdisk_upload_6550.log"
    #     ],
    #     labels=[
    #         "tmeeting_video",
    #         "tmeeting_audio",
    #         "fastping",
    #         "zhihu",
    #         "qqmusic",
    #         "bilibili_video",
    #         "bilibili_live",
    #         "tiktok",
    #         "wget_anaconda",
    #         "netdisk_upload"
    #     ],
    #     hybrid_encoder=hybrid_encoder,
    #     label_encoder=label_encoder,
    #     save_path="data/srsRAN/dataset_Xy_6550.npz"
    # )
    # with open("hybrid_encoder", 'wb') as f:
    #     pickle.dump(hybrid_encoder, f)
    # with open("label_encoder", 'wb') as f:
    #     pickle.dump(label_encoder, f)
    # # 6040
    # dataset = SrsRANLteDataset(
    #     params=params,
    #     read_log_paths=[
    #         "data/srsRAN/srsenb1018-2/qqmusic_standard_6040.log",
    #         "data/srsRAN/srsenb1018-2/bilibili_480p_6040.log",
    #         "data/srsRAN/srsenb1018-2/wget_anaconda_6040.log",
    #         "data/srsRAN/srsenb1018-2/bilibili_live_6040.log",
    #         "data/srsRAN/srsenb1018-2/tmeeting_video_6040.log",
    #         "data/srsRAN/srsenb1018-2/tmeeting_audio_6040.log",
    #         "data/srsRAN/srsenb1018-2/zhihu_6040.log",
    #         "data/srsRAN/srsenb1018-2/fastping_1721601_6040.log",
    #     ],
    #     labels=[
    #         "qqmusic",
    #         "bilibili_video",
    #         "wget_anaconda",
    #         "bilibili_live",
    #         "tmeeting_video",
    #         "tmeeting_audio",
    #         "zhihu",
    #         "fastping"
    #     ],
    #     hybrid_encoder=hybrid_encoder,
    #     label_encoder=label_encoder,
    #     save_path="data/srsRAN/dataset_Xy_6040.npz"
    # )
    # # 7060
    # dataset = SrsRANLteDataset(
    #     params=params,
    #     read_log_paths=[
    #         "data/srsRAN/srsenb1020/tmeeting_video_7060.log",
    #         "data/srsRAN/srsenb1020/tmeeting_audio_7060.log",
    #         "data/srsRAN/srsenb1020/fastping_1721601_7060.log",
    #         "data/srsRAN/srsenb1020/zhihu_browse_7060.log",
    #         "data/srsRAN/srsenb1020/qqmusic_standard_7060.log",
    #         "data/srsRAN/srsenb1020/bilibili_1080p_7060.log",
    #         "data/srsRAN/srsenb1020/bilibili_live_7060.log",
    #         "data/srsRAN/srsenb1020/tiktok_browse_7060.log",
    #         "data/srsRAN/srsenb1020/wget_anaconda_7060.log",
    #         "data/srsRAN/srsenb1022/netdisk_upload_7060.log"
    #     ],
    #     labels=[
    #         "tmeeting_video",
    #         "tmeeting_audio",
    #         "fastping",
    #         "zhihu",
    #         "qqmusic",
    #         "bilibili_video",
    #         "bilibili_live",
    #         "tiktok",
    #         "wget_anaconda",
    #         "netdisk_upload"
    #     ],
    #     hybrid_encoder=hybrid_encoder,
    #     label_encoder=label_encoder,
    #     save_path="data/srsRAN/dataset_Xy_7060.npz"
    # )
    # # 8080
    # dataset = SrsRANLteDataset(
    #     params=params,
    #     read_log_paths=[
    #         "data/srsRAN/srsenb1009/qqmusic_standard.log",
    #         "data/srsRAN/srsenb0926/enb_bilibili_1080.log",
    #         "data/srsRAN/srsenb1009/wget_anaconda.log",
    #         "data/srsRAN/srsenb1009/bilibili_live.log",
    #         "data/srsRAN/srsenb1009/tiktok_browse.log",
    #         "data/srsRAN/srsenb1009/tmeeting_video.log",
    #         "data/srsRAN/srsenb1009/tmeeting_audio.log",
    #         "data/srsRAN/srsenb1009/zhihu_browse.log",
    #         "data/srsRAN/srsenb1009/fastping_1721601.log",
    #         "data/srsRAN/srsenb1022/netdisk_upload_8080.log"
    #     ],
    #     labels=[
    #         "qqmusic",
    #         "bilibili_video",
    #         "wget_anaconda",
    #         "bilibili_live",
    #         "tiktok",
    #         "tmeeting_video",
    #         "tmeeting_audio",
    #         "zhihu",
    #         "fastping",
    #         "netdisk_upload"
    #     ],
    #     hybrid_encoder=hybrid_encoder,
    #     label_encoder=label_encoder,
    #     save_path="data/srsRAN/dataset_Xy_8080.npz"
    # )
    # print(label_encoder.classes_)
    # print(hybrid_encoder.channels_minmax_columns)
    # print(hybrid_encoder.channels_onehot_columns)
    # print(hybrid_encoder.channels_embedded_columns)

    # print(dataset.channel_n_components)
    # print(dataset.n_samples_of_classes)

    # """Unit test of SrsRANLteDataLoaders"""
    # dataloaders = SrsRANLteDataLoaders(
    #     params=utils.HyperParams(json_path="experiments/base/params.json"),
    #     read_npz_path="data/srsRAN/srsenb1009/dataset_Xy.npz"
    # )

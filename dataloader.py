import datetime
import json
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
from preprocess import SrsRANLteRecord, AmariNSARecord, SrsRANLteLogFile, AmariNSALogFile


class SrsRANLteHybridEncoder:
    """Embed a list of records, modify embedded_message of each record directly during transformation"""
    def __init__(self):
        self.channels: List[str] = utils.srsRANLte_channels
        self.channels_minmax_columns: List[List[str]] = []
        self.channels_onehot_columns: List[List[str]] = []
        self.channels_minmax_scalers: List[MinMaxScaler] = []
        self.channels_onehot_encoders: List[OneHotEncoder] = []
        self.channels_embedded_columns: List[List[str]] = []
        self.fitted: bool = False
        self._reset()

    def _reset(self):
        self.channels_minmax_columns = [[] for _ in self.channels]
        self.channels_onehot_columns = [[] for _ in self.channels]
        self.channels_minmax_scalers = [MinMaxScaler(clip=False) for _ in self.channels]
        self.channels_onehot_encoders = [
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore") for _ in self.channels
        ]
        self.channels_embedded_columns = [[] for _ in self.channels]
        self.fitted = False

    def fit_embed(self, records: List[SrsRANLteRecord]):
        self._reset()
        for idx, channel in enumerate(self.channels):
            records_channel = [record for record in records if record.basic_info["channel"] == channel]
            df_raw = pd.DataFrame([record.message for record in records_channel])
            df_raw = df_raw.fillna("-1")
            for column in (t := tqdm.tqdm(df_raw.columns)):
                t.set_postfix({"fitting_channel": channel, "column": column})
                try:
                    df_raw[column] = df_raw[column].apply(eval)
                    self.channels_minmax_columns[idx].append(column)
                except (NameError, TypeError, SyntaxError) as _:
                    self.channels_onehot_columns[idx].append(column)
                t.set_postfix({"fitting_channel": channel})
            df_embedded = pd.DataFrame()
            if self.channels_minmax_columns[idx]:
                scaled = pd.DataFrame(
                    self.channels_minmax_scalers[idx].fit_transform(df_raw[self.channels_minmax_columns[idx]])
                )
                df_embedded = pd.concat([df_embedded, scaled], axis=1)
                self.channels_embedded_columns[idx].extend(self.channels_minmax_scalers[idx].get_feature_names_out())
            if self.channels_onehot_columns[idx]:
                encoded = pd.DataFrame(
                    self.channels_onehot_encoders[idx].fit_transform(df_raw[self.channels_onehot_columns[idx]])
                )
                df_embedded = pd.concat([df_embedded, encoded], axis=1)
                self.channels_embedded_columns[idx].extend(self.channels_onehot_encoders[idx].get_feature_names_out())
            df_embedded = df_embedded.to_numpy()
            for index, record in enumerate(records_channel):
                record.embedded_message = df_embedded[index]
        self.fitted = True

    def embed(self, records: List[SrsRANLteRecord]):
        if not self.fitted:
            raise NotFittedError("This SrsRANLteHybridEncoder instance is not fitted yet. ")
        for idx, channel in enumerate(self.channels):
            records_channel = [record for record in records if record.basic_info["channel"] == channel]
            df_raw = pd.DataFrame([record.message for record in records_channel])
            df_raw = df_raw.fillna("-1")
            for minmax_column in (t := tqdm.tqdm(self.channels_minmax_columns[idx])):
                t.set_postfix({"embedding_channel": channel, "minmax_column": minmax_column})
                df_raw[minmax_column] = df_raw[minmax_column].apply(utils.rough_eval)
                t.set_postfix({"embedding_channel": channel})
            df_embedded = pd.DataFrame()
            if self.channels_minmax_columns[idx]:
                scaled = pd.DataFrame(
                    self.channels_minmax_scalers[idx].transform(df_raw[self.channels_minmax_columns[idx]])
                )
                df_embedded = pd.concat([df_embedded, scaled], axis=1)
            if self.channels_onehot_columns[idx]:
                encoded = pd.DataFrame(
                    self.channels_onehot_encoders[idx].transform(df_raw[self.channels_onehot_columns[idx]])
                )
                df_embedded = pd.concat([df_embedded, encoded], axis=1)
            df_embedded = df_embedded.to_numpy()
            for index, record in enumerate(records_channel):
                record.embedded_message = df_embedded[index]

    def get_channel_n_components(self) -> Dict[str, int]:
        if not self.fitted:
            raise NotFittedError("This SrsRANLteHybridEncoder instance is not fitted yet. ")
        return {self.channels[i]: len(self.channels_embedded_columns[i]) for i in range(len(self.channels))}


class SrsRANLteDataset(Dataset):
    def __init__(
            self,
            params: utils.HyperParams,
            read_log_paths: List[str] = None,
            labels: List[str] = None,
            hybrid_encoder: SrsRANLteHybridEncoder = SrsRANLteHybridEncoder(),
            label_encoder: LabelEncoder = LabelEncoder(),
            save_path: str = None,
            read_npz_paths: List[str] = None
    ):
        """
        To load dataset directly from one or multiple npz file: set re_preprocess of params to False and pass a list
        of npz paths to `read_npz_paths` params.

        To preprocess the dataset from raw log files: set re_preprocess of params to True and pass a list of log paths
        to `read_log_paths` with a list of corresponding ground truth labels to `labels`. Pre-fitted feature encoder
        and label encoder can be used by passing encoder instances to `hybrid_encoder` and `label_encoder`.
        """
        if not params.re_preprocess and read_npz_paths:
            self.re_preprocessed: bool = False
            Xy = np.load(read_npz_paths[0])
            self.X: np.ndarray = Xy["X"]
            self.y: np.ndarray = Xy["y"]
            for read_npz_path in read_npz_paths[1:]:
                Xy = np.load(read_npz_path)
                self.X = np.concatenate([self.X, Xy["X"]], axis=0)
                self.y = np.concatenate([self.y, Xy["y"]], axis=0)
        elif read_log_paths and labels:
            if len(read_log_paths) != len(labels):
                raise ValueError(
                    "Found input read log paths and labels with inconsistent numbers: [{}, {}]".format(
                        len(read_log_paths), len(labels)
                    )
                )
            self.re_preprocessed: bool = True
            logfiles: List[SrsRANLteLogFile] = SrsRANLteDataset._construct_logfiles(params, read_log_paths, labels)
            self.hybrid_encoder = hybrid_encoder
            self._embed_features(logfiles)
            self.X: np.ndarray = self._form_dataset_X(logfiles)
            self.label_encoder = label_encoder
            self.y: np.ndarray = self._form_dataset_y(logfiles)
            if save_path:
                np.savez(save_path, X=self.X, y=self.y)
        else:
            raise TypeError("__init__() failed to neither construct nor load dataset")

    @staticmethod
    def _construct_logfiles(
            params: utils.HyperParams,
            read_paths: List[str],
            labels: List[str]
    ) -> List[SrsRANLteLogFile]:
        logfiles: List[SrsRANLteLogFile] = []
        for idx in range(len(read_paths)):
            logfiles.append(SrsRANLteLogFile(
                    read_paths[idx],
                    labels[idx],
                    params.window_size,
                    params.tb_len_threshold
            ))
        return logfiles

    def _embed_features(self, logfiles: List[SrsRANLteLogFile]):
        records: List[SrsRANLteRecord] = [record for logfile in logfiles for record in logfile.records]
        if self.hybrid_encoder.fitted:
            self.hybrid_encoder.embed(records)
        else:
            self.hybrid_encoder.fit_embed(records)

    def _form_dataset_X(self, logfiles: List[SrsRANLteLogFile]) -> np.ndarray:
        channel_n_components: Dict[str, int] = self.hybrid_encoder.get_channel_n_components()
        raw_X: List[np.ndarray] = []
        for logfile in logfiles:
            for sample in logfile.samples:
                raw_X.append(sample.form_sample_X(channel_n_components))
        return np.array(raw_X)

    def _form_dataset_y(self, logfiles: List[SrsRANLteLogFile]) -> np.ndarray:
        raw_y: List[str] = [sample.label for logfile in logfiles for sample in logfile.samples]
        if "classes_" in self.label_encoder.__dict__:
            return self.label_encoder.transform(raw_y)
        else:
            return self.label_encoder.fit_transform(raw_y)

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

    # Unit test of SrsRANLteDataset
    params = utils.HyperParams(json_path="experiments/base/params.json")
    hybrid_encoder = SrsRANLteHybridEncoder()
    label_encoder = LabelEncoder()
    # 6550
    dataset = SrsRANLteDataset(
        params=params,
        read_log_paths=[
            "data/srsRAN/srsenb1020/tmeeting_video_6550.log",
            "data/srsRAN/srsenb1020/tmeeting_audio_6550.log",
            "data/srsRAN/srsenb1020/fastping_1721601_6550.log",
            "data/srsRAN/srsenb1020/zhihu_browse_6550.log",
            "data/srsRAN/srsenb1020/qqmusic_standard_6550.log",
            "data/srsRAN/srsenb1020/bilibili_1080p_6550.log",
            "data/srsRAN/srsenb1020/bilibili_live_6550.log",
            "data/srsRAN/srsenb1020/tiktok_browse_6550.log",
            "data/srsRAN/srsenb1020/wget_anaconda_6550.log",
            "data/srsRAN/srsenb1022/netdisk_upload_6550.log"
        ],
        labels=[
            "tmeeting_video",
            "tmeeting_audio",
            "fastping",
            "zhihu",
            "qqmusic",
            "bilibili_video",
            "bilibili_live",
            "tiktok",
            "wget_anaconda",
            "netdisk_upload"
        ],
        hybrid_encoder=hybrid_encoder,
        label_encoder=label_encoder,
        save_path="data/srsRAN/dataset_Xy_6550.npz"
    )
    with open("hybrid_encoder", 'wb') as f:
        pickle.dump(hybrid_encoder, f)
    with open("label_encoder", 'wb') as f:
        pickle.dump(label_encoder, f)
    # 6040
    dataset = SrsRANLteDataset(
        params=params,
        read_log_paths=[
            "data/srsRAN/srsenb1018-2/qqmusic_standard_6040.log",
            "data/srsRAN/srsenb1018-2/bilibili_480p_6040.log",
            "data/srsRAN/srsenb1018-2/wget_anaconda_6040.log",
            "data/srsRAN/srsenb1018-2/bilibili_live_6040.log",
            "data/srsRAN/srsenb1018-2/tmeeting_video_6040.log",
            "data/srsRAN/srsenb1018-2/tmeeting_audio_6040.log",
            "data/srsRAN/srsenb1018-2/zhihu_6040.log",
            "data/srsRAN/srsenb1018-2/fastping_1721601_6040.log",
        ],
        labels=[
            "qqmusic",
            "bilibili_video",
            "wget_anaconda",
            "bilibili_live",
            "tmeeting_video",
            "tmeeting_audio",
            "zhihu",
            "fastping"
        ],
        hybrid_encoder=hybrid_encoder,
        label_encoder=label_encoder,
        save_path="data/srsRAN/dataset_Xy_6040.npz"
    )
    # 7060
    dataset = SrsRANLteDataset(
        params=params,
        read_log_paths=[
            "data/srsRAN/srsenb1020/tmeeting_video_7060.log",
            "data/srsRAN/srsenb1020/tmeeting_audio_7060.log",
            "data/srsRAN/srsenb1020/fastping_1721601_7060.log",
            "data/srsRAN/srsenb1020/zhihu_browse_7060.log",
            "data/srsRAN/srsenb1020/qqmusic_standard_7060.log",
            "data/srsRAN/srsenb1020/bilibili_1080p_7060.log",
            "data/srsRAN/srsenb1020/bilibili_live_7060.log",
            "data/srsRAN/srsenb1020/tiktok_browse_7060.log",
            "data/srsRAN/srsenb1020/wget_anaconda_7060.log",
            "data/srsRAN/srsenb1022/netdisk_upload_7060.log"
        ],
        labels=[
            "tmeeting_video",
            "tmeeting_audio",
            "fastping",
            "zhihu",
            "qqmusic",
            "bilibili_video",
            "bilibili_live",
            "tiktok",
            "wget_anaconda",
            "netdisk_upload"
        ],
        hybrid_encoder=hybrid_encoder,
        label_encoder=label_encoder,
        save_path="data/srsRAN/dataset_Xy_7060.npz"
    )
    # 8080
    dataset = SrsRANLteDataset(
        params=params,
        read_log_paths=[
            "data/srsRAN/srsenb1009/qqmusic_standard.log",
            "data/srsRAN/srsenb0926/enb_bilibili_1080.log",
            "data/srsRAN/srsenb1009/wget_anaconda.log",
            "data/srsRAN/srsenb1009/bilibili_live.log",
            "data/srsRAN/srsenb1009/tiktok_browse.log",
            "data/srsRAN/srsenb1009/tmeeting_video.log",
            "data/srsRAN/srsenb1009/tmeeting_audio.log",
            "data/srsRAN/srsenb1009/zhihu_browse.log",
            "data/srsRAN/srsenb1009/fastping_1721601.log",
            "data/srsRAN/srsenb1022/netdisk_upload_8080.log"
        ],
        labels=[
            "qqmusic",
            "bilibili_video",
            "wget_anaconda",
            "bilibili_live",
            "tiktok",
            "tmeeting_video",
            "tmeeting_audio",
            "zhihu",
            "fastping",
            "netdisk_upload"
        ],
        hybrid_encoder=hybrid_encoder,
        label_encoder=label_encoder,
        save_path="data/srsRAN/dataset_Xy_8080.npz"
    )
    print(label_encoder.classes_)
    print(hybrid_encoder.channels_minmax_columns)
    print(hybrid_encoder.channels_onehot_columns)
    print(hybrid_encoder.channels_embedded_columns)

    # print(dataset.channel_n_components)
    # print(dataset.n_samples_of_classes)

    # """Unit test of SrsRANLteDataLoaders"""
    # dataloaders = SrsRANLteDataLoaders(
    #     params=utils.HyperParams(json_path="experiments/base/params.json"),
    #     read_npz_path="data/srsRAN/srsenb1009/dataset_Xy.npz"
    # )

    print("END")

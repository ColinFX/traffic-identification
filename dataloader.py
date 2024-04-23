import json
import os.path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

import utils
from preprocess import LogFile


class HybridEncoder:
    def __init__(self, channels: List[str] = None):
        self.channels: List[str] = channels if channels is not None else utils.srsRANLte_channels
        self.channels_minmax_columns: List[List[str]] = [[] for _ in self.channels]
        self.channels_onehot_columns: List[List[str]] = [[] for _ in self.channels]
        self.channels_minmax_scalers: List[utils.AdvancedMinMaxScaler] = []
        self.channels_onehot_encoders: List[utils.AdvancedOneHotEncoder] = []
        self.got_metadata: bool = False
        self.is_fit: bool = False
        self._reset_estimators()

    def _raise_raw_type_error(self):
        raise TypeError(
            "Input raw should be one of following formats: list of pkl paths, single pkl path, list of Logfile "
            "objects, or single LogFile object."
        )

    def _collect_columns_metadata_from_logfile(self, logfile: LogFile):
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

    def collect_columns_metadata(self, raw: List[str] or str or List[LogFile] or LogFile):
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
            elif isinstance(raw, list) and isinstance(raw[0], LogFile):
                for logfile in tqdm.tqdm(raw, postfix={"step": "get_metadata"}):
                    self._collect_columns_metadata_from_logfile(logfile)
            elif isinstance(raw, LogFile):
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

    def _form_raw_dataframe(self, logfile: LogFile, channel_idx: int) -> pd.DataFrame:
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

    def _fit_logfile(self, logfile: LogFile):
        for channel_idx, channel in enumerate(self.channels):
            df_raw = self._form_raw_dataframe(logfile, channel_idx)
            if self.channels_minmax_columns[channel_idx]:
                self.channels_minmax_scalers[channel_idx].fit(df_raw[self.channels_minmax_columns[channel_idx]])
            if self.channels_onehot_columns[channel_idx]:
                self.channels_onehot_encoders[channel_idx].fit(
                    df_raw[self.channels_onehot_columns[channel_idx]]
                )

    def fit(self, raw: List[str] or str or List[LogFile] or LogFile):
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
            elif isinstance(raw, list) and isinstance(raw[0], LogFile):
                for logfile in tqdm.tqdm(raw, postfix={"step": "fit_estimators"}):
                    self._fit_logfile(logfile)
            elif isinstance(raw, LogFile):
                self._fit_logfile(raw)
            else:
                self._raise_raw_type_error()

    def _transform_logfile(self, logfile: LogFile) -> Tuple[np.ndarray, np.ndarray]:
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
            logfile: LogFile,
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
            raw: List[str] or str or List[LogFile] or LogFile,
            save_pkl_paths: List[str] or str = None,
            save_npz_paths: List[str] or str = None
    ):
        """
        Compute `embedded_message` for each record and `x` for each sample in all logfiles.
        Save structured X with labels in new npz files and overwrite original pkl files if input is in format pkl
        path(s) or if input is in format logfile(s) and save paths_84 are given.
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
            elif isinstance(raw, list) and isinstance(raw[0], LogFile):
                for logfile_idx in tqdm.tqdm(range(len(raw)), postfix={"step": "transform_samples"}):
                    self._transform_logfile_and_save(
                        raw[logfile_idx],
                        save_pkl_paths[logfile_idx] if save_pkl_paths else None,
                        save_npz_paths[logfile_idx] if save_npz_paths else None
                    )
            elif isinstance(raw, LogFile):
                self._transform_logfile_and_save(raw, save_pkl_paths, save_npz_paths)
            else:
                self._raise_raw_type_error()


class AwarenessDataset(Dataset):
    def __init__(self, **kwargs):
        self.X: np.ndarray = np.empty(0)
        self.y: np.ndarray = np.empty(0)
        self.labels: np.ndarray = np.empty(0)

    def encode_labels(self, label_encoder: LabelEncoder):
        self.y = label_encoder.transform(self.labels)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.tensor(self.X[idx], dtype=torch.float32), int(self.y[idx])


class SrsranDataset(AwarenessDataset):
    """Read data from npz files containing `X` and `labels`"""
    def __init__(
            self,
            read_solitary_npz_paths: Optional[List[str]] = None,
            read_shared_npz_paths: Optional[List[str]] = None,
            shared_split_percentage_start: Optional[float] = None,
            shared_split_percentage_end: Optional[float] = None,
            label_mapping: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        if read_solitary_npz_paths:
            for read_solitary_npz_path in read_solitary_npz_paths:
                self._add_npz_file(read_solitary_npz_path)
        if read_shared_npz_paths:
            for read_shared_npz_path in read_shared_npz_paths:
                self._add_npz_file(read_shared_npz_path, shared_split_percentage_start, shared_split_percentage_end)
        if label_mapping:
            self.labels = np.array([label_mapping[label] for label in self.labels])

    def _add_npz_file(
            self,
            read_npz_path: str,
            shared_split_percentage_start: Optional[float] = None,
            shared_split_percentage_end: Optional[float] = None
    ):
        npz_dict = np.load(read_npz_path)
        logfile_X: np.ndarray = npz_dict["X"]
        logfile_labels: np.ndarray = npz_dict["labels"]
        if shared_split_percentage_start is not None and shared_split_percentage_end is not None:
            split_start = max(int(shared_split_percentage_start * logfile_X.shape[0]), 0)
            split_end = min(int(shared_split_percentage_end * logfile_X.shape[0]), logfile_X.shape[0])
            logfile_X = logfile_X[split_start:split_end]
            logfile_labels = logfile_labels[split_start:split_end]
        if not self.X.size > 0:
            self.X = logfile_X
            self.labels = logfile_labels
        else:
            self.X = np.concatenate([self.X, logfile_X], axis=0)
            self.labels = np.concatenate([self.labels, logfile_labels], axis=0)


class AmarisoftDataset(AwarenessDataset):
    def __init__(
            self,
            read_npz_paths: Optional[List[str]] = None,
            solitary_labels: Optional[List[str]] = None,
            shared_labels: Optional[List[str]] = None,
            shared_split_percentage_start: Optional[float] = None,
            shared_split_percentage_end: Optional[float] = None,
            label_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Read data from npz files containing `X` and `labels`.
        `solitary_labels` and `shared_labels` are labels before label mapping.
        `shared_split_percentage_start` and `shared_split_percentage_end` required if `shared_labels` given.
        """
        super().__init__()
        t = tqdm.tqdm(read_npz_paths)
        for read_npz_path in t:
            t.set_postfix({"step": "concatenate_Xy", "npz_path": read_npz_path})
            self._add_npz_file(
                read_npz_path,
                solitary_labels,
                shared_labels,
                shared_split_percentage_start,
                shared_split_percentage_end
            )
        if label_mapping:
            self.labels = np.array([label_mapping[label] for label in self.labels])

    def _add_by_label_mask(
            self,
            logfile_X: np.ndarray,
            logfile_labels: np.ndarray,
            label_mask: np.ndarray
    ):
        if np.sum(label_mask):
            if not self.X.size > 0:
                self.X = logfile_X[label_mask]
                self.labels = logfile_labels[label_mask]
            else:
                self.X = np.concatenate([self.X, logfile_X[label_mask]], axis=0)
                self.labels = np.concatenate([self.labels, logfile_labels[label_mask]], axis=0)

    def _add_npz_file(
            self,
            read_npz_path: str,
            solitary_labels: Optional[List[str]] = None,
            shared_labels: Optional[List[str]] = None,
            shared_split_percentage_start: Optional[float] = None,
            shared_split_percentage_end: Optional[float] = None
    ):
        npz_dict = np.load(read_npz_path)
        logfile_X: np.ndarray = npz_dict["X"]
        logfile_labels: np.ndarray = npz_dict["labels"]
        if solitary_labels:
            for solitary_label in solitary_labels:
                label_mask = np.equal(logfile_labels.reshape(-1), solitary_label)
                self._add_by_label_mask(logfile_X, logfile_labels, label_mask)
        if shared_labels:
            for shared_label in shared_labels:
                label_indices = [idx for idx in range(len(logfile_labels)) if logfile_labels[idx] == shared_label]
                start_count = int(shared_split_percentage_start*len(label_indices))
                end_count = int(shared_split_percentage_end*len(label_indices))
                label_indices = label_indices[start_count:end_count]
                label_mask = np.array([idx in label_indices for idx in range(len(logfile_labels))])
                self._add_by_label_mask(logfile_X, logfile_labels, label_mask)


class AwarenessDataLoaders:
    def __init__(
            self,
            label_encoder: Optional[LabelEncoder] = None,
            **kwargs
    ):
        self.label_encoder = label_encoder

    def save_label_encoder(self, save_pkl_path: str):
        with open(save_pkl_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

    def save_npz(self, save_npz_path: str):
        np.savez(
            save_npz_path,
            X_train=self.train_dataset.X if hasattr(self, "train_dataset") else None,
            X_val=self.val_dataset.X if hasattr(self, "val_dataset") else None,
            X_test=self.test_dataset.X if hasattr(self, "test_dataset") else None,
            y_train=self.train_dataset.y if hasattr(self, "train_dataset") else None,
            y_val=self.val_dataset.y if hasattr(self, "val_dataset") else None,
            y_test=self.test_dataset.y if hasattr(self, "test_dataset") else None
        )


class SrsranDataLoaders(AwarenessDataLoaders):
    def __init__(
            self,
            params: utils.HyperParams,
            split_percentages: Optional[List[float]] = None,
            read_train_npz_paths: Optional[List[str]] = None,
            read_val_test_npz_paths: Optional[List[str]] = None,
            read_train_val_test_npz_paths: Optional[List[str]] = None,
            label_mapping: Optional[Dict[str, str]] = None,
            label_encoder: Optional[LabelEncoder] = None,
            save_npz_path: Optional[str] = None
    ):
        """
        Split percentages given in params can be overwritten by passing percentages to `split_percentages`.
        """
        super().__init__(label_encoder)
        if read_train_val_test_npz_paths and (read_train_npz_paths or read_val_test_npz_paths):
            raise ValueError("Unexpected dataloader scenario.")

        if not split_percentages or len(split_percentages) != 3:
            split_percentages = [
                (1 - params.split_val_percentage - params.split_test_percentage),
                params.split_val_percentage,
                params.split_test_percentage
            ]
        else:
            split_percentages = [split_percentage/sum(split_percentages) for split_percentage in split_percentages]

        all_labels = np.empty(0)
        if read_train_npz_paths or (read_train_val_test_npz_paths and split_percentages[0] > 0):
            self.train_dataset = SrsranDataset(
                read_solitary_npz_paths=read_train_npz_paths,
                read_shared_npz_paths=read_train_val_test_npz_paths,
                shared_split_percentage_start=0.,
                shared_split_percentage_end=split_percentages[0],
                label_mapping=label_mapping
            )
            all_labels = np.concatenate([all_labels, self.train_dataset.labels])
        if split_percentages[1] > 0 and (read_train_val_test_npz_paths or read_val_test_npz_paths):
            if read_train_val_test_npz_paths:
                self.val_dataset = SrsranDataset(
                    read_solitary_npz_paths=None,
                    read_shared_npz_paths=read_train_val_test_npz_paths,
                    shared_split_percentage_start=split_percentages[0],
                    shared_split_percentage_end=split_percentages[0]+split_percentages[1],
                    label_mapping=label_mapping
                )
            elif read_val_test_npz_paths:
                self.val_dataset = SrsranDataset(
                    read_solitary_npz_paths=None,
                    read_shared_npz_paths=read_val_test_npz_paths,
                    shared_split_percentage_start=0.,
                    shared_split_percentage_end=split_percentages[1] / (split_percentages[1] + split_percentages[2]),
                    label_mapping=label_mapping
                )
            all_labels = np.concatenate([all_labels, self.val_dataset.labels])
        if split_percentages[2] > 0 and (read_train_val_test_npz_paths or read_val_test_npz_paths):
            if read_train_val_test_npz_paths:
                self.test_dataset = SrsranDataset(
                    read_solitary_npz_paths=None,
                    read_shared_npz_paths=read_train_val_test_npz_paths,
                    shared_split_percentage_start=split_percentages[0]+split_percentages[1],
                    shared_split_percentage_end=1.,
                    label_mapping=label_mapping
                )
            elif read_val_test_npz_paths:
                self.test_dataset = SrsranDataset(
                    read_solitary_npz_paths=None,
                    read_shared_npz_paths=read_val_test_npz_paths,
                    shared_split_percentage_start=split_percentages[1] / (split_percentages[1] + split_percentages[2]),
                    shared_split_percentage_end=1.,
                    label_mapping=label_mapping
                )
            all_labels = np.concatenate([all_labels, self.test_dataset.labels])
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_labels)

        if hasattr(self, "train_dataset"):
            self.train_dataset.encode_labels(self.label_encoder)
            self.train = DataLoader(self.train_dataset, params.batch_size, shuffle=True)
        if hasattr(self, "val_dataset"):
            self.val_dataset.encode_labels(self.label_encoder)
            self.val = DataLoader(self.val_dataset, params.batch_size, shuffle=False)
        if hasattr(self, "test_dataset"):
            self.test_dataset.encode_labels(self.label_encoder)
            self.test = DataLoader(self.test_dataset, params.batch_size, shuffle=False)

        if save_npz_path:
            self.save_npz(save_npz_path)


class AmarisoftDataLoaders(AwarenessDataLoaders):
    def __init__(
            self,
            params: utils.HyperParams,
            split_percentages: Optional[List[float]] = None,
            read_npz_paths: Optional[List[str]] = None,
            train_solitary_labels: Optional[List[str]] = None,
            val_test_solitary_labels: Optional[List[str]] = None,
            train_val_test_shared_labels: Optional[List[str]] = None,
            label_mapping: Optional[Dict[str, str]] = None,
            label_encoder: Optional[LabelEncoder] = None,
            save_npz_path: Optional[str] = None
    ):
        """Get train, validation and test dataloader"""
        super().__init__(label_encoder)
        if train_val_test_shared_labels and (train_solitary_labels or val_test_solitary_labels):
            raise ValueError("Unexpected dataloader scenario.")

        if not split_percentages or len(split_percentages) != 3:
            split_percentages = [
                (1 - params.split_val_percentage - params.split_test_percentage),
                params.split_val_percentage,
                params.split_test_percentage
            ]
        else:
            split_percentages = [split_percentage/sum(split_percentages) for split_percentage in split_percentages]

        all_labels = np.empty(0)
        if train_solitary_labels or (train_val_test_shared_labels and split_percentages[0] > 0):
            self.train_dataset = AmarisoftDataset(
                read_npz_paths=read_npz_paths,
                solitary_labels=train_solitary_labels,
                shared_labels=train_val_test_shared_labels,
                shared_split_percentage_start=0.,
                shared_split_percentage_end=split_percentages[0],
                label_mapping=label_mapping
            )
            all_labels = np.concatenate([all_labels, self.train_dataset.labels])
        if split_percentages[1] > 0 and (train_val_test_shared_labels or val_test_solitary_labels):
            if train_val_test_shared_labels:
                self.val_dataset = AmarisoftDataset(
                    read_npz_paths=read_npz_paths,
                    solitary_labels=None,
                    shared_labels=train_val_test_shared_labels,
                    shared_split_percentage_start=split_percentages[0],
                    shared_split_percentage_end=split_percentages[0]+split_percentages[1],
                    label_mapping=label_mapping
                )
            elif val_test_solitary_labels:
                self.val_dataset = AmarisoftDataset(
                    read_npz_paths=read_npz_paths,
                    solitary_labels=None,
                    shared_labels=val_test_solitary_labels,
                    shared_split_percentage_start=0.,
                    shared_split_percentage_end=split_percentages[1] / (split_percentages[1] + split_percentages[2]),
                    label_mapping=label_mapping
                )
            all_labels = np.concatenate([all_labels, self.val_dataset.labels])
        if split_percentages[2] > 0 and (train_val_test_shared_labels or val_test_solitary_labels):
            if train_val_test_shared_labels:
                self.test_dataset = AmarisoftDataset(
                    read_npz_paths=read_npz_paths,
                    solitary_labels=None,
                    shared_labels=train_val_test_shared_labels,
                    shared_split_percentage_start=split_percentages[0]+split_percentages[1],
                    shared_split_percentage_end=1.,
                    label_mapping=label_mapping
                )
            elif val_test_solitary_labels:
                self.test_dataset = AmarisoftDataset(
                    read_npz_paths=read_npz_paths,
                    solitary_labels=None,
                    shared_labels=val_test_solitary_labels,
                    shared_split_percentage_start=split_percentages[1] / (split_percentages[1] + split_percentages[2]),
                    shared_split_percentage_end=1.,
                    label_mapping=label_mapping
                )
            all_labels = np.concatenate([all_labels, self.test_dataset.labels])
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_labels)

        if hasattr(self, "train_dataset"):
            self.train_dataset.encode_labels(self.label_encoder)
            self.train = DataLoader(self.train_dataset, params.batch_size, shuffle=True)
        if hasattr(self, "val_dataset"):
            self.val_dataset.encode_labels(self.label_encoder)
            self.val = DataLoader(self.val_dataset, params.batch_size, shuffle=False)
        if hasattr(self, "test_dataset"):
            self.test_dataset.encode_labels(self.label_encoder)
            self.test = DataLoader(self.test_dataset, params.batch_size, shuffle=False)

        if save_npz_path:
            self.save_npz(save_npz_path)


if __name__ == "__main__":
    import utils
    from preprocess import *
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    timetable = {
        ((datetime.time(12,  2,  0), datetime.time(12,  9, 32)), "live-74"),
        ((datetime.time(12,  9, 46), datetime.time(12, 16, 42)), "video-74"),
        ((datetime.time(12, 17, 30), datetime.time(12, 20,  0)), "meet-74"),
        ((datetime.time(12, 20,  1), datetime.time(12, 23, 10)), "call-74"),
        ((datetime.time(12, 23, 17), datetime.time(12, 25, 41)), "down-74"),
        ((datetime.time(12, 26,  3), datetime.time(12, 40, 43)), "up-74"),
        ((datetime.time(12, 40, 43), datetime.time(12, 42, 53)), "tiktok-74"),

        ((datetime.time(12, 56, 30), datetime.time(12, 59, 29)), "live-84"),
        ((datetime.time(12, 59, 43), datetime.time(13,  7, 55)), "video-84"),
        ((datetime.time(13,  8,  7), datetime.time(13, 11,  7)), "meet-84"),
        ((datetime.time(13, 11, 16), datetime.time(13, 14, 24)), "call-84"),
        ((datetime.time(13, 14, 33), datetime.time(13, 19, 18)), "down-84"),
        ((datetime.time(13, 20,  5), datetime.time(13, 35, 14)), "up-84"),
        ((datetime.time(13, 35, 33), datetime.time(13, 39, 44)), "tiktok-84"),

        ((datetime.time(14,  0, 54), datetime.time(14,  5, 20)), "live-68"),
        ((datetime.time(14,  5, 43), datetime.time(14,  9,  5)), "video-68"),
        ((datetime.time(14,  9, 13), datetime.time(14, 14, 21)), "meet-68"),
        ((datetime.time(14, 14, 23), datetime.time(14, 17, 45)), "call-68"),
        ((datetime.time(14, 17, 53), datetime.time(14, 20,  6)), "down-68"),
        ((datetime.time(14, 20, 17), datetime.time(14, 23, 20)), "up-68"),
        ((datetime.time(14, 24, 15), datetime.time(14, 27, 38)), "tiktok-68"),

        ((datetime.time(14, 46, 33), datetime.time(14, 52, 37)), "live-80"),
        ((datetime.time(14, 52, 56), datetime.time(14, 56, 33)), "video-80"),
        ((datetime.time(14, 58, 26), datetime.time(15,  1,  2)), "meet-80"),
        ((datetime.time(15,  1,  4), datetime.time(15,  6,  7)), "call-80"),
        ((datetime.time(15,  6, 12), datetime.time(15,  8, 38)), "down-80"),
        ((datetime.time(15, 11, 40), datetime.time(15, 22, 19)), "up-80"),
        ((datetime.time(15, 24,  3), datetime.time(15, 28, 14)), "tiktok-80"),

        ((datetime.time(17, 45,  3), datetime.time(17, 48, 33)), "live-70"),
        ((datetime.time(17, 48, 34), datetime.time(17, 50, 41)), "video-70"),
        ((datetime.time(17, 51, 19), datetime.time(17, 54, 38)), "meet-70"),
        ((datetime.time(17, 54, 39), datetime.time(17, 58,  0)), "call-70"),
        ((datetime.time(17, 59, 51), datetime.time(18,  0, 36)), "down-70"),
        ((datetime.time(18,  1,  3), datetime.time(18,  5, 32)), "up-70"),
        ((datetime.time(18,  6, 12), datetime.time(18, 10,  4)), "tiktok-70"),

        ((datetime.time(18, 16, 50), datetime.time(18, 23,  1)), "live-72"),
        ((datetime.time(18, 23, 21), datetime.time(18, 25, 51)), "video-72"),
        ((datetime.time(18, 27,  5), datetime.time(18, 29, 37)), "meet-72"),
        ((datetime.time(18, 29, 57), datetime.time(18, 33,  3)), "call-72"),
        ((datetime.time(18, 34, 18), datetime.time(18, 35, 36)), "down-72"),
        ((datetime.time(18, 36, 37), datetime.time(18, 41, 43)), "up-72"),
        ((datetime.time(18, 43,  2), datetime.time(18, 47, 13)), "tiktok-72"),

        ((datetime.time(18, 50, 13), datetime.time(18, 55,  0)), "live-78"),
        ((datetime.time(18, 55,  0), datetime.time(18, 59, 12)), "video-78"),
        ((datetime.time(18, 59, 22), datetime.time(19,  1, 44)), "meet-78"),
        ((datetime.time(19,  1, 46), datetime.time(19,  5, 55)), "call-78"),
        ((datetime.time(19,  6,  0), datetime.time(19,  8, 26)), "down-78"),
        ((datetime.time(19,  9,  2), datetime.time(19, 17,  0)), "up-78"),
        ((datetime.time(19, 17, 48), datetime.time(19, 20, 55)), "tiktok-78"),

        ((datetime.time(19, 38, 19), datetime.time(19, 41, 34)), "live-76"),
        ((datetime.time(19, 42, 30), datetime.time(19, 46, 38)), "video-76"),
        ((datetime.time(19, 47, 15), datetime.time(19, 50, 48)), "meet-76"),
        ((datetime.time(19, 50, 51), datetime.time(19, 53, 49)), "call-76"),
        ((datetime.time(19, 54,  1), datetime.time(19, 56, 26)), "down-76"),
        ((datetime.time(19, 57,  2), datetime.time(20,  4,  4)), "up-76"),
        ((datetime.time(20,  5,  1), datetime.time(20,  8, 55)), "tiktok-76"),

        ((datetime.time(20, 11,  5), datetime.time(20, 17, 15)), "live-82"),
        ((datetime.time(20, 17, 24), datetime.time(20, 22, 10)), "video-82"),
        ((datetime.time(20, 22, 10), datetime.time(20, 25, 58)), "meet-82"),
        ((datetime.time(20, 26,  4), datetime.time(20, 29, 56)), "call-82"),
        ((datetime.time(20, 30,  0), datetime.time(20, 34,  9)), "down-82"),
        ((datetime.time(20, 35,  2), datetime.time(20, 47, 59)), "up-82"),
        ((datetime.time(20, 48,  4), datetime.time(20, 51, 57)), "tiktok-82"),

        ((datetime.time(21,  0, 16), datetime.time(21,  5, 47)), "live-66"),
        ((datetime.time(21,  5, 49), datetime.time(21, 10, 18)), "video-66"),
        ((datetime.time(21, 10, 46), datetime.time(21, 15, 40)), "meet-66"),
        ((datetime.time(21, 15, 46), datetime.time(21, 20, 53)), "call-66"),
        ((datetime.time(21, 21,  2), datetime.time(21, 23, 32)), "down-66"),
        ((datetime.time(21, 24,  1), datetime.time(21, 29,  2)), "up-66"),
        ((datetime.time(21, 29,  2), datetime.time(21, 35,  8)), "tiktok-66")
    }

    label_m = {}
    for gain in [66, 68, 70, 72, 74, 76, 78, 80, 82, 84]:
        for app in ["live", "video", "meet", "call", "down", "up", "tiktok"]:
            label_m[app + "-" + str(gain)] = app

    # load
    dataloaders = AmarisoftDataLoaders(
        params=utils.HyperParams("experiments/base/params.json"),
        split_percentages=[0.7, 0, 0.3],
        read_npz_paths=["data/NR/SA/20240412/gnb0412(2).npz"],
        train_solitary_labels=None,
        val_test_solitary_labels=None,
        train_val_test_shared_labels=["live-74", "video-74"],
        label_mapping=label_m,
        label_encoder=None,
        save_npz_path="data/NR/SA/20240411/Xy.npz1"
    )
    dataloaders.save_label_encoder("data/NR/SA/20240411/label_encoder.pickle")

    X_train = dataloaders.train_dataset.X.reshape(dataloaders.train_dataset.X.shape[0], -1)
    y_train = dataloaders.train_dataset.y
    X_test = dataloaders.test_dataset.X.reshape(dataloaders.test_dataset.X.shape[0], -1)
    y_test = dataloaders.test_dataset.y

    # train
    model = lgb.LGBMClassifier(random_state=17)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    print(
        accuracy_score(y_true=y_test, y_pred=y_test_pred),
        precision_score(y_true=y_test, y_pred=y_test_pred, average="macro"),
        recall_score(y_true=y_test, y_pred=y_test_pred, average="macro"),
        f1_score(y_true=y_test, y_pred=y_test_pred, average="macro")
    )
    print(confusion_matrix(y_true=y_test, y_pred=y_test_pred))

    # lm = {}
    # for gain in [66, 69, 72, 75, 78, 81, 84]:
    #     for app in ["bililive", "bilivideo", "netdisk", "tmeetingaudio", "tmeetingvideo", "wget"]:
    #         lm[app + str(gain)] = app
    #         lm[app + str(gain) + "_10"] = app
    #
    # params = utils.HyperParams(json_path="./experiments/base/params.json")
    #
    # paths_84 = utils.listdir_with_suffix("data/srsRAN/srsenb0219", ".npz")
    # paths_84 = [path for path in paths_84 if "84_10" in path]
    # paths_81 = utils.listdir_with_suffix("data/srsRAN/srsenb0219", ".npz")
    # paths_81 = [path for path in paths_81 if "81_10" in path]
    #
    # dataloaders = SrsranDataLoaders(
    #     params=params,
    #     split_percentages=[0.6, 0.2, 0.2],
    #     read_train_npz_paths=paths_84,
    #     read_val_test_npz_paths=paths_81,
    #     label_mapping=lm,
    # )
    #
    # import lightgbm as lgb
    # model = lgb.LGBMClassifier()
    # model.fit(dataloaders.train_dataset.X.reshape(-1, 590), dataloaders.train_dataset.y)
    # y_val_pred = model.predict(dataloaders.val_dataset.X.reshape(-1, 590))
    # from sklearn.metrics import accuracy_score
    # print(accuracy_score(y_true=dataloaders.val_dataset.y, y_pred=y_val_pred))

    # hybrid_encoder = SrsRANLteHybridEncoder()
    # hybrid_encoder.collect_columns_metadata(data_folder)
    # hybrid_encoder.load_columns_metadata("data/srsRAN/srsenb0219/columns_metadata.json")
    # hybrid_encoder.save_columns_metadata(data_folder)
    # hybrid_encoder.fit(data_folder)
    # with open(os.path.join(data_folder, "hybrid_encoder.pickle"), "wb") as file:
    #     pickle.dump(hybrid_encoder, file)
    # with open("data/srsRAN/srsenb0219/hybrid_encoder.pickle", "rb") as file:
    #     hybrid_encoder = pickle.load(file)
    # hybrid_encoder.transform(data_folder)

    # with open("tmp/hybrid_encoder.pkl", "rb") as file:
    #     hybrid_encoder = pickle.load(file)
    # hybrid_encoder.transform(data_folder)

    # """Unit test of AmarisoftDataset"""
    # params = utils.HyperParams(json_path="../experiments/base/params.json")
    # params.re_preprocess = True
    # dl = AmarisoftDataLoaders(
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

    # """Unit test of AmarisoftDataLoaders"""
    # params = utils.HyperParams(json_path="experiments/base/params.json")
    # dataloaders = AmarisoftDataLoaders(
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

    # # Unit test of SrsranDataset
    # params = utils.HyperParams(json_path="experiments/base/params.json")
    # hybrid_encoder = SrsRANLteHybridEncoder()
    # label_encoder = LabelEncoder()
    # # 6550
    # dataset = SrsranDataset(
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
    # dataset = SrsranDataset(
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
    # dataset = SrsranDataset(
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
    # dataset = SrsranDataset(
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

    # """Unit test of SrsranDataLoaders"""
    # dataloaders = SrsranDataLoaders(
    #     params=utils.HyperParams(json_path="experiments/base/params.json"),
    #     read_npz_path="data/srsRAN/srsenb1009/dataset_Xy.npz"
    # )

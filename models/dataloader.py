from typing import Dict, List, Tuple

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

import utils


class TCDataset(Dataset):
    def __init__(self, read_path: str, params: utils.HyperParams):
        super().__init__()
        self.dataframe: pd.DataFrame = pd.read_csv(read_path)
        self.dataframes: List[pd.DataFrame] = self._separate_channels(
            ["PDCCH", "PDSCH", "PHICH", "PRACH", "PUCCH", "PUSCH", "SRS"]
        )
        self._refine_columns()
        self._refine_rows()
        self.samples: List[torch.Tensor, str] = self._segment_samples()
        self.encoded_samples: List[torch.Tensor, torch.Tensor] = self._encode_labels()


    def _separate_channels(self, channels: List[str]) -> List[pd.DataFrame]:
        """Separate records of different channels in multiple dataframes"""
        dataframes: List[pd.DataFrame] = []
        for channel in channels:
            dataframes.append(self.dataframe[self.dataframe["channel"] == channel])
        return dataframes

    def _refine_columns(self):
        """Remove columns with no enough meaningful data from dataframes for each channel"""
        for idx in range(len(self.dataframes)):
            self.dataframes[idx] = self.dataframes[idx].dropna(
                axis=1,
                thresh=int(0.1*len(self.dataframes[idx]))
            )  # 0.1 CONFIGURABLE

    def _refine_rows(self):
        """Remove rows with only essential information from dataframes for each channel"""
        for idx in range(len(self.dataframes)):
            non_essential_columns = -1 # TODO

    def _segment_samples(self) -> List[torch.Tensor, str]:
        pass  # TODO

    def _encode_labels(self) -> List[torch.Tensor, torch.Tensor]:
        """One-hot-key encode labels of each sample"""
        pass  # TODO

    def __len__(self):
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pass


class TCDataLoader(DataLoader):
    def __init__(self, mode: str):
        super().__init__()

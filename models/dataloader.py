import datetime
import json
import os.path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
from sklearn.preprocessing import LabelEncoder

import utils
from preprocess import GNBLogFile


class GNBDataset(Dataset):
    def __init__(
            self,
            read_paths: List[str],
            feature_path: str,
            timetables: List[List[Tuple[Tuple[datetime.time, datetime.time], str]]],
            window_size: int = 1,
            tb_len_threshold: int = 150,
            save_path: str = None
    ):
        """Read log from multiple files and generate generalized dataset (X,y) for ML/DL models"""
        self.feature_map: Dict[str, Dict[str, List[str]]] = utils.get_feature_map(feature_path)
        self.window_size = window_size
        self.logfiles: List[GNBLogFile] = self._construct_logfiles(
            read_paths,
            timetables,
            tb_len_threshold
        )
        self._embed_features()
        self.label_encoder = LabelEncoder()
        self.X: np.ndarray = self._form_dataset_X()
        self.y: np.ndarray = self._form_dataset_y()
        if save_path:
            self._save_Xy(save_path)

    def _construct_logfiles(
            self,
            read_paths: List[str],
            timetables: List[List[Tuple[Tuple[datetime.time, datetime.time], str]]],
            tb_len_threshold: int
    ):
        """Read all logfiles from paths in the given list"""
        logfiles: List[GNBLogFile] = []
        for idx in (t := tqdm.trange(len(read_paths))):
            t.set_postfix({"read_path": "\""+read_paths[idx]+"\""})
            logfiles.append(GNBLogFile(
                    read_paths[idx],
                    self.feature_map,
                    timetables[idx],
                    self.window_size,
                    tb_len_threshold
            ))
        return logfiles

    def _embed_features(self):
        """Processing key_info vector to pure numeric, NAIVE APPROACH"""
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

    def _form_dataset_X(self) -> np.ndarray:
        """Assemble combined vector for each sample as input to ML/DL models"""
        raw_X: List[np.ndarray] = []
        for logfile in self.logfiles:
            for sample in logfile.samples:
                raw_X.append(sample.form_sample_X(self.feature_map))
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
        np.savez(save_path, X=self.X, y=self.y)

    def plot_channel_statistics(self):
        """Plot bar chart of channel statistics in labelled timezone before sampling, CONFIG ONLY"""
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
        print("\nTag combinations of different physical layer channels: ")
        for channel in self.feature_map.keys():
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


class GNBDataLoader(DataLoader):
    pass


if __name__ == "__main__":
    """Unit test of GNBDataset"""
    dataset = GNBDataset(
        read_paths=["../data/NR/1st-example/gnb0.log"],
        feature_path="../experiments/base/features.json",
        timetables=[[
            ((datetime.time(9, 48, 20), datetime.time(9, 58, 40)), "navigation_web"),
            ((datetime.time(10, 1, 40), datetime.time(10, 13, 20)), "streaming_youtube")
        ]],
        window_size=1,
        tb_len_threshold=150,
        save_path="../data/NR/1st-example/dataset_Xy.npz"
    )
    dataset.plot_channel_statistics()
    dataset.plot_tb_len_statistics()
    dataset.count_feature_combinations()

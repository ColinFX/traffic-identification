import abc
import csv
import datetime
import json
import os.path
import re
from typing import Dict, List, Match, Pattern, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.preprocessing import LabelEncoder

import utils


class GNBRecord:
    def __init__(self, raw_record: List[str]):
        self.label: str = ""
        self.raw_record = raw_record
        match: Match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s\[([A-Z0-9]+)]', raw_record[0])
        self.time: datetime.time = datetime.datetime.strptime(match.groups()[0], "%H:%M:%S.%f").time()
        self.layer: str = match.groups()[1]
        self.basic_info: Dict[str, str] = self._extract_basic_info()
        self.short_message: Dict[str, str] = self._extract_short_message()
        self.long_message: Dict[str, str] = self._extract_long_message()
        self.key_info: List[str or float or int] = []
        self._reformat_prb_symb()

    @abc.abstractmethod
    def _extract_basic_info(self) -> Dict[str, str]:
        return {}

    @abc.abstractmethod
    def _extract_short_message(self) -> Dict[str, str]:
        return {}

    @abc.abstractmethod
    def _extract_long_message(self) -> Dict[str, str]:
        return {}

    def _reformat_prb_symb(self):
        for keyword in ["prb", "symb"]:
            if keyword in self.short_message.keys():
                if "," in self.short_message[keyword]:
                    pairs = self.short_message[keyword].split(",")
                else:
                    pairs = [self.short_message[keyword]]
                keyword_start: int = 101
                keyword_end: int = -1
                keyword_len: int = 0
                for pair in pairs:
                    if ":" in pair:
                        start, len_ = map(int, pair.split(':'))
                    else:
                        start = int(pair)
                        len_ = 1
                    keyword_start = min(keyword_start, start)
                    keyword_end = max(keyword_end, start + len_)
                    keyword_len += len_
                self.short_message[keyword+"_start"] = str(keyword_start)
                self.short_message[keyword+"_end"] = str(keyword_end)
                self.short_message[keyword+"_len"] = str(keyword_len)

    def get_record_label(self, timetable: List[Tuple[Tuple[datetime.time, datetime.time], str]]) -> str:
        """Get ground truth label from given `timetable` for one `record`"""
        for range_, label in timetable:
            if range_[0] <= self.time < range_[1]:
                return label
        return "noise"


class GNBRecordPHY(GNBRecord):
    def __init__(self, raw_record: List[str]):
        super().__init__(raw_record)

    def _extract_basic_info(self) -> Dict[str, str]:
        match: Match = re.match(
            r'\S+\s+\[\S+]\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\.(\S+)\s+(\S+):',
            self.raw_record[0]
        )
        keys = ["dir", "ue_id", "cell_id", "rnti", "frame", "subframe", "channel"]
        return dict(zip(keys, match.groups()))

    def _extract_short_message(self) -> Dict[str, str]:
        short_message_str: str = self.raw_record[0].split(':', 1)[1]
        if "CW1" in short_message_str:
            short_message_str = short_message_str.split("CW1", 1)[0]
        return dict(re.findall(r"(\S+)=(\S+)", short_message_str))

    def _extract_long_message(self) -> Dict[str, str]:
        long_message_str: str = " ".join(self.raw_record[1:])
        return dict(re.findall(r"(\S+)=(\S+)", long_message_str))


class GNBRecordRLC(GNBRecord):
    def __init__(self, raw_record: List[str]):
        super().__init__(raw_record)

    def _extract_basic_info(self) -> Dict[str, str]:
        match: Match = re.match(r'\S+\s+\[\S+]\s+(\S+)\s+(\S+)\s+(\S+)', self.raw_record[0])
        keys = ["dir", "ue_id", "bearer"]
        return dict(zip(keys, match.groups()))

    def _extract_short_message(self) -> Dict[str, str]:
        return dict(re.findall(r"(\S+)=(\S+)", self.raw_record[0]))

    def _extract_long_message(self) -> Dict[str, str]:
        return {}


class GNBRecordGTPU(GNBRecord):
    def __init__(self, raw_record: List[str]):
        super().__init__(raw_record)

    def _extract_basic_info(self) -> Dict[str, str]:
        match: Match = re.match(r'\S+\s+\[\S+]\s+(\S+)\s(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)', self.raw_record[0])
        keys = ["dir", "ip", "port"]
        return dict(zip(keys, match.groups()))

    def _extract_short_message(self) -> Dict[str, str]:
        short_message: Dict[str, str] = dict(re.findall(r"(\S+)=(\S+)", self.raw_record[0]))
        match: Match = re.match(
            r'.* (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)\s+>\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)',
            self.raw_record[0]
        )
        if match:
            keys = ["source_ip", "source_port", "destination_ip", "destination_port"]
            short_message.update(dict(zip(keys, match.groups())))
        return short_message

    def _extract_long_message(self) -> Dict[str, str]:
        long_message: Dict[str, str] = {}
        for line in self.raw_record[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                long_message[key] = value
        return long_message


class GNBSample:
    def __init__(
            self,
            records: List[GNBRecord],
            period: int,
            frame_cycle: int,
            window_size: int
    ):
        self.records = records
        self.period = period
        self.frame_cycle = frame_cycle
        self.window_size = window_size
        self.tb_len: int = self._count_tb_len()
        self.label: str = self._get_sample_label()

    def _count_tb_len(self) -> int:
        """Calculate sum of tb_len of records in one sample as amount of data transmitted"""
        tb_len_sum: int = 0
        for record in self.records:
            if "tb_len" in record.short_message.keys():
                tb_len_sum += int(record.short_message["tb_len"])
        return tb_len_sum

    def _get_sample_label(self) -> str:
        """Get label for each newly formed sample by majority voting of records"""
        voting: Dict[str, int] = {}
        for record in self.records:
            if record.label in voting.keys():
                voting[record.label] += 1
            else:
                voting[record.label] = 1
        return max(voting, key=voting.get)

    def form_sample_X(self, feature_map: Dict[str, Dict[str, List[str]]]) -> np.ndarray:
        """Construct array as direct input to ML/DL models, use only after all features are numerical, NAIVE APPROACH"""
        raw_X: List[List[int or float]] = []
        for frame in range(self.frame_cycle * self.window_size, (self.frame_cycle+1) * self.window_size):
            for subframe in range(20):
                raw_X_subframe: List[int or float] = []
                for cell_id in ["03", "04"]:
                    for channel in feature_map.keys():
                        channel_in_subframe_flag = False
                        for record in self.records:
                            if (
                                not channel_in_subframe_flag and
                                record.basic_info["channel"] == channel and
                                record.basic_info["cell_id"] == cell_id and
                                int(record.basic_info["frame"]) == frame and
                                int(record.basic_info["subframe"]) == subframe
                            ):
                                raw_X_subframe.extend(record.key_info)
                                channel_in_subframe_flag = True
                        if not channel_in_subframe_flag:
                            raw_X_subframe.extend([-1] * sum([len(value) for value in feature_map[channel].values()]))
                raw_X.append(raw_X_subframe)
        return np.array(raw_X)


class GNBLogFile:
    def __init__(
            self,
            read_path: str,
            save_path: str,
            feature_map: Dict[str, Dict[str, List[str]]],
            timetable: List[Tuple[Tuple[datetime.time, datetime.time], str]],
            window_size: int,
            tb_len_threshold: int
    ):
        """Read log from `read_path` and save preprocessed physical layer data records for ML/DL models"""
        with open(read_path, 'r') as f:
            self.lines: List[str] = f.readlines()
        self.date: datetime.date = self._process_header()
        self.raw_records: List[List[str]] = self._group_lines()
        self.records: List[GNBRecord] = [self._reformat_record(raw_record) for raw_record in self.raw_records]
        self._filter_phy_drb_records()
        self._sort_records()
        self._add_record_labels(timetable)
        self._extract_key_features(feature_map)
        self.export_json(save_path)
        self.samples: List[GNBSample] = self._regroup_records(window_size)
        self._filter_samples(tb_len_threshold)

    def _process_header(self) -> datetime.date:
        """Remove header marked by `#` and get date"""
        i: int = 0
        while i < len(self.lines) and (self.lines[i].startswith('#')):
            i += 1
        date_str: str = re.search(r'\d{4}-\d{2}-\d{2}', self.lines[i-1]).group()
        date: datetime.date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        self.lines = self.lines[i:]
        return date

    def _group_lines(self) -> List[List[str]]:
        """Group lines into blocks as `raw_records` with plain text"""
        raw_records: List[List[str]] = []
        pattern: Pattern = re.compile(r'\d{2}:\d{2}:\d{2}\.\d{3} \[[A-Z0-9]+]')
        current_record: List[str] = []
        for line in self.lines:
            if pattern.match(line):
                if current_record:
                    raw_records.append(current_record)
                current_record = [line]
            else:
                current_record.append(line)
        if current_record:
            raw_records.append(current_record)
        return raw_records

    @staticmethod
    def _reformat_record(raw_record: List[str]) -> GNBRecord:
        """Convert `raw_record` with plain text into `GNBRecord` instance"""
        if "[PHY]" in raw_record[0]:
            return GNBRecordPHY(raw_record)
        elif "[RLC]" in raw_record[0]:
            return GNBRecordRLC(raw_record)
        elif "[GTPU]" in raw_record[0]:
            return GNBRecordGTPU(raw_record)
        else:
            return GNBRecord(raw_record)

    def _filter_phy_drb_records(self):
        """Keep only data records of physical layer"""
        filtered_records: List[GNBRecord] = []
        drb_flag: bool = False
        for record in self.records:
            if record.layer == "RLC":
                if "DRB" in record.basic_info["bearer"]:
                    drb_flag = True
                elif "SRB" in record.basic_info["bearer"]:
                    drb_flag = False
                else:
                    print("ERROR")
                    print(record.time, record.layer, record.basic_info)
            elif record.layer == "PHY":
                if drb_flag:
                    filtered_records.append(record)
            else:
                pass
        self.records = filtered_records

    def _sort_records(self):
        """Sort physical layer records in period-frame-subframe order"""
        current_period: int = 0
        last_frame: int = -1
        for record in self.records:
            if int(record.basic_info["frame"]) - last_frame < -100:  # -100 CONFIGURABLE
                current_period += 1
            record.basic_info["period"] = str(current_period)
            last_frame = int(record.basic_info["frame"])
        self.records.sort(
            key=lambda record: (
                int(record.basic_info["period"]),
                int(record.basic_info["frame"]),
                int(record.basic_info["subframe"])
            )
        )

    def _add_record_labels(
            self,
            timetable: List[Tuple[Tuple[datetime.time, datetime.time], str]],
            delete_noise: bool = True
    ):
        """Add ground truth label for all `records` by given `timetable`, delete records without label if requested"""
        for idx, record in enumerate(self.records):
            label = record.get_record_label(timetable)
            record.label = label
            if delete_noise and not record.label:
                del self.records[idx]

    def _extract_key_features(self, feature_map: Dict[str, Dict[str, List[str]]]):
        """Extract wanted features to key_info list in raw data types for each physical layer record"""
        for record in self.records:
            key_info: List[str or float or int] = []
            if record.basic_info["channel"] in feature_map.keys():
                for feature in feature_map[record.basic_info["channel"]]["basic_info"]:
                    if feature in record.basic_info.keys():
                        key_info.append(record.basic_info[feature])
                    else:
                        key_info.append(-1)
                for feature in feature_map[record.basic_info["channel"]]["short_message"]:
                    if feature in record.short_message.keys():
                        key_info.append(record.short_message[feature])
                    else:
                        key_info.append(-1)
                for feature in feature_map[record.basic_info["channel"]]["long_message"]:
                    if feature in record.long_message.keys():
                        key_info.append(record.long_message[feature])
                    else:
                        key_info.append(-1)
            record.key_info = key_info
        return feature_map

    def export_json(self, save_path: str):
        """Save physical layer records with label to json file, CONFIG ONLY"""
        with open(save_path, 'w') as f:
            for record in self.records:
                json.dump(vars(record), f, indent=4, default=str)
                f.write("\n")

    def export_csv(self, save_path: str):
        """Save physical layer records with label to csv file, CONFIG ONLY"""
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            keys_basic_info: List[str] = list(set().union(*[obj.basic_info.keys() for obj in self.records]))
            keys_short_message: List[str] = list(set().union(*[obj.short_message.keys() for obj in self.records]))
            keys_long_message: List[str] = list(set().union(*[obj.long_message.keys() for obj in self.records]))
            writer.writerow(["label", "time", "layer"] + keys_basic_info + keys_short_message + keys_long_message)
            for record in self.records:
                if record.label:
                    row = [record.label, record.time, record.layer]
                    for key in keys_basic_info:
                        row.append(record.basic_info.get(key, np.nan))
                    for key in keys_short_message:
                        row.append(record.short_message.get(key, np.nan))
                    for key in keys_long_message:
                        row.append(record.long_message.get(key, np.nan))
                    writer.writerow(row)

    def _regroup_records(self, window_size: int) -> List[GNBSample]:
        """Form samples by fixed window size (number of frames, recommended to be power of 2)"""
        samples: List[GNBSample] = []
        current_period = -1
        current_frame_cycle = -1
        current_sample_records: List[GNBRecord] = []
        for record in self.records:
            if (
                int(record.basic_info["period"]) == current_period and
                int(record.basic_info["frame"]) // window_size == current_frame_cycle
            ):
                current_sample_records.append(record)
            else:
                if current_sample_records:
                    samples.append(GNBSample(current_sample_records, current_period, current_frame_cycle, window_size))
                current_sample_records = [record]
                current_period = int(record.basic_info["period"])
                current_frame_cycle = int(record.basic_info["frame"]) // window_size

        if current_sample_records:
            samples.append(GNBSample(current_sample_records, current_period, current_frame_cycle, window_size))
        return samples

    def _filter_samples(self, threshold: int):
        """Keep only samples with enough data transmitted and meaningful label"""
        filtered_samples: List[GNBSample] = []
        for sample in self.samples:
            if sample.tb_len >= threshold and sample.label != "noise":
                filtered_samples.append(sample)
        self.samples = filtered_samples


class GNBDataset:
    def __init__(
            self,
            read_paths: List[str],
            save_paths: List[str],
            feature_path: str,
            timetables: List[List[Tuple[Tuple[datetime.time, datetime.time], str]]],
            window_size: int = 1,
            tb_len_threshold: int = 150
    ):
        """Read log from multiple files and generate generalized dataset (X,y) for """
        self.feature_map: Dict[str, Dict[str, List[str]]] = utils.get_feature_map(feature_path)
        self.window_size = window_size
        self.logfiles: List[GNBLogFile] = self._construct_logfiles(
            read_paths,
            save_paths,
            timetables,
            tb_len_threshold
        )
        self._embed_features()
        self.label_encoder = LabelEncoder()
        self.X: np.ndarray = self._form_dataset_X()
        self.y: np.ndarray = self._form_dataset_y()

    def _construct_logfiles(
            self,
            read_paths: List[str],
            save_paths: List[str],
            timetables: List[List[Tuple[Tuple[datetime.time, datetime.time], str]]],
            tb_len_threshold: int
    ):
        """Read all logfiles from paths in the given list"""
        logfiles: List[GNBLogFile] = []
        for idx in (t := tqdm.trange(len(read_paths))):
            t.set_postfix({"read_path": "\""+read_paths[idx]+"\""})
            logfiles.append(
                GNBLogFile(
                    read_paths[idx],
                    save_paths[idx],
                    self.feature_map,
                    timetables[idx],
                    self.window_size,
                    tb_len_threshold
                )
            )
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

    def save_Xy(self, save_dir: str):
        np.savez(os.path.join(save_dir, "dataset_Xy.npz"), X=self.X, y=self.y)

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


if __name__ == "__main__":
    """Unit test of preprocessing"""
    dataset = GNBDataset(
        read_paths=["data/NR/1st-example/gnb0.log"],
        save_paths=["data/NR/1st-example/export.json"],
        feature_path="experiments/base/features.json",
        timetables=[[
            ((datetime.time(9, 48, 20), datetime.time(9, 58, 40)), "navigation_web"),
            ((datetime.time(10, 1, 40), datetime.time(10, 13, 20)), "streaming_youtube")
        ]],
        window_size=1,
        tb_len_threshold=150
    )
    dataset.save_Xy(save_dir="data/NR/1st-example")
    dataset.plot_channel_statistics()
    dataset.plot_tb_len_statistics()
    dataset.count_feature_combinations()

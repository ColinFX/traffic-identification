import abc
import csv
import datetime
import json
import re
from typing import Dict, List, Match, Pattern, Tuple

import numpy as np
import tqdm

import utils


class SRSENBRecord:
    def __init__(self, raw_record: List[str]):
        match: Match = re.match(
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})\s\[([A-Z0-9]+)\s*]\s\[([A-Z])]\s(.*)',
            raw_record[0]
        )
        self.datetime: datetime.datetime = datetime.datetime.fromisoformat(match.groups()[0])
        self.layer: str = match.groups()[1]
        self.log_level: str = match.groups()[2]
        self.basic_info: Dict[str, str]
        self.message: Dict[str, str]
        self.basic_info, self.message = self._extract_info_message(match.groups()[3])
        self._reformat_values()
        self.label: str = ""
        self.embedded_info: np.ndarray = np.empty([])

    @staticmethod
    @abc.abstractmethod
    def _extract_info_message(raw_info: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        return {}, {}

    def _reformat_values(self):
        for key in self.message.copy().keys():
            if key == "rb":
                match_rb: Match = re.match(r'\((\d+),(\d+)\)', self.message[key])
                self.message["rb_start"] = match_rb.groups()[0]
                self.message["rb_end"] = match_rb.groups()[1]
                self.message["rb_len"] = str(int(match_rb.groups()[1]) - int(match_rb.groups()[0]))
                del self.message["rb"]
            else:
                match_unit: Match = re.match(r'([+\-.\d]+)\s([a-zA-Z]+)', self.message[key])
                if match_unit:
                    self.message[key] = match_unit.groups()[0]
                match_brace: Match = re.match(r'\{(.+)}', self.message[key])
                if match_brace:
                    self.message[key] = match_brace.groups()[0]

    def get_record_label(self, timetable: List[Tuple[Tuple[datetime.datetime, datetime.datetime], str]]) -> str:
        for range_, label in timetable:
            if range_[0] <= self.datetime < range_[1]:
                return label
        return ""


class SRSENBRecordPHY(SRSENBRecord):
    @staticmethod
    def _extract_info_message(raw_info: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        basic_info: Dict[str, str] = {}
        message: Dict[str, str] = {}
        match: Match = re.match(r'\[\s*(\d+)]\s([A-Z]+):\s(.*)', raw_info)
        basic_info["subframe"] = match.groups()[0]
        basic_info["channel"] = match.groups()[1]
        remaining_string = match.groups()[2].replace(';', ',')
        remaining_string = re.sub(r'\s\(cc=\d\)', '', remaining_string)
        parts = remaining_string.split(', ')
        for part in parts:
            sub_parts = part.split("=")
            if len(sub_parts) == 2:
                if sub_parts[0] in ["cc", "rnti"]:
                    basic_info[sub_parts[0]] = sub_parts[1]
                else:
                    message[sub_parts[0]] = sub_parts[1]
        return basic_info, message


class SRSENBRecordRLC(SRSENBRecord):
    @staticmethod
    def _extract_info_message(raw_info: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        if "DRB" in raw_info:
            return {"type": "DRB"}, {}
        elif "SRB" in raw_info:
            return {"type": "SRB"}, {}
        else:
            return {"type": ""}, {}


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
        self.message = self.short_message.copy()
        self.message.update(self.long_message)
        self.key_info: List[str or float or int] = []
        self.embedded_info: np.ndarray = np.empty([])
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
        return ""


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


class SRSENBSample:
    def __init__(
            self,
            records: List[SRSENBRecord],
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
        tb_len_sum: int = 0
        for record in self.records:
            if "tbs" in record.message.keys():
                tb_len_sum += int(record.message["tbs"])
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

    def form_sample_X(self, channel_n_components: Dict[str, int]):
        raw_X: List[List[int or float]] = []
        for subframe in range(
                self.frame_cycle * self.window_size * 10,
                (self.frame_cycle+1) * self.window_size * 10
        ):
            raw_X_subframe: List[float] = []
            for channel in channel_n_components.keys():
                channel_in_subframe_flag = False
                for record in self.records:
                    if (
                        record.basic_info["channel"] == channel and
                        int(record.basic_info["subframe"]) == subframe
                    ):
                        channel_in_subframe_flag = True
                        raw_X_subframe.extend(record.embedded_info)
                        break
                if not channel_in_subframe_flag:
                    raw_X_subframe.extend([0] * channel_n_components[channel])
            raw_X.append(raw_X_subframe)
        return np.array(raw_X)


class GNBSample:
    def __init__(
            self,
            records: List[GNBRecord],
            period: int,
            frame_cycle: int,
            window_size: int,
            pca_n_components: int
    ):
        self.records = records
        self.period = period
        self.frame_cycle = frame_cycle
        self.window_size = window_size
        self.pca_n_components = pca_n_components
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

    def form_sample_X_naive(self, feature_map: Dict[str, Dict[str, List[str]]]) -> np.ndarray:
        """Construct array as direct input to ML/DL models, use only after all features are numerical, DEPRECATED"""
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

    def form_sample_X(self) -> np.ndarray:
        """Construct array as direct input to ML/DL models, use only after all records are embedded"""
        raw_X: List[List[int or float]] = []
        for frame in range(self.frame_cycle * self.window_size, (self.frame_cycle+1) * self.window_size):
            for subframe in range(10):
                raw_X_subframe: List[float] = []
                for cell_id, slot in [("03", 0), ("04", 0), ("04", 1)]:
                    if cell_id == "04":
                        subframe = 2 * subframe + slot
                    for channel in utils.GNB_cell_channels[cell_id]:
                        channel_in_subframe_flag = False
                        for record in self.records:
                            if (
                                record.basic_info["channel"] == channel and
                                record.basic_info["cell_id"] == cell_id and
                                int(record.basic_info["frame"]) == frame and
                                int(record.basic_info["subframe"]) == subframe
                            ):
                                channel_in_subframe_flag = True
                                raw_X_subframe.extend(record.embedded_info)
                                break
                        if not channel_in_subframe_flag:
                            raw_X_subframe.extend([0] * self.pca_n_components)
                raw_X.append(raw_X_subframe)
        return np.array(raw_X)

    def form_sample_X_CNN(self):
        """Construct image-like array as input to CNN classifier"""
        raw_X: List[List[float]] = []
        for frame in range(self.frame_cycle * self.window_size, (self.frame_cycle+1) * self.window_size):
            for subframe in range(10):
                for cell_id, slot in [("03", 0), ("04", 0), ("04", 1)]:
                    if cell_id == "04":
                        subframe = 2 * subframe + slot
                    raw_X_subframe: List[float] = []
                    for channel in utils.GNB_cell_channels["04"]:
                        channel_in_subframe_flag = False
                        for record in self.records:
                            if (
                                record.basic_info["channel"] == channel and
                                record.basic_info["cell_id"] == cell_id and
                                int(record.basic_info["frame"]) == frame and
                                int(record.basic_info["subframe"]) == subframe
                            ):
                                channel_in_subframe_flag = True
                                raw_X_subframe.extend(record.embedded_info)
                                break
                        if not channel_in_subframe_flag:
                            raw_X_subframe.extend([0] * self.pca_n_components)
                    raw_X.append(raw_X_subframe)
        return np.array(raw_X)


class SRSENBLogFile:
    def __init__(
            self,
            read_path: str,
            timetable: List[Tuple[Tuple[datetime.datetime, datetime.datetime], str]],
            window_size: int,
            tbs_threshold: int
    ):
        with open(read_path, 'r') as f:
            lines: List[str] = f.readlines()
        self.records: List[SRSENBRecord] = []
        t = tqdm.tqdm(lines)
        t.set_postfix({"read_path": read_path})
        for line in t:
            if record := self._reformat_record(line):
                self.records.append(record)
        self.begin_datetime = self.records[0].datetime
        self.end_datetime = self.records[-1].datetime
        self._filter_phy_drb_records()
        self._add_record_periods()
        self._add_record_labels(timetable)
        self._trim_beginning_end()
        self.samples: List[SRSENBSample] = self._regroup_records(window_size)
        self._filter_samples(tbs_threshold)

    @staticmethod
    def _reformat_record(raw_record: str) -> SRSENBRecord or None:
        if "[PHY" in raw_record and ": " in raw_record:
            return SRSENBRecordPHY([raw_record])
        elif "[RLC" in raw_record:
            return SRSENBRecordRLC([raw_record])
        else:
            return None

    def _filter_phy_drb_records(self):
        filtered_records: List[SRSENBRecord] = []
        drb_flag: bool = False
        for record in self.records:
            if "RLC" in record.layer:
                if "DRB" in record.basic_info["type"]:
                    drb_flag = True
                elif "SRB" in record.basic_info["type"]:
                    drb_flag = False
            elif "PHY" in record.layer:
                if drb_flag:
                    filtered_records.append(record)
        self.records = filtered_records

    def _add_record_periods(self):
        current_period = 0
        last_subframe = 0
        for record in self.records:
            if int(record.basic_info["subframe"]) < last_subframe:
                current_period += 1
            record.basic_info["period"] = str(current_period)
            last_subframe = int(record.basic_info["subframe"])

    def _add_record_labels(
            self,
            timetable: List[Tuple[Tuple[datetime.datetime, datetime.datetime], str]],
            delete_noise: bool = True
    ):
        for idx, record in enumerate(self.records):
            label = record.get_record_label(timetable)
            record.label = label
            if delete_noise and not record.label:
                del self.records[idx]

    def _trim_beginning_end(
            self,
            delta: datetime.timedelta = datetime.timedelta(seconds=60)
    ):
        trimmed_records: List[SRSENBRecord] = []
        for record in self.records:
            if self.begin_datetime + delta < record.datetime < self.end_datetime - delta:
                trimmed_records.append(record)
        self.records = trimmed_records

    def _regroup_records(self, window_size: int) -> List[SRSENBSample]:
        samples: List[SRSENBSample] = []
        current_period = -1
        current_frame_cycle = -1
        current_sample_records: List[SRSENBRecord] = []
        for record in self.records:
            if (
                int(record.basic_info["period"]) == current_period and
                int(record.basic_info["subframe"]) // 10 // window_size == current_frame_cycle
            ):
                current_sample_records.append(record)
            else:
                if current_sample_records:
                    samples.append(
                        SRSENBSample(current_sample_records, current_period, current_frame_cycle, window_size)
                    )
                current_sample_records = [record]
                current_period = int(record.basic_info["period"])
                current_frame_cycle = int(record.basic_info["subframe"]) // 10 // window_size
        if current_sample_records:
            samples.append(SRSENBSample(current_sample_records, current_period, current_frame_cycle, window_size))
        return samples

    def _filter_samples(self, threshold: int):
        filtered_samples: List[SRSENBSample] = []
        for sample in self.samples:
            if sample.tb_len >= threshold and sample.label:
                filtered_samples.append(sample)
        self.samples = filtered_samples


class GNBLogFile:
    def __init__(
            self,
            read_path: str,
            feature_map: Dict[str, Dict[str, List[str]]],
            timetable: List[Tuple[Tuple[datetime.time, datetime.time], str]],
            window_size: int,
            pca_n_components: int,
            tb_len_threshold: int
    ):
        """
        Read log from `read_path` and save preprocessed physical layer data records for ML/DL models,
        feature_map param DEPRECATED
        """
        with open(read_path, 'r') as f:
            self.lines: List[str] = f.readlines()
        self.date: datetime.date = self._process_header()
        self.raw_records: List[List[str]] = self._group_lines()
        self.records: List[GNBRecord] = [self._reformat_record(raw_record) for raw_record in self.raw_records]
        self._filter_phy_drb_records()
        self._sort_records()
        self._add_record_labels(timetable)
        self.samples: List[GNBSample] = self._regroup_records(window_size, pca_n_components)
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
        """Extract wanted features to key_info list in raw data types for each physical layer record, DEPRECATED"""
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

    def _regroup_records(self, window_size: int, pca_n_components: int) -> List[GNBSample]:
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
                    samples.append(GNBSample(current_sample_records, current_period, current_frame_cycle, window_size, pca_n_components))
                    # TODO: pass everything by params
                current_sample_records = [record]
                current_period = int(record.basic_info["period"])
                current_frame_cycle = int(record.basic_info["frame"]) // window_size

        if current_sample_records:
            samples.append(GNBSample(current_sample_records, current_period, current_frame_cycle, window_size, pca_n_components))
        return samples

    def _filter_samples(self, threshold: int):
        """Keep only samples with enough data transmitted and meaningful label"""
        filtered_samples: List[GNBSample] = []
        for sample in self.samples:
            if sample.tb_len >= threshold and sample.label:
                filtered_samples.append(sample)
        self.samples = filtered_samples

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


if __name__ == "__main__":
    # """Unit test of GNBLogFile"""
    # logfile = GNBLogFile(
    #     read_path="data/NR/1st-example/gnb0.log",
    #     feature_map=utils.get_feature_map("experiments/base/features.json"),
    #     timetable=[
    #         ((datetime.time(9, 48, 20), datetime.time(9, 58, 40)), "navigation_web"),
    #         ((datetime.time(10, 1, 40), datetime.time(10, 13, 20)), "streaming_youtube")
    #     ],
    #     window_size=1,
    #     pca_n_components=4,
    #     tb_len_threshold=150
    # )
    # logfile.export_json(save_path="data/NR/1st-example/export.json")
    # logfile.export_csv(save_path="data/NR/1st-example/export.csv")

    """Unit test of SRSENBLogFile"""
    logfile = SRSENBLogFile(
        read_path="data/srsRAN/srsenb0926/enb_ping_1721601.log",
        timetable=[(
            (
                (datetime.datetime(2023, 9, 26, 0, 0)),
                (datetime.datetime(2023, 9, 26, 23, 59))
            ),
            "ping"
        )],
        window_size=1,
        tbs_threshold=100
    )
    channel_count = {}
    for record in logfile.records:
        if record.basic_info["channel"] in channel_count:
            channel_count[record.basic_info["channel"]] += 1
        else:
            channel_count[record.basic_info["channel"]] = 1
    print(channel_count)
    print("END")

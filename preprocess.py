import abc
import csv
import datetime
import json
import re
from typing import Dict, List, Match, Pattern, Tuple, Optional

import numpy as np
import pickle
import tqdm

import utils


class Record:
    def __init__(self):
        self.layer: str = ""
        self.basic_info: Dict[str, str] = {}
        self.message: Dict[str, str] = {}
        self.label: str = ""
        self.embedded_message: np.ndarray = np.empty([])


class SrsranRecord(Record):
    def __init__(self, raw_record: List[str]):
        super().__init__()
        match: Match = re.match(
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})\s\[([A-Z0-9\-]+)\s*]\s\[([A-Z])]\s(.*)',
            raw_record[0]
        )
        self.datetime: datetime.datetime = datetime.datetime.fromisoformat(match.groups()[0])
        self.layer: str = match.groups()[1]
        self.basic_info, self.message = self._extract_info_message(match.groups()[3])
        self._reformat_values()

    @staticmethod
    @abc.abstractmethod
    def _extract_info_message(raw_info: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        pass

    def _reformat_values(self):
        for key in self.message.copy().keys():
            # process rb, ex. "rb=(2,4)"
            if key == "rb":
                match_rb: Match = re.match(r'\((\d+),(\d+)\)', self.message[key])
                self.message["rb_start"] = match_rb.groups()[0]
                self.message["rb_end"] = match_rb.groups()[1]
                self.message["rb_len"] = str(int(match_rb.groups()[1]) - int(match_rb.groups()[0]))
                del self.message["rb"]
            else:
                # remove units, ex. "snr=4.1 dB"
                match_unit: Match = re.match(r'([+\-.\d]+)\s([a-zA-Z]+)', self.message[key])
                if match_unit:
                    self.message[key] = match_unit.groups()[0]
                # remove braces, ex. "tbs={32}"
                match_brace: Match = re.match(r'\{(.+)}', self.message[key])
                if match_brace:
                    self.message[key] = match_brace.groups()[0]


class SrsranRecordPHY(SrsranRecord):
    @staticmethod
    def _extract_info_message(raw_info: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        basic_info: Dict[str, str] = {}
        message: Dict[str, str] = {}
        match: Match = re.match(r'\[\s*(\d+)]\s([A-Z]+):\s(.*)', raw_info)
        basic_info["subframe"] = match.groups()[0]
        basic_info["channel"] = match.groups()[1]
        # remove duplicated cc info, ex. "cqi=0 (cc=0)"
        remaining_string = re.sub(r'\s\(cc=\d\)', '', match.groups()[2])
        parts = re.split(r", |; ", remaining_string)
        for part in parts:
            sub_parts = part.split("=")
            if len(sub_parts) == 2:
                if sub_parts[0] in ["cc", "rnti"]:
                    basic_info[sub_parts[0]] = sub_parts[1]
                else:
                    message[sub_parts[0]] = sub_parts[1]
        return basic_info, message


class SrsranRecordRLC(SrsranRecord):
    @staticmethod
    def _extract_info_message(raw_info: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        if "DRB" in raw_info:
            return {"type": "DRB"}, {}
        elif "SRB" in raw_info:
            return {"type": "SRB"}, {}
        else:
            return {"type": ""}, {}


class AmarisoftRecord(Record):
    def __init__(self, raw_record: List[str]):
        super().__init__()
        match: Match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s\[([A-Z0-9]+)]', raw_record[0])
        self.time: datetime.time = datetime.datetime.strptime(match.groups()[0], "%H:%M:%S.%f").time()
        self.layer: str = match.groups()[1]
        self.basic_info: Dict[str, str] = self._extract_basic_info(raw_record)
        self.message: Dict[str, str] = self._extract_short_message(raw_record)
        self.message.update(self._extract_long_message(raw_record))
        self._reformat_values()

    @staticmethod
    @abc.abstractmethod
    def _extract_basic_info(raw_record: List[str]) -> Dict[str, str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def _extract_short_message(raw_record: List[str]) -> Dict[str, str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def _extract_long_message(raw_record: List[str]) -> Dict[str, str]:
        pass

    def _reformat_values(self):
        for keyword in ["prb", "symb"]:
            # prb and symb sample format: "a:b,c:d"
            if keyword in self.message.keys():
                if "," in self.message[keyword]:
                    pairs = self.message[keyword].split(",")
                else:
                    pairs = [self.message[keyword]]
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
                self.message[keyword+"_start"] = str(keyword_start)
                self.message[keyword+"_end"] = str(keyword_end)
                self.message[keyword+"_len"] = str(keyword_len)
                del self.message[keyword]
        for keyword in ["re_symb", "chan_symb"]:
            # re_symb and chan_symb sample format: a,b,c,d,...
            if keyword in self.message.keys():
                values = self.message[keyword].split(",")
                for idx, value in enumerate(values):
                    self.message[keyword+"_"+str(idx)] = value
                del self.message[keyword]

    def get_record_label(self, timetable: List[Tuple[Tuple[datetime.time, datetime.time], str]]) -> str:
        """Get ground truth label from given `timetable` for one `record`"""
        for range_, label in timetable:
            if range_[0] <= self.time < range_[1]:
                return label
        return ""


class AmarisoftRecordPHY(AmarisoftRecord):
    def __init__(self, raw_record: List[str]):
        super().__init__(raw_record)

    @staticmethod
    def _extract_basic_info(raw_record: List[str]) -> Dict[str, str]:
        match: Match = re.match(
            r'\S+\s+\[\S+]\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\.(\S+)\s+(\S+):',
            raw_record[0]
        )
        keys = ["dir", "ue_id", "cell_id", "rnti", "frame", "subframe", "channel"]
        return dict(zip(keys, match.groups()))

    @staticmethod
    def _extract_short_message(raw_record: List[str]) -> Dict[str, str]:
        short_message_str: str = raw_record[0].split(':', 1)[1]
        if "CW1" in short_message_str:
            short_message_str = short_message_str.split("CW1", 1)[0]
        return dict(re.findall(r"(\S+)=(\S+)", short_message_str))

    @staticmethod
    def _extract_long_message(raw_record: List[str]) -> Dict[str, str]:
        long_message_str: str = " ".join(raw_record[1:])
        return dict(re.findall(r"(\S+)=(\S+)", long_message_str))


class AmarisoftRecordRLC(AmarisoftRecord):
    def __init__(self, raw_record: List[str]):
        super().__init__(raw_record)

    @staticmethod
    def _extract_basic_info(raw_record: List[str]) -> Dict[str, str]:
        match: Match = re.match(r'\S+\s+\[\S+]\s+(\S+)\s+(\S+)\s+(\S+)', raw_record[0])
        keys = ["dir", "ue_id", "bearer"]
        return dict(zip(keys, match.groups()))

    @staticmethod
    def _extract_short_message(raw_record: List[str]) -> Dict[str, str]:
        return dict(re.findall(r"(\S+)=(\S+)", raw_record[0]))

    @staticmethod
    def _extract_long_message(raw_record: List[str]) -> Dict[str, str]:
        return {}


class AmarisoftRecordGTPU(AmarisoftRecord):
    def __init__(self, raw_record: List[str]):
        super().__init__(raw_record)

    @staticmethod
    def _extract_basic_info(raw_record: List[str]) -> Dict[str, str]:
        match: Match = re.match(
            r'\S+\s+\[\S+]\s+(\S+)\s(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)',
            raw_record[0]
        )
        keys = ["dir", "ip", "port"]
        return dict(zip(keys, match.groups()))

    @staticmethod
    def _extract_short_message(raw_record: List[str]) -> Dict[str, str]:
        short_message: Dict[str, str] = dict(re.findall(r"(\S+)=(\S+)", raw_record[0]))
        match: Match = re.match(
            r'.* (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)\s+>\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)',
            raw_record[0]
        )
        if match:
            keys = ["source_ip", "source_port", "destination_ip", "destination_port"]
            short_message.update(dict(zip(keys, match.groups())))
        return short_message

    @staticmethod
    def _extract_long_message(raw_record: List[str]) -> Dict[str, str]:
        long_message: Dict[str, str] = {}
        for line in raw_record[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                long_message[key] = value
        return long_message


class Sample:
    def __init__(
            self,
            records: List[Record],
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
        self.x: np.ndarray = np.empty(0)

    @abc.abstractmethod
    def _count_tb_len(self) -> int:
        pass

    @abc.abstractmethod
    def _get_sample_label(self) -> str:
        pass

    @abc.abstractmethod
    def form_sample_X(self, channels_columns_num: List[int]) -> np.ndarray:
        pass


class SrsranSample(Sample):
    def __init__(
            self,
            records: List[SrsranRecord],
            period: int,
            frame_cycle: int,
            window_size: int
    ):
        super().__init__(records, period, frame_cycle, window_size)

    def _count_tb_len(self) -> int:
        tb_len_sum: int = 0
        for record in self.records:
            if "tbs" in record.message.keys():
                tb_len_sum += int(record.message["tbs"])
        return tb_len_sum

    def _get_sample_label(self) -> str:
        voting: Dict[str, int] = {}
        for record in self.records:
            if record.label in voting.keys():
                voting[record.label] += 1
            else:
                voting[record.label] = 1
        return max(voting, key=voting.get)

    def form_sample_X(self, channels_columns_num: List[int]):
        raw_X: List[List[int or float]] = []
        for subframe in range(
                self.frame_cycle * self.window_size * 10,
                (self.frame_cycle+1) * self.window_size * 10
        ):
            raw_X_subframe: List[float] = []
            for channel_idx, channel in enumerate(utils.srsRANLte_channels):
                channel_in_subframe_flag = False
                for record in self.records:
                    if (
                        record.basic_info["channel"] == channel and
                        int(record.basic_info["subframe"]) == subframe
                    ):
                        channel_in_subframe_flag = True
                        raw_X_subframe.extend(record.embedded_message)
                        break
                if not channel_in_subframe_flag:
                    raw_X_subframe.extend([0] * channels_columns_num[channel_idx])
            raw_X.append(raw_X_subframe)
        return np.array(raw_X)


class AmarisoftSample(Sample):
    def __init__(
            self,
            records: List[AmarisoftRecord],
            period: int,
            frame_cycle: int,
            window_size: int
    ):
        super().__init__(records, period, frame_cycle, window_size)

    def _count_tb_len(self) -> int:
        """Calculate sum of tb_len of records in one sample as amount of data transmitted"""
        tb_len_sum: int = 0
        for record in self.records:
            if "tb_len" in record.message.keys():
                tb_len_sum += int(record.message["tb_len"])
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

    def form_sample_X_NSA(self, feature_map: Dict[str, Dict[str, List[str]]]) -> np.ndarray:
        """
        Construct array as direct input to ML/DL models, use only after all features are numerical.
        This function is not compatible with HybridEncoder.
        """
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
                                raw_X_subframe.extend(record.message)
                                channel_in_subframe_flag = True
                        if not channel_in_subframe_flag:
                            raw_X_subframe.extend([-1] * sum([len(value) for value in feature_map[channel].values()]))
                raw_X.append(raw_X_subframe)
        return np.array(raw_X)

    def form_sample_X(self, channels_columns_num: List[int]) -> np.ndarray:
        """Construct array as direct input to ML/DL models, use only after all records are embedded"""
        raw_X: List[List[int or float]] = []
        for subframe in range(
                self.frame_cycle * self.window_size * 20,
                (self.frame_cycle+1) * self.window_size * 20
        ):
            raw_X_subframe: List[float] = []
            for channel_idx, channel in enumerate(utils.amariSA_channels):
                channel_in_subframe_flag = False
                for record in self.records:
                    if (
                        record.basic_info["channel"] == channel and
                        int(record.basic_info["subframe"]) == subframe
                    ):
                        channel_in_subframe_flag = True
                        raw_X_subframe.extend(record.embedded_message)
                        break
                if not channel_in_subframe_flag:
                    raw_X_subframe.extend([0] * channels_columns_num[channel_idx])
            raw_X.append(raw_X_subframe)
        return np.array(raw_X)


class LogFile:
    def __init__(self):
        self.records: List[AmarisoftRecord or SrsranRecord] = []
        self.samples: List[AmarisoftSample or SrsranSample] = []

    @abc.abstractmethod
    def regroup_records(self, window_size: int) -> List[Sample]:
        pass

    def filter_samples(self, threshold: int):
        """Keep only samples with enough data transmitted and meaningful label"""
        filtered_samples: List[Sample] = []
        for sample in self.samples:
            if sample.tb_len >= threshold and sample.label:
                filtered_samples.append(sample)
        self.samples = filtered_samples

    def form_sample_xs(self, channels_columns_num: List[int]):
        for sample in self.samples:
            sample.x = sample.form_sample_X(channels_columns_num)


class SrsranLogFile(LogFile):
    def __init__(
            self,
            read_path: str,
            label: str,
            window_size: int,
            tbs_threshold: int,
            delta_begin: datetime.timedelta = datetime.timedelta(seconds=60),
            delta_end: datetime.timedelta = datetime.timedelta(seconds=10)
    ):
        super().__init__()
        with open(read_path, 'r') as f:
            lines: List[str] = f.readlines()
        t = tqdm.tqdm(lines)
        t.set_postfix({"read_path": read_path})
        for line in t:
            if record := self._reformat_record(line):
                record.label = label
                self.records.append(record)
        self._filter_phy_drb_records()
        self._add_record_periods()
        self._trim_head_tail(delta_begin, delta_end)
        self.samples: List[SrsranSample] = self.regroup_records(window_size)
        self.filter_samples(tbs_threshold)

    @staticmethod
    def _reformat_record(raw_record: str) -> Optional[SrsranRecord]:
        if "[PHY" in raw_record and "CH: " in raw_record:
            return SrsranRecordPHY([raw_record])
        elif "[RLC" in raw_record:
            return SrsranRecordRLC([raw_record])
        else:
            return None

    def _filter_phy_drb_records(self):
        filtered_records: List[SrsranRecord] = []
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

    def _trim_head_tail(
            self,
            delta_head: datetime.timedelta = datetime.timedelta(seconds=60),
            delta_tail: datetime.timedelta = datetime.timedelta(seconds=10)
    ):
        beginning_datetime = self.records[0].datetime
        end_datetime = self.records[-1].datetime
        trimmed_records: List[SrsranRecord] = []
        for record in self.records:
            if beginning_datetime + delta_head < record.datetime < end_datetime - delta_tail:
                trimmed_records.append(record)
        self.records = trimmed_records

    def regroup_records(self, window_size: int) -> List[SrsranSample]:
        samples: List[SrsranSample] = []
        current_period = -1
        current_frame_cycle = -1
        current_sample_records: List[SrsranRecord] = []
        for record in self.records:
            if (
                int(record.basic_info["period"]) == current_period and
                int(record.basic_info["subframe"]) // 10 // window_size == current_frame_cycle
            ):
                current_sample_records.append(record)
            else:
                if current_sample_records:
                    samples.append(
                        SrsranSample(current_sample_records, current_period, current_frame_cycle, window_size)
                    )
                current_sample_records = [record]
                current_period = int(record.basic_info["period"])
                current_frame_cycle = int(record.basic_info["subframe"]) // 10 // window_size
        if current_sample_records:
            samples.append(SrsranSample(current_sample_records, current_period, current_frame_cycle, window_size))
        return samples

    def get_snr_statistics(self) -> Dict[str, float]:
        """Get mean uplink (from UE to ENB) signal-to-noise ratio, CONFIG ONLY """
        records_snr: List[float] = []
        for sample in self.samples:
            for record in sample.records:
                if "snr" in record.message.keys():
                    records_snr.append(float(record.message["snr"]))
        return {"min": np.min(records_snr), "avg": np.average(records_snr), "max": np.max(records_snr)}

    def get_channel_statistics(self) -> Dict[str, int]:
        """CONFIG ONLY"""
        channel_records: Dict[str, int] = {}
        for record in self.records:
            if record.basic_info["channel"] in channel_records:
                channel_records[record.basic_info["channel"]] += 1
            else:
                channel_records[record.basic_info["channel"]] = 1
        return channel_records

    def get_mcs_statistics(self) -> Dict[int, int]:
        """CONFIG ONLY"""
        mcs_counter: Dict[int, int] = {}
        for record in self.records:
            if "mcs" in record.message.keys():
                mcs_record = int(record.message["mcs"])
                if mcs_record in mcs_counter.keys():
                    mcs_counter[mcs_record] += 1
                else:
                    mcs_counter[mcs_record] = 1
        return dict(sorted(mcs_counter.items()))


class AmarisoftLogFile(LogFile):
    def __init__(
            self,
            read_path: str,
            timetable: List[Tuple[Tuple[datetime.time, datetime.time], str]],
            window_size: int,
            tb_len_threshold: int
    ):
        """Read log from `read_path` and save preprocessed physical layer data records for ML/DL models."""
        super().__init__()
        with open(read_path, 'r') as f:
            lines: List[str] = f.readlines()
        lines, self.date = self._remove_header(lines)
        raw_records: List[List[str]] = self._group_lines(lines)
        t = tqdm.tqdm(raw_records)
        t.set_postfix({"read_path": read_path})
        for raw_record in t:
            if record := self._reformat_record(raw_record):
                self.records.append(record)
        self._filter_phy_drb_records(remove_DRB=False)
        self._sort_records()
        self._add_record_labels(timetable)
        self.samples: List[AmarisoftSample] = self.regroup_records(window_size)
        self.filter_samples(tb_len_threshold)

    @staticmethod
    def _group_lines(lines: List[str]) -> List[List[str]]:
        """Group lines into blocks as `raw_records` with plain text"""
        raw_records: List[List[str]] = []
        pattern: Pattern = re.compile(r'\d{2}:\d{2}:\d{2}\.\d{3} \[[A-Z0-9]+]')
        current_record: List[str] = []
        for line in lines:
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
    def _remove_header(lines: List[str]) -> Tuple[List[str], datetime.date]:
        """Remove header marked by `#` and get date"""
        i: int = 0
        while i < len(lines) and (lines[i].startswith('#')):
            i += 1
        date_str: str = re.search(r'\d{4}-\d{2}-\d{2}', lines[i-1]).group()
        date: datetime.date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        lines = lines[i:]
        return lines, date

    @staticmethod
    def _reformat_record(raw_record: List[str]) -> Optional[AmarisoftRecord]:
        """Convert `raw_record` with plain text into `AmarisoftRecord` instance"""
        if "[PHY]" in raw_record[0]:
            return AmarisoftRecordPHY(raw_record)
        elif "[RLC]" in raw_record[0]:
            return AmarisoftRecordRLC(raw_record)
        elif "[GTPU]" in raw_record[0]:
            return AmarisoftRecordGTPU(raw_record)
        else:
            return None

    def _filter_phy_drb_records(self, remove_DRB: bool = False):
        """Keep only data records of physical layer"""
        filtered_records: List[AmarisoftRecord] = []
        if remove_DRB:
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
        else:
            for record in self.records:
                if record.layer == "PHY":
                    filtered_records.append(record)
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
            key=lambda record: [
                int(record.basic_info["period"]),
                int(record.basic_info["frame"]),
                int(record.basic_info["subframe"])
            ]
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

    def regroup_records(self, window_size: int) -> List[AmarisoftSample]:
        """Form samples by fixed window size (number of frames, recommended to be power of 2)"""
        samples: List[AmarisoftSample] = []
        current_period = -1
        current_frame_cycle = -1
        current_sample_records: List[AmarisoftRecord] = []
        for record in self.records:
            if (
                int(record.basic_info["period"]) == current_period and
                int(record.basic_info["frame"]) // window_size == current_frame_cycle
            ):
                current_sample_records.append(record)
            else:
                if current_sample_records:
                    samples.append(
                        AmarisoftSample(current_sample_records, current_period, current_frame_cycle, window_size)
                    )
                current_sample_records = [record]
                current_period = int(record.basic_info["period"])
                current_frame_cycle = int(record.basic_info["frame"]) // window_size

        if current_sample_records:
            samples.append(AmarisoftSample(current_sample_records, current_period, current_frame_cycle, window_size))
        return samples

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
            keys_message: List[str] = list(set().union(*[obj.message.keys() for obj in self.records]))
            writer.writerow(["label", "time", "layer"] + keys_basic_info + keys_message)
            for record in self.records:
                if record.label:
                    row = [record.label, record.time, record.layer]
                    for key in keys_basic_info:
                        row.append(record.basic_info.get(key, np.nan))
                    for key in keys_message:
                        row.append(record.message.get(key, np.nan))
                    writer.writerow(row)


if __name__ == "__main__":
    # data_folder = "data/srsRAN/srsenb0219"
    # for file_path in utils.listdir_with_suffix(data_folder, ".log"):
    #     label = os.path.splitext(os.path.split(file_path)[1])[0]
    #     logfile = SrsranLogFile(
    #         read_path=file_path,
    #         label=label,
    #         window_size=1,
    #         tbs_threshold=0
    #     )
    #     with open(os.path.join(data_folder, label+".pkl"), "wb") as file:
    #         pickle.dump(logfile, file)

    # # Unit test of AmarisoftLogFile
    # logfile = AmarisoftLogFile(
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

    # # Unit test of SrsranLogFile
    # lf = SrsranLogFile(
    #     read_path="data/srsRAN/srsenb0219/bililive66.log",
    #     label="test",
    #     window_size=1,
    #     tbs_threshold=0,
    #     # delta_begin=datetime.timedelta(seconds=600)
    # )
    # print("snr stat: ", lf.get_snr_statistics())
    # print("channel stat: ", lf.get_channel_statistics())
    # print("mcs: ", lf.get_mcs_statistics())
    # for th in [0, 1, 10, 20, 30, 50, 100, 150, 200, 300]:
    #     print(th, sum([sample.tb_len >= th for sample in lf.samples]), end=" ")

    # Unit test of AmarisoftLogFile
    lf = AmarisoftLogFile(
        read_path="data/NR/SA/20240412/gnb0(2).log",
        timetable=[((datetime.time(0, 0, 0), datetime.time(23, 59, 59)), "test")],
        window_size=1,
        tb_len_threshold=0
    )

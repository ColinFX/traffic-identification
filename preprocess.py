import abc
import csv
import datetime
import json
import re
import time
from typing import Dict, List, Match, Pattern, Tuple

import matplotlib.pyplot as plt


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


class GNBLogFile:
    def __init__(
            self,
            read_path: str,
            save_path: str,
            timetable: List[Tuple[Tuple[datetime.time, datetime.time], str]],
            window_size: int = 1,
            tb_len_threshold: int = 150
    ):
        """Read log from `read_path` and save preprocessed physical layer data records for ML/DL models"""
        with open(read_path, 'r') as f:
            self.lines: List[str] = f.readlines()
        self.date: datetime.date = self._process_header()
        self.raw_records: List[List[str]] = self._group_lines()
        self.records: List[GNBRecord] = [self._reformat_record(raw_record) for raw_record in self.raw_records]
        self._filter_phy_drb_records()
        self._sort_records()
        self._add_labels(timetable)
        self._export_json(save_path)
        self.samples: List[List[GNBRecord]] = self._regroup_records(window_size)
        self._filter_samples(tb_len_threshold)
        self.sample_labels: List[str] = self._reform_sample_labels()

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

    @staticmethod
    def _get_label(record: GNBRecord, timetable: List[Tuple[Tuple[datetime.time, datetime.time], str]]) -> str:
        """Get ground truth label from given `timetable` for one `record`"""
        for range_, label in timetable:
            if range_[0] <= record.time < range_[1]:
                return label
        return ""

    def _add_labels(
            self,
            timetable: List[Tuple[Tuple[datetime.time, datetime.time], str]],
            delete_unlabelled: bool = True
    ):
        """Add ground truth label for all `records` by given `timetable`, delete records without label if requested"""
        for idx, record in enumerate(self.records):
            label = self._get_label(record, timetable)
            record.label = label
            if delete_unlabelled and not record.label:
                del self.records[idx]

    def _export_json(self, save_path: str):
        """Save physical layer records with label to json file"""
        with open(save_path, 'w') as f:
            for record in self.records:
                json.dump(vars(record), f, indent=4, default=str)
                f.write("\n")

    # def _export_csv(self, save_path: str):
    #     """Save physical layer records with label to csv file, CONFIG ONLY"""
    #     with open(save_path, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         keys_basic_info: List[str] = list(set().union(*[obj.basic_info.keys() for obj in self.records]))
    #         keys_short_message: List[str] = list(set().union(*[obj.short_message.keys() for obj in self.records]))
    #         keys_long_message: List[str] = list(set().union(*[obj.long_message.keys() for obj in self.records]))
    #         writer.writerow(["label", "time", "layer"] + keys_basic_info + keys_short_message + keys_long_message)
    #         for record in self.records:
    #             if record.label:
    #                 row = [record.label, record.time, record.layer]
    #                 for key in keys_basic_info:
    #                     row.append(record.basic_info.get(key, np.nan))
    #                 for key in keys_short_message:
    #                     row.append(record.short_message.get(key, np.nan))
    #                 for key in keys_long_message:
    #                     row.append(record.long_message.get(key, np.nan))
    #                 writer.writerow(row)

    def _regroup_records(self, window_size: int) -> List[List[GNBRecord]]:
        """Regroup records by fixed window size (number of frame)"""
        samples: List[List[GNBRecord]] = []
        current_period = -1
        current_frame = - window_size
        current_sample: List[GNBRecord] = []
        for record in self.records:
            if (
                int(record.basic_info["period"]) == current_period and
                int(record.basic_info["frame"]) < current_frame + window_size
            ):
                current_sample.append(record)
            else:
                current_period = int(record.basic_info["period"])
                current_frame = int(record.basic_info["frame"])
                if current_sample:
                    samples.append(current_sample)
                current_sample = [record]
        if current_sample:
            samples.append(current_sample)
        return samples

    @staticmethod
    def count_tb_len(sample: List[GNBRecord]) -> int:
        """Calculate sum of tb_len of records in one sample as amount of data transmitted"""
        tb_len_sum: int = 0
        for record in sample:
            if "tb_len" in record.short_message.keys():
                tb_len_sum += int(record.short_message["tb_len"])
        return tb_len_sum

    def _filter_samples(self, threshold: int):
        """Keep only samples with enough data transmitted"""
        filtered_samples: List[List[GNBRecord]] = []
        for sample in self.samples:
            if GNBLogFile.count_tb_len(sample) >= threshold:
                filtered_samples.append(sample)
        self.samples = filtered_samples

    def _reform_sample_labels(self) -> List[str]:
        """Get label for each newly formed sample by majority voting of records"""
        sample_labels: List[str] = []
        for sample in self.samples:
            voting: Dict[str, int] = {}
            for record in sample:
                if record.label in voting.keys():
                    voting[record.label] += 1
                else:
                    voting[record.label] = 1
            sample_labels.append(max(voting, key=voting.get))
        return sample_labels

    def count_feature_combinations(self):
        """Count different combinations of features for each physical channel for feature selection, CONFIG ONLY"""
        for channel in ["PDSCH", "PDCCH", "PUCCH", "SRS", "PUSCH", "PHICH", "PRACH"]:
            print(">", channel)
            combinations = {}
            for sample in self.samples:
                for record in sample:
                    if record.basic_info["channel"] == channel:
                        tags = list(record.basic_info.keys())
                        tags.extend(list(record.short_message.keys()))
                        tags.extend(list(record.long_message.keys()))
                        tags = str(sorted(tags))
                        if tags not in combinations.keys():
                            combinations[tags] = 1
                        else:
                            combinations[tags] += 1
            all_tags = sorted(list(set().union(*[json.loads(key.replace("'", "\"")) for key in combinations.keys()])))
            new_combinations = {}
            for key in combinations.keys():
                new_key = all_tags.copy()
                for tag_idx, tag in enumerate(new_key):
                    if ("'" + str(tag) + "'") not in key:
                        new_key[tag_idx] = " " * len(new_key[tag_idx])
                new_combinations[str(new_key)] = combinations[key]
            for key in new_combinations:
                print("{:>10}\t".format(int(new_combinations[key])), ' '.join(json.loads(key.replace("'", "\""))))
        print("\n")


if __name__ == "__main__":
    # preprocess
    start = time.time()
    log = GNBLogFile(
        read_path="data/NR/1st-example/gnb0.log",
        save_path="data/NR/1st-example/export.json",
        timetable=[
            ((datetime.time(9, 48, 20), datetime.time(9, 58, 40)), "navigation_web"),
            ((datetime.time(10, 1, 40), datetime.time(10, 13, 20)), "streaming_youtube")
        ],
        window_size=1,
        tb_len_threshold=150
    )
    print("Preprocessing finished in {:.2f} seconds".format(time.time() - start))

    # channel statistics in labelled timezone
    channel_stat = {}
    for record in log.records:
        if record.basic_info["channel"] in channel_stat.keys():
            channel_stat[record.basic_info["channel"]] += 1
        else:
            channel_stat[record.basic_info["channel"]] = 1
    plt.bar(channel_stat.keys(), channel_stat.values())
    plt.title("PHY Records of Different Channels in Dataset (total {})".format(len(log.records)))
    plt.ylabel("# records")
    plt.show()

    # tb_len statistics after regroup and threshold filtering
    tb_lens_web = []
    tb_lens_youtube = []
    for i, label in enumerate(log.sample_labels):
        if label == "navigation_web":
            tb_lens_web.append(GNBLogFile.count_tb_len(log.samples[i]))
        elif label == "streaming_youtube":
            tb_lens_youtube.append(GNBLogFile.count_tb_len(log.samples[i]))
        else:
            pass
    plt.hist([tb_lens_web, tb_lens_youtube], density=False, histtype='bar', stacked=False, label=["web", "youtube"])
    plt.yscale('log')
    plt.title("Samples with Different sum(tb_len) After Threshold (total {})".format(len(log.samples)))
    plt.ylabel("# samples")
    plt.xlabel("sum(tb_len)")
    plt.legend()
    plt.show()

    # reorganize tags
    print("\nTag combinations of different physical layer channels: ")
    log.count_feature_combinations()

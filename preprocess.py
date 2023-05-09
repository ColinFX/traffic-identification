import abc
import csv
import datetime
import re
import time
from typing import Dict, List


class GNBRecord:
    def __init__(self, raw_record: List[str]):
        self.raw_record = raw_record
        match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s\[([A-Z0-9]+)]', raw_record[0])
        self.time: datetime.time = datetime.datetime.strptime(match.groups()[0], "%H:%M:%S.%f").time()
        self.layer: str = match.groups()[1]
        self.basic_info: Dict[str, str] = self._extract_basic_info()
        self.short_message: Dict[str, str] = self._extract_short_message()
        self.long_message: Dict[str, str] = self._extract_long_message()

    @abc.abstractmethod
    def _extract_basic_info(self) -> Dict[str, str]:
        return {}

    @abc.abstractmethod
    def _extract_short_message(self) -> Dict[str, str]:
        return {}

    @abc.abstractmethod
    def _extract_long_message(self) -> Dict[str, str]:
        return {}


class GNBRecordPHY(GNBRecord):
    def __init__(self, raw_record: List[str]):
        super().__init__(raw_record)

    def _extract_basic_info(self) -> Dict[str, str]:
        match = re.match(r'\S+\s+\[\S+]\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+):', self.raw_record[0])
        keys = ["dir", "ue_id", "cell_id", "rnti", "sfn", "channel"]
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
        match = re.match(r'\S+\s+\[\S+]\s+(\S+)\s+(\S+)\s+(\S+)', self.raw_record[0])
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
        match = re.match(r'\S+\s+\[\S+]\s+(\S+)\s(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)', self.raw_record[0])
        keys = ["dir", "ip", "port"]
        return dict(zip(keys, match.groups()))

    def _extract_short_message(self) -> Dict[str, str]:
        short_message: Dict[str, str] = dict(re.findall(r"(\S+)=(\S+)", self.raw_record[0]))
        match = re.match(
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
    def __init__(self, read_path: str, save_path: str):
        self.filename = read_path
        with open(read_path, 'r') as f:
            self.lines: List[str] = f.readlines()
        self.date: datetime.date = self._process_header()
        self.raw_records: List[List[str]] = self._group_lines()
        self.records: List[GNBRecord] = [self._reformat_record(raw_record) for raw_record in self.raw_records]
        self._filter_phy_drb_record()
        self._export_csv(save_path)

    def _process_header(self) -> datetime.date:
        """Remove header marked by `#` and get date"""
        i = 0
        while i < len(self.lines) and (self.lines[i].startswith('#')):
            i += 1
        date_str: str = re.search(r'\d{4}-\d{2}-\d{2}', self.lines[i-1]).group()
        date: datetime.date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        self.lines = self.lines[i:]
        return date

    def _group_lines(self) -> List[List[str]]:
        """Group lines into blocks as raw_records with plain text"""
        raw_records: List[List[str]] = []
        pattern = re.compile(r'\d{2}:\d{2}:\d{2}\.\d{3} \[[A-Z0-9]+]')
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
        """Convert raw_record with plain text into GNBRecord instance"""
        if "[PHY]" in raw_record[0]:
            return GNBRecordPHY(raw_record)
        elif "[RLC]" in raw_record[0]:
            return GNBRecordRLC(raw_record)
        elif "[GTPU]" in raw_record[0]:
            return GNBRecordGTPU(raw_record)
        else:
            return GNBRecord(raw_record)

    def _filter_phy_drb_record(self):
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

    # def _filter_rlc_record(self):
    #     """Keep only records of RLC layer, config ONLY"""
    #     self.records = [record for record in self.records if record.layer == "RLC"]

    def _export_csv(self, save_path: str):
        """Save records to csv file"""
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            keys_basic_info: List[str] = list(set().union(*[obj.basic_info.keys() for obj in self.records]))
            keys_short_message: List[str] = list(set().union(*[obj.short_message.keys() for obj in self.records]))
            keys_long_message: List[str] = list(set().union(*[obj.long_message.keys() for obj in self.records]))
            writer.writerow(["time", "layer"] + keys_basic_info + keys_short_message + keys_long_message)
            for record in self.records:
                row = [record.time, record.layer]
                for key in keys_basic_info:
                    row.append(record.basic_info.get(key, ""))
                for key in keys_short_message:
                    row.append(record.short_message.get(key, ""))
                for key in keys_long_message:
                    row.append(record.long_message.get(key, ""))
                writer.writerow(row)

    # def count_layers(self) -> Dict[str, int]:
    #     """Count number of each kind of layer in the record"""
    #     count: Dict[str, int] = {}
    #     for record in self.records:
    #         if record.layer in count.keys():
    #             count[record.layer] += 1
    #         else:
    #             count[record.layer] = 1
    #     return count


if __name__ == "__main__":
    start = time.time()
    log = GNBLogFile(read_path="data/NR/1st-example/gnb0.log", save_path="data/NR/1st-example/export.csv")
    print("Preprocessing finished in {:.2f} seconds".format(time.time() - start))

import os
import re
from typing import List


class ENBLogFile:
    def __init__(self, filename: str):
        self.filename: str = filename
        self.lines: List[str] = []
        self.layer_records: List[List[str]] = []
        self.records: List[List[List[str]]] = []
        with open(filename, 'r') as f:
            self.lines = f.readlines()
            i = 0
            while i < len(self.lines) and (self.lines[i].startswith('#') or self.lines[i].startswith('+')):
                i += 1
            self.lines = self.lines[i:]
        self.lines = [line.rstrip('\n') for line in self.lines]
        self.lines = [line for line in self.lines if not line == ""]
        self.segment_lines()
        self.segment_layer_records()
        self.filter_downlink_records()

    def segment_lines(self) -> None:
        """Segment `lines` into `layer_records` by layer marks"""
        pattern = re.compile(r'\d{2}:\d{2}:\d{2}\.\d{3} \[[A-Z0-9]+]')
        self.layer_records = []
        current_layer_record: List[str] = []
        for line in self.lines:
            if pattern.match(line):
                if len(current_layer_record) > 0:
                    self.layer_records.append(current_layer_record)
                current_layer_record = [line]
            else:
                current_layer_record.append(line)
        if len(current_layer_record) > 0:
            self.layer_records.append(current_layer_record)

    def segment_layer_records(self) -> None:
        """Segment `layer_records` into `records` according to timestamp"""
        pattern = re.compile(r'(\d{2}:\d{2}:\d{2})\.\d{3}')
        self.records = []
        current_record: List[List[str]] = []
        current_timestamp: str = ''
        for layer_record in self.layer_records:
            match = pattern.match(layer_record[0])
            if match:
                timestamp = match.group(0)
                if timestamp != current_timestamp:
                    if len(current_record) > 0:
                        self.records.append(current_record)
                    current_record = [layer_record]
                    current_timestamp = timestamp
                else:
                    current_record.append(layer_record)

    def filter_downlink_records(self) -> None:
        """Keep only `layer_records` of downlink data stream in `records`"""
        drb_records: List[List[List[str]]] = []
        for record in self.records:
            drb_layer_records: List[List[str]] = []
            data_flag: bool = False
            label_flag: bool = False
            for layer_record in record:
                if "[GTPU] FROM" in layer_record[0] or "DL" in layer_record[0]:
                    drb_layer_records.append(layer_record)
                if "DRB" in layer_record[0] and "D/C=0" in layer_record[0]:
                    data_flag = True
                if "[GTPU] FROM" in layer_record[0]:
                    label_flag = True
            if drb_layer_records and data_flag and label_flag:
                drb_records.append(drb_layer_records)
        self.records = drb_records

    def save_layer_records(self, layer: str, filename: str, ignore_empty: bool) -> bool:
        """Save all `layer_records` of the specific layer"""
        if ignore_empty and len(self.records) == 0:
            return False
        with open(filename, 'w') as f:
            for record in self.records:
                for layer_record in record:
                    if layer in layer_record[0]:
                        f.write('\n'.join(layer_record))
                        f.write('\n')
                f.write('\n')
            return True

    def save_records(self, filename: str, ignore_empty: bool) -> bool:
        """Save all `records`"""
        return self.save_layer_records("", filename, ignore_empty)


def preprocess_batch_ENBLogFile(data_dir: str, ignore_empty: bool):
    os.makedirs(os.path.join(data_dir, "drb"), exist_ok=True)
    pattern = re.compile(r"enb-export-\d{8}-\d{6}\.log")
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and pattern.match(file_name):
            export_path = os.path.join(data_dir, "drb", file_name)
            log = ENBLogFile(file_path)
            saved = log.save_records(export_path, ignore_empty)
            if saved:
                print(f"{os.path.getsize(export_path):10} bytes saved to {export_path}")


if __name__ == "__main__":
    preprocess_batch_ENBLogFile(
        data_dir="./data/lte/enb-export",
        ignore_empty=True
    )

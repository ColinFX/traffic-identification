import json
import os
import re
from typing import List, Dict


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
        self._segment_lines()
        self._segment_layer_records()

    def _segment_lines(self) -> None:
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

    def _segment_layer_records(self) -> None:
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
                    if current_record:
                        self.records.append(current_record)
                    current_record = [layer_record]
                    current_timestamp = timestamp
                else:
                    current_record.append(layer_record)
        if current_record:
            self.records.append(current_record)

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

    @staticmethod
    def extract_phy(layer_record: List[str]):
        """
        Extract information from a PHY `s`

        PHY Log file format:
            time layer dir_ ue_id cell RNTI frame.subframe channel:short_content
            long_content
        """
        s = '\n'.join(layer_record)
        match = re.match(r'(\S+) \[(\S+)] (\S+) (\S+) (\S+) (\S+)\s+(\S+)', s)
        if not match:
            return None
        time, layer, dir_, ue_id, cell, rnti, frame_subframe = match.groups()
        frame, subframe = frame_subframe.split('.')

        match = re.search(r'(\S+): (.+)', s)
        if not match:
            return None
        channel, short_content = match.groups()
        short_content_list = re.findall(r"(\S+=\S+)", short_content)
        short_content_dict: Dict[str, str] = {}
        for item in short_content_list:
            key, value = item.split('=')
            short_content_dict[key] = value

        long_content_list = re.findall(r'\n(\S+=\S+)', s)
        long_content_dict: Dict[str, str] = {}
        for item in long_content_list:
            key, value = item.split('=')
            long_content_dict[key] = value

        return {
            'time': time,
            'layer': layer,
            'dir_': dir_,
            'ue_id': ue_id,
            'cell': cell,
            'rnti': rnti,
            'frame': frame,
            'subframe': subframe,
            'channel': channel,
            'short_content': short_content_dict,
            'long_content': long_content_dict
        }

    @staticmethod
    def extract_rlc(layer_record: List[str]):
        """
        Extract information from an RLC `layer_record`

        RLC Log file format:
            time layer dir ue_id identifier short_content
        """
        s = '\n'.join(layer_record)
        match = re.match(r'(\S+) \[(\S+)] (\S+) (\S+) (\S+)', s)
        if not match:
            return None
        time, layer, dir_, ue_id, identifier = match.groups()

        short_content_list = re.findall(r'(\S+=\S+)', s)
        short_content_dict: Dict[str, str] = {}
        for item in short_content_list:
            key, value = item.split('=')
            short_content_dict[key] = value

        return {
            'time': time,
            'layer': layer,
            'dir': dir_,
            'ue_id': ue_id,
            'identifier': identifier,
            'short_content': short_content_dict
        }

    @staticmethod
    def extract_mac(layer_record: List[str]):
        """
        Extract information from an MAC `layer_record`

        MAC Log file format:
            time layer dir ue_id cell_id short_content
        """
        s = '\n'.join(layer_record)
        match = re.match(r'(\S+) \[(\S+)] (\S+) (\S+) (\S+)', s)
        if not match:
            return None
        time, layer, dir_, ue_id, cell_id = match.groups()

        short_content_list = re.findall(r'(\S+[=:]\S+)', s)
        short_content_dict: Dict[str, str] = {}
        for item in short_content_list:
            key, value = item.split('=|:')
            short_content_dict[key] = value

        return {
            'time': time,
            'layer': layer,
            'dir': dir_,
            'ue_id': ue_id,
            'cell_id': cell_id,
            'short_content': short_content_dict
        }

    @staticmethod
    def extract_gtpu(layer_record: List[str]):
        """
        Extract information from a GTPU `layer_record`

        GTPU Log file format:
            time layer dir ip short_content
            long_content
        """
        s = '\n'.join(layer_record)
        match = re.match(r'(\S+) \[(\S+)] (\S+) (\S+):(\d+) (.+)', s)
        if not match:
            return None
        time, layer, dir_, ip, port, short_content = match.groups()

        short_content_list = re.findall(r"(\S+=\S+)", short_content)
        short_content_dict: Dict[str, str] = {}
        for item in short_content_list:
            key, value = item.split('=')
            short_content_dict[key] = value

        long_content_list = re.findall(r'\n\s+(\S+):(.+)', s)
        long_content_dict: Dict[str, str] = {}
        for item in long_content_list:
            key, value = item
            long_content_dict[key] = value

        expeditor_ip, expeditor_port, receiver_ip, receiver_port = None, None, None, None
        match = re.search(
            r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+) > (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)',
            short_content
        )
        if match:
            expeditor_ip, expeditor_port, receiver_ip, receiver_port = match.groups()

        return {
            'time': time,
            'layer': layer,
            'dir_': dir_,
            'ip': ip,
            'port': port,
            'expeditor_ip': expeditor_ip,
            'expeditor_port': expeditor_port,
            'receiver_ip': receiver_ip,
            'receiver_port': receiver_port,
            'short_content': short_content_dict,
            'long_content': long_content_dict
        }


def preprocess_batch_enb_logfile(data_dir: str, ignore_empty: bool):
    os.makedirs(os.path.join(data_dir, "drb"), exist_ok=True)
    pattern = re.compile(r"enb-export-\d{8}-\d{6}\.log")
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and pattern.match(file_name):
            export_path = os.path.join(data_dir, "drb", file_name)
            log = ENBLogFile(file_path)
            log.filter_downlink_records()
            saved = log.save_records(export_path, ignore_empty)
            if saved:
                print(f"{os.path.getsize(export_path):10} bytes saved to {export_path}")


if __name__ == "__main__":
    # preprocess_batch_enb_logfile(
    #     data_dir="./data/lte/enb-export",
    #     ignore_empty=True
    # )

    log = ENBLogFile("./data/lte/enb-export-test/enb-export-00000000-000000.log")
    for record in log.records:
        for layer_record in record:
            if "GTPU" in layer_record[0]:
                print(">>>>")
                print(layer_record)
                print(json.dumps(ENBLogFile.extract_gtpu(layer_record), indent=4))

    # log = ENBLogFile("./data/lte/enb-export-test/enb-export-00000000-000001.log")
    # log.save_layer_records(layer="GTPU", filename="./data/lte/enb-export-test/enb-export-00000000-000001-gtpu.log",
    #                        ignore_empty=False)

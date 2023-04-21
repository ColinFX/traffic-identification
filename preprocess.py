import re
from typing import List


class LogFile:
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
        self.filter_records()

    def segment_lines(self) -> None:
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

    def filter_records(self) -> None:
        drb_records: List[List[List[str]]] = []
        for record in self.records:
            keep_record = False
            for layer_record in record:
                if "DRB" in layer_record[0]:
                    keep_record = True
                    break
            if keep_record:
                drb_records.append(record)
        self.records = drb_records

    def save_layer_records(self, layer: str, filename: str) -> None:
        with open(filename, 'w') as f:
            for record in self.records[:100]:  # CONFIG HERE
                for layer_record in record:
                    if layer in layer_record[0]:
                        f.write('\n'.join(layer_record))
                        f.write('\n')
                f.write('\n')

    def save_records(self, filename: str) -> None:
        self.save_layer_records("", filename)


if __name__ == "__main__":
    log = LogFile("./data/lte/enb-export-20230413-135802.log")
    print(len(log.lines))
    print(len(log.layer_records))
    print(len(log.records))
    log.save_records("./data/lte/enb-export-20230413-135802-drb.log")
    log.save_layer_records("PDCP", "./data/lte/enb-export-20230413-135802-pdcp.log")
    log.save_layer_records("RLC", "./data/lte/enb-export-20230413-135802-rlc.log")
    log.save_layer_records("MAC", "./data/lte/enb-export-20230413-135802-mac.log")
    log.save_layer_records("PHY", "./data/lte/enb-export-20230413-135802-phy.log")
    log.save_layer_records("GTPU", "./data/lte/enb-export-20230413-135802-gtpu.log")

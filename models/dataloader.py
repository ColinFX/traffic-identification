from typing import Dict

from torch.utils.data import DataLoader, Dataset


class TCDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx: int) -> Dict[str, object]:
        pass


class TCDataLoader(DataLoader):
    def __init__(self, mode: str):
        super().__init__()

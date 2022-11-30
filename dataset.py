import torch
from torch.utils.data import Dataset
import numpy as np


class JetDataset(Dataset):
    def __init__(self, transform=None, files: list = [], channel: str = "ecal") -> None:
        super(JetDataset, self).__init__()
        self.files = files
        self.channel = channel
        self.transform = transform
        self.len = len(self.files)

    @property
    def raw_file_names(self):
        return self.files

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx > self.len:
            raise IndexError(
                f"index {idx} exceeds total number of instances in the dataset"
            )

        with np.load(self.files[idx]) as f:
            event = torch.from_numpy(f[self.channel])
            if self.transform:
                event = self.transform(event)

            return event

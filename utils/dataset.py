import torch
from torch.utils.data import Dataset
import numpy as np


class JetDataset(Dataset):
    def __init__(self, files: list = [], channel: str = "ecal", normalize=True, energy_scale=None) -> None:
        super(JetDataset, self).__init__()
        self.files = files
        self.channel = channel
        self.len = len(self.files)
        self.normalize = normalize
        self.scale = energy_scale

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

            if self.normalize is True:
                event[:, 0] = (event[:, 0] - 0.) / 125
                event[:, 1] = (event[:, 1] - 0.) / 125
            if self.scale is not None:
                event[:, 2] = (event[:, 2] - self.scale[0]) / (self.scale[1] - self.scale[0])
            return event

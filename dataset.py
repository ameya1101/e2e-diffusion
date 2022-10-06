import torch
from torch.utils.data import Dataset
import numpy as np
from pyarrow.parquet import ParquetFile


class JetDataset(Dataset):
    def __init__(self, PATH, transforms=None, columns=["X_jets"], channels=[0, 1, 2]) -> None:
        super(JetDataset, self).__init__()

        self.parquets = []
        self.channels = channels
        self.transforms = transforms
        self.columns = columns
        cumrows = 0
        for file in PATH:
            parquet = ParquetFile(file)
            rows = parquet.num_row_groups
            cumrows += rows
            self.parquets.append((parquet, rows, cumrows))

    def __len__(self):
        return sum(file[1] for file in self.parquets)

    def __getitem__(self, index):
        for parquet, rows, cumrows in self.parquets:
            if index > cumrows:
                raise IndexError("Item index out of range")
            if index < cumrows:
                break
            else:
                continue

        index = index - (cumrows - rows)
        row = parquet.read_row_group(index, columns=self.columns).to_pydict()
        data = np.float32(row["X_jets"][0])

        data[data < 1e-3] = 0.0
        data = torch.from_numpy(data)

        if self.transforms:
            data = self.transforms(data)
        data = data[self.channels]
        return data

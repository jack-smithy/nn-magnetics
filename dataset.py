from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

_DTYPES = {
    "mps": torch.float32,
    "cuda": torch.float64,
    "cpu": torch.float64,
}


def _make_train_test_split(path: Path) -> List[np.ndarray]:
    # initialise empty arrays for data
    input_data = np.empty((0, 6))
    output_data = np.empty((0, 3))

    # iterate over all the files in the directory
    for file in path.iterdir():
        data = np.load(file)
        grid = data["grid"]
        length = len(grid)

        # select relevant parts of the input data
        # in this case, magnet dims, susceptibility, and point in space
        input_data_new = np.vstack(
            (
                np.ones(length) * data["a"],
                np.ones(length) * data["b"],
                np.ones(length) * data["chi"],
                grid[:, 0] / data["a"],
                grid[:, 1] / data["b"],
                grid[:, 2],
            )
        ).T

        # get the corresponding labels
        output_data_new = data["grid_field_reduced"]

        # concat these to the data arrays
        input_data = np.concatenate((input_data, input_data_new))
        output_data = np.concatenate((output_data, output_data_new))

    # sanity check to make sure they are the same length
    assert input_data.shape[0] == output_data.shape[0]

    return train_test_split(
        input_data,
        output_data,
        test_size=0.3,
        shuffle=True,
    )


def make_train_test_split(path: Path | str) -> List[np.ndarray]:
    if isinstance(path, str):
        path = Path(path)

    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")

    return _make_train_test_split(path=path)


class DemagData(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device: str):
        self.X = torch.tensor(X).to(device, dtype=_DTYPES[device])
        self.y = torch.tensor(y).to(device, dtype=_DTYPES[device])

        assert self.X.shape[0] == self.y.shape[0]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

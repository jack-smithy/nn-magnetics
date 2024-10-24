from pathlib import Path
from typing import List, Tuple
from enum import Enum

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

_DTYPES = {
    "mps": torch.float32,
    "cuda": torch.float64,
    "cpu": torch.float64,
}


class ChiMode(Enum):
    ISOTROPIC = "isotropic"
    ANISOTROPIC = "anisotropic"


def get_data(path: Path, chi_mode: ChiMode) -> Tuple[np.ndarray, ...]:
    # initialise empty arrays for data
    input_dim = 6 if chi_mode == ChiMode.ISOTROPIC else 7
    output_dim = 6

    input_data = np.empty((0, input_dim))
    output_data = np.empty((0, output_dim))
    n_magnets = len([f for f in path.iterdir()])

    # iterate over all the files in the directory
    for file in path.iterdir():
        data = np.load(file)
        grid = data["grid"]
        length = len(grid)

        # select relevant parts of the input data
        # in this case, magnet dims, susceptibility, and point in space
        if chi_mode == ChiMode.ANISOTROPIC:
            input_data_new = np.vstack(
                (
                    np.ones(length) * data["a"],
                    np.ones(length) * data["b"],
                    np.ones(length) * data["chi_perp"],
                    np.ones(length) * data["chi_long"],
                    grid[:, 0] / data["a"],
                    grid[:, 1] / data["b"],
                    grid[:, 2],
                )
            ).T
        else:
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
        output_data_new = np.concatenate(
            (data["grid_field"], data["grid_field_ana"]),
            axis=1,
        )

        # concat these to the data arrays
        input_data = np.concatenate((input_data, input_data_new))
        output_data = np.concatenate((output_data, output_data_new))

    # sanity check to make sure they are the same length
    assert input_data.shape[0] == output_data.shape[0]

    return (
        input_data.reshape(n_magnets, -1, input_dim),
        output_data.reshape(n_magnets, -1, output_dim),
    )


def _make_train_test_split(
    path: Path,
    test_size: float,
    val_size: float,
    chi_mode: ChiMode,
) -> List[np.ndarray]:
    # initialise empty arrays for data
    input_dim = 6 if chi_mode == ChiMode.ISOTROPIC else 7
    output_dim = 6

    input_data = np.empty((0, input_dim))
    output_data = np.empty((0, output_dim))

    # iterate over all the files in the directory
    for file in path.iterdir():
        data = np.load(file)
        grid = data["grid"]
        length = len(grid)

        # select relevant parts of the input data
        # in this case, magnet dims, susceptibility, and point in space
        if chi_mode == ChiMode.ANISOTROPIC:
            input_data_new = np.vstack(
                (
                    np.ones(length) * data["a"],
                    np.ones(length) * data["b"],
                    np.ones(length) * data["chi_perp"],
                    np.ones(length) * data["chi_long"],
                    grid[:, 0] / data["a"],
                    grid[:, 1] / data["b"],
                    grid[:, 2],
                )
            ).T
        else:
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
        output_data_new = np.concatenate(
            (data["grid_field"], data["grid_field_ana"]),
            axis=1,
        )

        # concat these to the data arrays
        input_data = np.concatenate((input_data, input_data_new))
        output_data = np.concatenate((output_data, output_data_new))

    # sanity check to make sure they are the same length
    assert input_data.shape[0] == output_data.shape[0]

    # Step 1: Split the data into training and test sets
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        input_data,
        output_data,
        test_size=(val_size + test_size),
        shuffle=False,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test,
        y_val_test,
        test_size=(1 - (val_size / (val_size + test_size))),
        shuffle=False,
    )

    return [
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
    ]


def make_train_test_split(
    path: Path | str,
    chi_mode: ChiMode,
    train_size: float = 0.5,
    test_size: float = 0.3,
    val_size: float = 0.2,
) -> List[np.ndarray]:
    if isinstance(path, str):
        path = Path(path)

    if not np.isclose(train_size + test_size + val_size, 1):
        raise ValueError(
            f"Splits do not add up to 1: Sum={train_size + test_size + val_size}"
        )

    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")

    return _make_train_test_split(
        path=path,
        test_size=test_size,
        val_size=val_size,
        chi_mode=chi_mode,
    )


class DemagData(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device: str = "cpu"):
        self.X = torch.tensor(X).to(device, dtype=_DTYPES[device])
        self.y = torch.tensor(y).to(device, dtype=_DTYPES[device])

        assert self.X.shape[0] == self.y.shape[0]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

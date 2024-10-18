from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
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
                grid[:, 0],
                grid[:, 1],
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

    # Step 1: Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        input_data,
        output_data,
        test_size=0.3,
        shuffle=True,
    )

    # Step 2: Scale the data using StandardScaler (you can also use MinMaxScaler)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit the scaler on the training data and transform both the training and test data
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Fit and scale the output data (y) separately if needed
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Now you can return the scaled data
    return [X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled]


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

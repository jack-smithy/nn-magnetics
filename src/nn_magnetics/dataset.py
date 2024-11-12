from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_DTYPES = {
    "mps": torch.float32,
    "cuda": torch.float64,
    "cpu": torch.float64,
}


class ChiMode(Enum):
    ISOTROPIC = "isotropic"
    ANISOTROPIC = "anisotropic"


class DemagData(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device: str = "cpu"):
        self.X = torch.tensor(X).to(device, dtype=_DTYPES[device])
        self.y = torch.tensor(y).to(device, dtype=_DTYPES[device])

        assert self.X.shape[0] == self.y.shape[0]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_data_parallel(path: str | Path, chi_mode: ChiMode) -> Tuple[np.ndarray, ...]:
    if isinstance(path, str):
        path = Path(path)

    # initialize empty lists for data
    input_dim = 6
    output_dim = 6

    input_data_list = []
    output_data_list = []
    n_magnets = len([f for f in path.iterdir()])

    # define a function to load and process one file
    def process_file(file):
        data = np.load(file)
        return get_one_magnet(chi_mode, data)

    # use ThreadPoolExecutor to parallelize file processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file) for file in path.iterdir()]

        for future in as_completed(futures):
            input_data_new, output_data_new = future.result()
            input_data_list.append(input_data_new)
            output_data_list.append(output_data_new)

    # concatenate all input and output data arrays
    input_data = np.concatenate(input_data_list)
    output_data = np.concatenate(output_data_list)

    # sanity check to make sure they are the same length
    assert input_data.shape[0] == output_data.shape[0]

    return (
        input_data.reshape(n_magnets, -1, input_dim),
        output_data.reshape(n_magnets, -1, output_dim),
    )


def get_data(path: str | Path, chi_mode: ChiMode) -> Tuple[np.ndarray, ...]:
    if isinstance(path, str):
        path = Path(path)

    # initialise empty arrays for data
    input_dim = 6
    output_dim = 6

    input_data = np.empty((0, input_dim))
    output_data = np.empty((0, output_dim))
    n_magnets = len([f for f in path.iterdir()])

    # iterate over all the files in the directory
    for file in path.iterdir():
        data = np.load(file)
        input_data_new, output_data_new = get_one_magnet(chi_mode, data)

        # concat these to the data arrays
        input_data = np.concatenate((input_data, input_data_new))
        output_data = np.concatenate((output_data, output_data_new))

    # sanity check to make sure they are the same length
    assert input_data.shape[0] == output_data.shape[0]

    return (
        input_data.reshape(n_magnets, -1, input_dim),
        output_data.reshape(n_magnets, -1, output_dim),
    )


def get_one_magnet(chi_mode, data):
    grid = data["grid"]
    length = len(grid)

    # select relevant parts of the input data
    # in this case, magnet dims, susceptibility, and point in space
    match chi_mode.value:
        # case ChiMode.ANISOTROPIC.value:
        #     input_data_new = np.vstack(
        #         (
        #             np.ones(length) * data["a"],
        #             np.ones(length) * data["b"],
        #             np.ones(length) * data["chi_perp"],
        #             np.ones(length) * data["chi_long"],
        #             grid[:, 0] / data["a"],
        #             grid[:, 1] / data["b"],
        #             grid[:, 2],
        #         )
        #     ).T
        case ChiMode.ISOTROPIC.value:
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
        case _:
            raise ValueError(f"Something gone wrong: {chi_mode.value}")

        # get the corresponding labels
    output_data_new = np.concatenate(
        (data["grid_field"], data["grid_field_reduced"]),
        axis=1,
    )

    return input_data_new, output_data_new

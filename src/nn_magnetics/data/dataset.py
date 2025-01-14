import typing as t
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

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

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class IsotropicData(Dataset):
    def __init__(self, path: Path | str, device: str = "cpu") -> None:
        super().__init__()

        if isinstance(path, str):
            path = Path(path)

        self.path = path
        self.device = device

        self.X, self.B = self._get_all_data()

    def __str__(self) -> str:
        return f"IsotropicData(\n\t{len(self)} data points\n\t{self.X.shape[1]} input features\n\t{self.B.shape[1]} output features\n)"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        assert len(self.X) == len(self.B)
        return len(self.X)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[index], dtype=torch.float32, device=self.device),
            torch.tensor(self.B[index], dtype=torch.float32, device=self.device),
        )

    def get_magnets(self) -> t.Tuple[torch.Tensor, torch.Tensor]:
        X, B = torch.tensor([]), torch.tensor([])
        with ThreadPoolExecutor() as e:
            futures = [e.submit(self._get_magnet, file) for file in self.path.iterdir()]
            for future in as_completed(futures):
                _X, _B = future.result()
                X = torch.cat((X, torch.tensor(_X).unsqueeze(0)))
                B = torch.cat((B, torch.tensor(_B).unsqueeze(0)))
        return (
            X.type(torch.float32).to(self.device),
            B.type(torch.float32).to(self.device),
        )

    def _get_all_data(self):
        input_data_all, output_data_all = [], []

        with ThreadPoolExecutor() as e:
            futures = [e.submit(self._get_magnet, file) for file in self.path.iterdir()]
            for future in as_completed(futures):
                input_data_new, output_data_new = future.result()
                input_data_all.append(input_data_new)
                output_data_all.append(output_data_new)

        input_data = np.concatenate(input_data_all)
        output_data = np.concatenate(output_data_all)

        assert input_data.shape[1] == output_data.shape[1]

        return input_data, output_data

    @staticmethod
    def _get_magnet(file):
        data = np.load(file)
        grid = data["grid"]
        length = len(grid)

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
            (data["grid_field"], data["grid_field_reduced"]),
            axis=1,
        )

        return input_data_new, output_data_new


class AnisotropicData(Dataset):
    def __init__(self, path: Path | str, device: str = "cpu") -> None:
        super().__init__()

        if isinstance(path, str):
            path = Path(path)

        self.path = path
        self.device = device

        self.X, self.B = self._get_all_data()

    def __str__(self) -> str:
        return f"IsotropicData(\n\t{len(self)} data points\n\t{self.X.shape[1]} input features\n\t{self.B.shape[1]} output features\n)"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        assert len(self.X) == len(self.B)
        return len(self.X)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[index], dtype=torch.float32, device=self.device),
            torch.tensor(self.B[index], dtype=torch.float32, device=self.device),
        )

    def get_magnets(self) -> t.Tuple[torch.Tensor, torch.Tensor]:
        X, B = torch.tensor([]), torch.tensor([])
        with ThreadPoolExecutor() as e:
            futures = [e.submit(self._get_magnet, file) for file in self.path.iterdir()]
            for future in as_completed(futures):
                _X, _B = future.result()
                X = torch.cat((X, torch.tensor(_X).unsqueeze(0)))
                B = torch.cat((B, torch.tensor(_B).unsqueeze(0)))
        return (
            X.type(torch.float32).to(self.device),
            B.type(torch.float32).to(self.device),
        )

    def _get_all_data(self):
        input_data_all, output_data_all = [], []

        with ThreadPoolExecutor() as e:
            futures = [e.submit(self._get_magnet, file) for file in self.path.iterdir()]
            for future in as_completed(futures):
                input_data_new, output_data_new = future.result()
                input_data_all.append(input_data_new)
                output_data_all.append(output_data_new)

        input_data = np.concatenate(input_data_all)
        output_data = np.concatenate(output_data_all)

        assert input_data.shape[0] == output_data.shape[0]

        return input_data, output_data

    @staticmethod
    def _get_magnet(file):
        data = np.load(file)
        grid = data["grid"]
        length = len(grid)

        input_data_new = np.vstack(
            (
                np.ones(length) * data["a"],
                np.ones(length) * data["b"],
                np.ones(length) * data["chi_x"],
                np.ones(length) * data["chi_y"],
                np.ones(length) * data["chi_z"],
                grid[:, 0] / data["a"],
                grid[:, 1] / data["b"],
                grid[:, 2],
            )
        ).T

        # get the corresponding labels
        output_data_new = np.concatenate(
            (data["grid_field"], data["grid_field_reduced"]),
            axis=1,
        )

        return input_data_new, output_data_new


##############################
####   Older stuff here   ####
##############################


def get_data_parallel(path: str | Path, chi_mode: ChiMode) -> t.Tuple[np.ndarray, ...]:
    if isinstance(path, str):
        path = Path(path)

    # initialize empty lists for data
    input_dim = 6 if chi_mode.value == ChiMode.ISOTROPIC.value else 7
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


def get_data(path: str | Path, chi_mode: ChiMode) -> t.Tuple[np.ndarray, ...]:
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
        case ChiMode.ANISOTROPIC.value:
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

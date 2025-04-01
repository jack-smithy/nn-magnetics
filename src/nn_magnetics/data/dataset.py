import typing as t
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class AnisotropicData(Dataset):
    def __init__(
        self,
        path: Path | str,
        device: str = "cpu",
        dtype=torch.float64,
    ) -> None:
        super().__init__()

        if isinstance(path, str):
            path = Path(path)

        self.path = path
        self.device = device
        self.dtype = dtype

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
            torch.tensor(
                self.X[index], device=self.device, dtype=self.dtype, requires_grad=True
            ),
            torch.tensor(
                self.B[index], device=self.device, dtype=self.dtype, requires_grad=True
            ),
        )

    def get_magnets(self) -> t.Tuple[torch.Tensor, torch.Tensor]:
        X, B = torch.tensor([]), torch.tensor([])
        with ThreadPoolExecutor() as e:
            futures = [e.submit(self._get_magnet, file) for file in self.path.iterdir()]
            for future in as_completed(futures):
                _X, _B = future.result()
                X = torch.cat((X, torch.tensor(_X, dtype=self.dtype).unsqueeze(0)))
                B = torch.cat((B, torch.tensor(_B, dtype=self.dtype).unsqueeze(0)))
        return (
            X.to(self.device),
            B.to(self.device),
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
                # grid[:, 0] / data["a"],
                # grid[:, 1] / data["b"],
                grid[:, 0],
                grid[:, 1],
                grid[:, 2],
            )
        ).T

        # get the corresponding labels
        output_data_new = np.concatenate(
            (data["grid_field"], data["grid_field_reduced"]),
            axis=1,
        )

        return input_data_new, output_data_new

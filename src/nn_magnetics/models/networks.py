from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

from nn_magnetics.models import BaseNetwork
from nn_magnetics.models.corrections import batch_rotation_matrices
from nn_magnetics.utils.physics import (
    invert_quaternion,
    multiply_quaternions,
    cartesian_to_spherical,
    spherical_to_cartesian,
)
from nn_magnetics.utils.physics import Bfield_homogeneous

type Activation = Callable[[torch.Tensor], torch.Tensor]


class AnalyticalModel(BaseNetwork):
    def __init__(
        self,
        do_output_activation: bool,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        p: float = 0,
        save_weights: bool = True,
    ) -> None:
        super().__init__(
            in_features=8,
            out_features=3,
            do_output_activation=do_output_activation,
            save_path=save_path,
            lr_scheduler=lr_scheduler,
            activation=activation,
            p=p,
            save_weights=save_weights,
        )

    def correct_ansatz(self, B_reduced: Tensor, prediction: Tensor) -> Tensor:
        return B_reduced

    def forward(self, x: Tensor) -> Tensor:
        observers, dimensions, polarizations, susceptibilities = self._prepare_inputs(x)

        B_reduced = Bfield_homogeneous(
            observers=observers,
            dimensions=dimensions,
            polarizations=polarizations,
        )

        return B_reduced

    @classmethod
    def load_from_path(
        cls,
        path,
        *,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> AnalyticalModel:
        model = AnalyticalModel(
            activation=activation,
            save_weights=save_weights,
            save_path=save_path,
            do_output_activation=do_output_activation,
            lr_scheduler=lr_scheduler,
            p=p,
        )
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class NoCorrectionNetwork(BaseNetwork):
    def __init__(
        self,
        do_output_activation: bool,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        p: float = 0,
        save_weights: bool = True,
    ) -> None:
        super().__init__(
            in_features=8,
            out_features=3,
            do_output_activation=do_output_activation,
            save_path=save_path,
            lr_scheduler=lr_scheduler,
            activation=activation,
            p=p,
            save_weights=save_weights,
        )

    def correct_ansatz(self, B_reduced: Tensor, prediction: Tensor) -> Tensor:
        return prediction

    @classmethod
    def load_from_path(
        cls,
        path,
        *,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> NoCorrectionNetwork:
        model = NoCorrectionNetwork(
            activation=activation,
            save_weights=save_weights,
            save_path=save_path,
            do_output_activation=do_output_activation,
            lr_scheduler=lr_scheduler,
            p=p,
        )
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class SphericalCorrectionNetwork(BaseNetwork):
    def __init__(
        self,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> None:
        super().__init__(
            in_features=8,
            out_features=3,
            save_path=save_path,
            save_weights=save_weights,
            lr_scheduler=lr_scheduler,
            activation=activation,
            do_output_activation=do_output_activation,
            p=p,
        )

    def correct_ansatz(self, B_reduced, prediction):
        r, theta, phi = cartesian_to_spherical(B_reduced).T

        r_corrected = r * prediction[:, 0]
        theta_corrected = theta + prediction[:, 1]
        phi_corrected = phi + prediction[:, 2]

        B_corrected_polar = torch.stack(
            [
                r_corrected,
                theta_corrected,
                phi_corrected,
            ],
            dim=-1,
        )

        return spherical_to_cartesian(B_corrected_polar)

    @classmethod
    def load_from_path(
        cls,
        path,
        *,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> SphericalCorrectionNetwork:
        model = SphericalCorrectionNetwork(
            activation=activation,
            save_weights=save_weights,
            save_path=save_path,
            do_output_activation=do_output_activation,
            lr_scheduler=lr_scheduler,
            p=p,
        )
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class AdditionCorrectionNetwork(BaseNetwork):
    def __init__(
        self,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> None:
        super().__init__(
            in_features=8,
            out_features=3,
            save_path=save_path,
            save_weights=save_weights,
            lr_scheduler=lr_scheduler,
            activation=activation,
            do_output_activation=do_output_activation,
            p=p,
        )

    def correct_ansatz(self, B_reduced, prediction):
        return B_reduced + prediction

    @classmethod
    def load_from_path(
        cls,
        path,
        *,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> AdditionCorrectionNetwork:
        model = AdditionCorrectionNetwork(
            activation=activation,
            save_weights=save_weights,
            save_path=save_path,
            do_output_activation=do_output_activation,
            lr_scheduler=lr_scheduler,
            p=p,
        )
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class FieldCorrectionNetwork(BaseNetwork):
    def __init__(
        self,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> None:
        super().__init__(
            in_features=8,
            out_features=3,
            save_path=save_path,
            save_weights=save_weights,
            lr_scheduler=lr_scheduler,
            activation=activation,
            do_output_activation=do_output_activation,
            p=p,
        )

    def correct_ansatz(self, B_reduced, prediction):
        return B_reduced * prediction

    @classmethod
    def load_from_path(
        cls,
        path,
        *,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> FieldCorrectionNetwork:
        model = FieldCorrectionNetwork(
            activation=activation,
            save_weights=save_weights,
            save_path=save_path,
            do_output_activation=do_output_activation,
            lr_scheduler=lr_scheduler,
            p=p,
        )
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class AngleAmpCorrectionNetwork(BaseNetwork):
    def __init__(
        self,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> None:
        super().__init__(
            in_features=8,
            out_features=4,
            save_path=save_path,
            lr_scheduler=lr_scheduler,
            activation=activation,
            save_weights=save_weights,
            do_output_activation=do_output_activation,
            p=p,
        )

    def correct_ansatz(self, B_reduced: Tensor, prediction: Tensor) -> Tensor:
        assert prediction.shape[1] == 4, f"Prediction shape is {prediction.shape}"

        angles = prediction[..., :3].clamp(-torch.pi, torch.pi)
        amplitudes = prediction[..., 3].clamp(0.8, 1.2)

        Rs_t = batch_rotation_matrices(angles)

        # Multiply each vector in B_reduced by the corresponding rotation matrix in Rs
        corrected = amplitudes[:, None] * torch.einsum("nij,nj->ni", Rs_t, B_reduced)

        return corrected

    @classmethod
    def load_from_path(
        cls,
        path,
        *,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> AngleAmpCorrectionNetwork:
        model = AngleAmpCorrectionNetwork(
            activation=activation,
            save_weights=save_weights,
            save_path=save_path,
            lr_scheduler=lr_scheduler,
            do_output_activation=do_output_activation,
            p=p,
        )
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class QuaternionNet(BaseNetwork):
    def __init__(
        self,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> None:
        super().__init__(
            in_features=8,
            out_features=5,
            save_path=save_path,
            lr_scheduler=lr_scheduler,
            activation=activation,
            do_output_activation=do_output_activation,
            save_weights=save_weights,
            p=p,
        )

    def correct_ansatz(self, B_reduced: Tensor, prediction: Tensor) -> Tensor:
        # separate amplitudes and rotations
        amplitudes, rotations = prediction[..., 0], prediction[..., 1:]

        # convert vector to quaternion p = (0, Bx, By, Bz)
        p = torch.cat([torch.zeros(B_reduced.shape[0], 1), B_reduced], dim=1)

        # normalize rotation quaternions
        q = rotations / rotations.norm(dim=1, keepdim=True)

        # rotate B-field
        qinv = invert_quaternion(q)
        pprime = multiply_quaternions(multiply_quaternions(qinv, p), q)

        # correct amplitude
        return amplitudes[:, None] * pprime[..., 1:]

    @classmethod
    def load_from_path(
        cls,
        path,
        *,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        do_output_activation: bool = False,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> QuaternionNet:
        model = QuaternionNet(
            activation=activation,
            save_weights=save_weights,
            save_path=save_path,
            lr_scheduler=lr_scheduler,
            do_output_activation=do_output_activation,
            p=p,
        )
        model.load_state_dict(torch.load(path, weights_only=True))
        return model

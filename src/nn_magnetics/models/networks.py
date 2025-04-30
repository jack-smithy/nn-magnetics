from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from nn_magnetics.models.corrections import batch_rotation_matrices
from nn_magnetics.models import BaseNetwork
from nn_magnetics.utils.physics import invert_quaternion, multiply_quaternions

type Activation = Callable[[torch.Tensor], torch.Tensor]


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class FieldCorrectionNetwork(BaseNetwork):
    def __init__(
        self,
        save_path: Path | None = None,
        save_weights: bool = True,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
    ) -> None:
        super().__init__(
            in_features=8,
            out_features=4,
            save_path=save_path,
            save_weights=save_weights,
            lr_scheduler=lr_scheduler,
            activation=activation,
        )

    @staticmethod
    def correct_ansatz(B_reduced, prediction):
        return B_reduced * prediction

    @classmethod
    def load_from_path(
        cls,
        path,
        *,
        activation,
        save_weights,
        save_path,
    ) -> FieldCorrectionNetwork:
        model = FieldCorrectionNetwork(
            save_path=save_path,
            save_weights=save_weights,
            activation=activation,
        )
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class AngleAmpCorrectionNetwork(BaseNetwork):
    def __init__(
        self,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        save_weights: bool = True,
    ) -> None:
        super().__init__(
            in_features=8,
            out_features=4,
            save_path=save_path,
            lr_scheduler=lr_scheduler,
            activation=activation,
            save_weights=save_weights,
        )

    @staticmethod
    def correct_ansatz(B_reduced: Tensor, prediction: Tensor) -> Tensor:
        assert prediction.shape[1] == 4, f"Prediction shape is {prediction.shape}"

        angles = prediction[..., :3]
        amplitudes = prediction[..., 3]

        Rs_t = batch_rotation_matrices(angles)

        # Multiply each vector in B_reduced by the corresponding rotation matrix in Rs
        return amplitudes[:, None] * torch.einsum("nij,nj->ni", Rs_t, B_reduced)

    @classmethod
    def load_from_path(
        cls,
        path,
        *,
        activation,
        save_weights,
        save_path,
    ) -> AngleAmpCorrectionNetwork:
        model = AngleAmpCorrectionNetwork(
            activation=activation,
            save_weights=save_weights,
            save_path=save_path,
        )
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class QuaternionNet(BaseNetwork):
    def __init__(
        self,
        in_features: int,
        hidden_dim_factor: int,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Callable[[Tensor], Tensor] = F.silu,
        save_weights: bool = True,
    ) -> None:
        super().__init__(
            in_features=8,
            out_features=5,
            save_path=save_path,
            lr_scheduler=lr_scheduler,
            activation=activation,
            save_weights=save_weights,
        )

        self.angle_head = nn.Linear(
            in_features=2 * hidden_dim_factor,
            out_features=2 * hidden_dim_factor,
        )

        self.amp_head = nn.Linear(
            in_features=2 * hidden_dim_factor,
            out_features=hidden_dim_factor,
        )

        self.output2 = nn.Linear(
            in_features=hidden_dim_factor,
            out_features=1,
        )

        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))

        x_angle = self.activation(self.angle_head(x))
        x_angle = self.activation(self.linear5(x_angle))
        x_angle = self.output(x_angle)

        x_amp = self.activation(self.amp_head(x))
        x_amp = self.activation(self.output2(x_amp))

        out = torch.cat((x_amp, x_angle), dim=1)

        if self.do_output_activation:
            return self.activation(out)

        return out

    @staticmethod
    def correct_ansatz(B_reduced: Tensor, prediction: Tensor) -> Tensor:
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

    # @classmethod
    # def load_from_path(
    #     cls,
    #     path,
    #     *,
    #     hidden_dim_factor,
    #     do_output_activation,
    #     activation,
    #     save_weights=False,
    # ) -> QuaternionNet:
    #     model = QuaternionNet(
    #         in_features=8,
    #         hidden_dim_factor=hidden_dim_factor,
    #         activation=activation,
    #         do_output_activation=do_output_activation,
    #         save_weights=save_weights,
    #     )
    #     model.load_state_dict(torch.load(path, weights_only=True))
    #     return model

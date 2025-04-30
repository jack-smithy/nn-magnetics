from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler

import wandb
from nn_magnetics.utils.metrics import angle_error, relative_amplitude_error
from nn_magnetics.utils.physics import Dz_cuboid, Bfield_homogeneous

type Activation = Callable[[torch.Tensor], torch.Tensor]


class BaseNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Activation = F.silu,
        save_weights: bool = True,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(
            in_features=in_features,
            out_features=128,
        )
        self.linear2 = nn.Linear(
            in_features=128,
            out_features=48,
        )
        self.linear3 = nn.Linear(
            in_features=48,
            out_features=48,
        )
        self.linear4 = nn.Linear(
            in_features=48,
            out_features=48,
        )
        self.linear5 = nn.Linear(
            in_features=48,
            out_features=48,
        )
        self.linear6 = nn.Linear(
            in_features=48,
            out_features=48,
        )
        self.linear7 = nn.Linear(
            in_features=48,
            out_features=128,
        )
        self.output = nn.Linear(
            in_features=128,
            out_features=out_features,
        )
        self.activation = activation
        self.best_weights = deepcopy(self).state_dict()
        self.save_path = save_path
        self.lr_scheduler = lr_scheduler
        self.save_weights = save_weights

    def _forward(self, x: Tensor) -> Tensor:
        feature = self.activation(self.linear1(x))
        feature = self.activation(self.linear2(feature))
        feature = self.activation(self.linear3(feature))
        feature = self.activation(self.linear4(feature))
        feature = self.activation(self.linear5(feature))
        feature = self.activation(self.linear6(feature))
        feature = self.activation(self.linear7(feature))
        return self.output(feature)

    def forward(self, x: Tensor) -> Tensor:
        observers, dimensions, polarizations, susceptibilities = self._prepare_inputs(x)

        B_reduced = Bfield_homogeneous(
            observers=observers,
            dimensions=dimensions,
            polarizations=polarizations,
        )

        feature = self._construct_feature(
            observers=observers,
            dimensions=dimensions,
            polarizations=polarizations,
            susceptibilities=susceptibilities,
        )

        prediction = self._forward(feature)

        return self.correct_ansatz(B_reduced=B_reduced, prediction=prediction)

    def _train_step(self, train_loader, criterion, optimizer):
        self.train()

        history = []
        for X, B in train_loader:
            # get the prediction (forward pass)
            B_demag = B[..., :3]
            B_corrected = self(X)

            # calculate loss
            loss = criterion(B_demag, B_corrected)
            history.append(loss.item())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return np.mean(history)

    def _valid_step(self, valid_loader, criterion):
        self.eval()
        with torch.no_grad():
            history = []
            angle_errors = []
            amplitude_errors = []

            for X, B in valid_loader:
                # get the prediction (forward pass)
                B_demag = B[..., :3]
                B_corrected = self(X)

                # calculate loss
                loss = criterion(B_demag, B_corrected)
                history.append(loss.item())

                # calculate eval metrics
                angle_err, amp_err = self._calculate_metrics(B_demag, B_corrected)
                angle_errors.append(angle_err)
                amplitude_errors.append(amp_err)

            return np.mean(history), np.mean(angle_errors), np.mean(amplitude_errors)

    def fit(self, train_loader, valid_loader, criterion, optimizer, epochs):
        train_losses = []
        validation_losses = []
        angle_errors = []
        amp_errors = []

        self.best_loss = np.inf
        for _ in tqdm.tqdm(range(epochs), unit="epochs"):
            train_loss = self._train_step(train_loader, criterion, optimizer)

            (
                validation_loss,
                angle_error,
                amplitude_error,
            ) = self._valid_step(valid_loader, criterion)

            if self.save_weights and validation_loss < self.best_loss:
                self.best_weights = deepcopy(self).state_dict()
                self.save()

            assert self.lr_scheduler is not None
            self.lr_scheduler.step()

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            angle_errors.append(angle_error)
            amp_errors.append(amplitude_error)

            if wandb.run is not None:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "angle_error": angle_error,
                        "amplitude_error": amplitude_error,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                )

        return train_losses, validation_losses, angle_errors, amp_errors

    def evaluate_model(self, eval_loader, criterion):
        self._valid_step(eval_loader, criterion)

    @staticmethod
    def _calculate_metrics(B_demag, B_corrected):
        angle_errors = angle_error(B_demag, B_corrected)
        amp_errors = relative_amplitude_error(B_demag, B_corrected, return_abs=True)
        return torch.mean(angle_errors), torch.mean(amp_errors)

    @staticmethod
    def correct_ansatz(B_reduced: Tensor, prediction: Tensor) -> Tensor:
        raise NotImplementedError()

    @classmethod
    def load_from_path(cls, path, *, activation, save_weights, save_path):
        raise NotImplementedError()

    def save(self):
        if self.save_path is not None:
            torch.save(self.best_weights, self.save_path / "best_weights.pt")
        else:
            warnings.warn(
                "You have tried to save a model without specifying a save path"
            )

    @staticmethod
    def _prepare_inputs(x: Tensor) -> tuple[Tensor, ...]:
        n_samples = x.shape[0]

        # extract the spatial coordinates and make a new tensor with gradients
        observers = x[:, 5:].clone().requires_grad_(True)

        # make tensor containing dimensions of each magnet in the batch
        dimensions = torch.concat((x[:, :2], torch.ones((n_samples, 1))), dim=1)

        # calculate the reduced polarizations from the dimensions and χ_z
        J_z = 1 / (1 + x[:, 4] * Dz_cuboid(dimensions)).unsqueeze(-1)
        polarizations = torch.concat(
            (torch.zeros((n_samples, 1)), torch.zeros((n_samples, 1)), J_z),
            dim=1,
        )

        # make tensor containing the susceptibilities for each magnet
        susceptibilities = x[:, 2:5].clone()

        return observers, dimensions, polarizations, susceptibilities

    @staticmethod
    def _construct_feature(
        observers,
        dimensions,
        polarizations,
        susceptibilities,
    ) -> Tensor:
        return torch.concat(
            (
                dimensions[..., :2],  # a, b
                susceptibilities,  # chi_x, chi_y, chi_z
                observers / dimensions,  # x/a, y/b, z
            ),
            dim=1,
        )

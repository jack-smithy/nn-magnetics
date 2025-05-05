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


class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Activation,
        p: float,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation
        self.dropout = nn.Dropout(p=p)
        self.layernorm = nn.LayerNorm(out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.layernorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class BaseNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Activation = F.silu,
        p: float = 0.2,
        save_weights: bool = True,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            FeedForward(in_features, 128, activation, p),
            FeedForward(128, 48, activation, p),
            FeedForward(48, 48, activation, p),
            FeedForward(48, 48, activation, p),
            FeedForward(48, 48, activation, p),
            FeedForward(48, 48, activation, p),
            FeedForward(48, 128, activation, p),
            nn.Linear(128, out_features),
        )

        self.best_weights = deepcopy(self).state_dict()
        self.save_path = save_path
        self.lr_scheduler = lr_scheduler
        self.save_weights = save_weights

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

        prediction = self.layers(feature)

        return self.correct_ansatz(B_reduced=B_reduced, prediction=prediction)

    def _train_step(self, train_loader: tuple[Tensor, Tensor], criterion, optimizer):
        self.train()

        history = []
        angle_errors = []
        amplitude_errors = []
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

            angle_err, amp_err = self._calculate_metrics(
                B_demag.detach(),
                B_corrected.detach(),
            )

            angle_errors.append(angle_err)
            amplitude_errors.append(amp_err)

        return np.mean(history), np.mean(angle_errors), np.mean(amplitude_errors)

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

    def fit(
        self,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        epochs,
    ):
        train_losses = []
        validation_losses = []
        train_angle_errors = []
        train_amp_errors = []
        validation_angle_errors = []
        validation_amp_errors = []

        self.best_loss = np.inf
        for _ in tqdm.tqdm(range(epochs), unit="epochs"):
            (
                train_loss,
                train_angle_error,
                train_amplitude_error,
            ) = self._train_step(train_loader, criterion, optimizer)

            (
                validation_loss,
                validation_angle_error,
                validation_amplitude_error,
            ) = self._valid_step(valid_loader, criterion)

            if self.save_weights and validation_loss < self.best_loss:
                self.best_weights = deepcopy(self).state_dict()
                self.save()

            assert self.lr_scheduler is not None
            self.lr_scheduler.step()

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            train_angle_errors.append(train_angle_error)
            train_amp_errors.append(train_amplitude_error)
            validation_angle_errors.append(validation_angle_error)
            validation_amp_errors.append(validation_amplitude_error)

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "validation/loss": validation_loss,
                        "train/angle_error": train_angle_error,
                        "train/amplitude_error": train_amplitude_error,
                        "validation/angle_error": validation_angle_error,
                        "validation/amplitude_error": validation_amplitude_error,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                )

        return (
            train_losses,
            validation_losses,
            validation_angle_errors,
            validation_amp_errors,
        )

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

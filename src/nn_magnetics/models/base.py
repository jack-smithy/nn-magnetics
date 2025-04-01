from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import tqdm
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler

import wandb
from nn_magnetics.utils.metrics import angle_error, relative_amplitude_error
from nn_magnetics.utils.physics import batch_rotation_matrices

type Activation = Callable[[torch.Tensor], torch.Tensor]


class Network(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim_factor: int,
        out_features: int,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Activation = F.silu,
        do_output_activation=True,
        save_weights: bool = True,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(
            in_features=in_features,
            out_features=4 * hidden_dim_factor,
        )
        self.linear2 = nn.Linear(
            in_features=4 * hidden_dim_factor,
            out_features=8 * hidden_dim_factor,
        )
        self.linear3 = nn.Linear(
            in_features=8 * hidden_dim_factor,
            out_features=4 * hidden_dim_factor,
        )
        self.linear4 = nn.Linear(
            in_features=4 * hidden_dim_factor,
            out_features=2 * hidden_dim_factor,
        )
        self.linear5 = nn.Linear(
            in_features=2 * hidden_dim_factor,
            out_features=hidden_dim_factor,
        )
        self.output = nn.Linear(
            in_features=hidden_dim_factor,
            out_features=out_features,
        )

        self.activation = activation
        self.do_output_activation = do_output_activation
        self.best_weights = deepcopy(self).state_dict()
        self.save_path = save_path
        self.lr_scheduler = lr_scheduler
        self.save_weights = save_weights

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def _train_step(self, train_loader, criterion, optimizer):
        self.train()

        history = []
        divergences = []
        for X, B in train_loader:
            # get the prediction (forward pass)
            B_demag = B[..., :3]
            B_predicted, divB = self(X)

            # calculate loss
            loss = criterion(B_demag, B_predicted, divB)
            history.append(loss.item())
            divergences.append(divB.mean().item())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return np.mean(history), np.mean(divergences)

    def _valid_step(self, valid_loader, criterion):
        self.eval()

        history = []
        angle_errors = []
        amplitude_errors = []
        divergences = []

        for X, B in valid_loader:
            # get the prediction (forward pass)
            B_demag = B[..., :3]
            B_predicted, divB = self(X)

            # calculate loss
            loss = criterion(B_demag, B_predicted, divB)
            history.append(loss.item())
            divergences.append(divB.mean().item())

            # calculate eval metrics
            angle_err, amp_err = self._calculate_metrics(B_demag, B_predicted)
            angle_errors.append(angle_err.detach().numpy())
            amplitude_errors.append(amp_err.detach().numpy())

        return (
            np.mean(history),
            np.mean(angle_errors),
            np.mean(amplitude_errors),
            np.mean(divergences),
        )

    def fit(self, train_loader, valid_loader, criterion, optimizer, epochs):
        train_losses = []
        validation_losses = []
        angle_errors = []
        amp_errors = []
        train_divergences = []
        validation_divergences = []

        self.best_loss = np.inf
        for _ in tqdm.tqdm(range(epochs), unit="epochs"):
            train_loss, train_divergence = self._train_step(
                train_loader, criterion, optimizer
            )

            (
                validation_loss,
                angle_error,
                amplitude_error,
                validation_divergence,
            ) = self._valid_step(valid_loader, criterion)

            if self.save_weights and validation_loss < self.best_loss:
                self.best_weights = deepcopy(self).state_dict()
                self.save()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            angle_errors.append(angle_error)
            amp_errors.append(amplitude_error)
            train_divergences.append(train_divergence)
            validation_divergences.append(validation_divergence)

            if wandb.run is not None:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "angle_error": angle_error,
                        "amplitude_error": amplitude_error,
                    }
                )

        return (
            train_losses,
            validation_losses,
            angle_errors,
            amp_errors,
            train_divergences,
            validation_divergences,
        )

    def evaluate_model(self, eval_loader, criterion):
        self._valid_step(eval_loader, criterion)

    def _calculate_metrics(self, B_demag, B_corrected):
        angle_errors = angle_error(B_demag, B_corrected)
        amp_errors = relative_amplitude_error(B_demag, B_corrected, return_abs=True)
        return torch.mean(angle_errors), torch.mean(amp_errors)

    def correct_ansatz(self, B_reduced: Tensor, prediction: Tensor) -> Tensor:
        raise NotImplementedError()

    @classmethod
    def load_from_path(cls, path, hidden_dim_factor):
        raise NotImplementedError()

    def save(self):
        if self.save_path is not None:
            torch.save(self.best_weights, self.save_path / "best_weights.pt")
        else:
            warnings.warn(
                "You have tried to save a model without specifying a save path"
            )


class GNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim_factor: int,
        out_features: int,
        save_path: Path | None = None,
        lr_scheduler: LRScheduler | None = None,
        activation: Activation = F.silu,
        do_output_activation=True,
        save_weights: bool = True,
    ) -> None:
        super().__init__()

        self.gcnconv1 = GATConv(
            in_channels=in_features,
            out_channels=4 * hidden_dim_factor,
        )

        self.gcnconv2 = GATConv(
            in_channels=4 * hidden_dim_factor,
            out_channels=8 * hidden_dim_factor,
        )
        self.gcnconv3 = GATConv(
            in_channels=8 * hidden_dim_factor,
            out_channels=4 * hidden_dim_factor,
        )

        # self.linear3 = nn.Linear(
        #     in_features=8 * hidden_dim_factor,
        #     out_features=4 * hidden_dim_factor,
        # )
        # self.linear4 = nn.Linear(
        #     in_features=4 * hidden_dim_factor,
        #     out_features=2 * hidden_dim_factor,
        # )
        self.linear5 = nn.Linear(
            in_features=4 * hidden_dim_factor,
            out_features=hidden_dim_factor,
        )
        self.output = nn.Linear(
            in_features=hidden_dim_factor,
            out_features=out_features,
        )

        self.activation = activation
        self.do_output_activation = do_output_activation
        self.best_weights = deepcopy(self).state_dict()
        self.save_path = save_path
        self.lr_scheduler = lr_scheduler
        self.save_weights = save_weights

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.activation(self.gcnconv1(x, edge_index))
        x = self.activation(self.gcnconv2(x, edge_index))
        x = self.activation(self.gcnconv3(x, edge_index))
        x = self.activation(self.linear5(x))

        if self.do_output_activation:
            return self.activation(self.output(x))

        return self.output(x)

    def _train_step(self, train_loader, criterion, optimizer):
        self.train()

        history = []
        for data in train_loader:
            X = data.x
            B = data.y
            edge_index = data.edge_index
            # get the prediction (forward pass)
            B_demag, B_reduced = B[..., :3], B[..., 3:]
            B_prediction = self(X, edge_index)
            B_corrected = self.correct_ansatz(B_reduced, B_prediction)

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

            for data in valid_loader:
                X = data.x
                B = data.y
                edge_index = data.edge_index
                B_demag, B_reduced = B[..., :3], B[..., 3:]
                B_prediction = self(X, edge_index)
                B_corrected = self.correct_ansatz(B_reduced, B_prediction)

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

            if self.lr_scheduler is not None:
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
                    }
                )

        return train_losses, validation_losses, angle_errors, amp_errors

    def evaluate_model(self, eval_loader, criterion):
        self._valid_step(eval_loader, criterion)

    def _calculate_metrics(self, B_demag, B_corrected):
        angle_errors = angle_error(B_demag, B_corrected)
        amp_errors = relative_amplitude_error(B_demag, B_corrected, return_abs=True)
        return torch.mean(angle_errors), torch.mean(amp_errors)

    def correct_ansatz(self, B_reduced: Tensor, prediction: Tensor) -> Tensor:
        assert prediction.shape[1] == 4

        angles = prediction[..., :3]
        amplitudes = prediction[..., 3]

        Rs_t = batch_rotation_matrices(angles)

        # Multiply each vector in B_reduced by the corresponding rotation matrix in Rs
        return amplitudes[:, None] * torch.einsum("nij,nj->ni", Rs_t, B_reduced)

    @classmethod
    def load_from_path(cls, path, hidden_dim_factor):
        raise NotImplementedError()

    def save(self):
        if self.save_path is not None:
            torch.save(self.best_weights, self.save_path / "best_weights.pt")
        else:
            warnings.warn(
                "You have tried to save a model without specifying a save path"
            )

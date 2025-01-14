from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
import tqdm
from pathlib import Path
from torch import nn
from copy import deepcopy

from nn_magnetics.corrections import angle_amp_correction
from nn_magnetics.utils.metrics import angle_error, relative_amplitude_error

type Activation = Callable[[torch.Tensor], torch.Tensor]


class Network(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim_factor: int,
        out_features: int,
        activation: Activation = F.silu,
        do_output_activation=True,
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
            in_features=1 * hidden_dim_factor,
            out_features=out_features,
        )
        self.activation = activation
        self.do_output_activation = do_output_activation
        self.best_weights = deepcopy(self).state_dict()

    def forward(
        self,
        x: Float[torch.Tensor, "in_features batch"],
    ) -> Float[torch.Tensor, "out_features batch"]:
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = self.activation(self.linear5(x))

        if self.do_output_activation:
            return self.activation(self.output(x))

        return self.output(x)

    def _train_step(self, train_loader, criterion, optimizer):
        self.train()

        history = []
        for X, B in train_loader:
            B_demag, B_reduced = B[..., :3], B[..., 3:]
            prediction = self(X)
            loss = criterion(B_demag, self.correct_ansatz(B_reduced, prediction))
            history.append(loss.item())
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
                B_demag, B_reduced = B[..., :3], B[..., 3:]
                prediction = self(X)
                B_corrected = self.correct_ansatz(B_reduced, prediction)
                loss = criterion(B_demag, B_corrected).item()

                angle_err, amp_err = self._calculate_metrics(B_demag, B_corrected)

                angle_errors.append(angle_err)
                amplitude_errors.append(amp_err)
                history.append(loss)

            return np.mean(history), np.mean(angle_errors), np.mean(amplitude_errors)

    def train_model(self, train_loader, valid_loader, criterion, optimizer, epochs):
        train_losses = []
        validation_losses = []
        angle_errors = []
        amp_errors = []

        best_loss = np.inf
        for _ in tqdm.tqdm(range(epochs), unit="epochs"):
            train_loss = self._train_step(train_loader, criterion, optimizer)

            (
                validation_loss,
                angle_error,
                amplitude_error,
            ) = self._valid_step(valid_loader, criterion)

            if validation_loss < best_loss:
                self.best_weights = deepcopy(self).state_dict()

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            angle_errors.append(angle_error)
            amp_errors.append(amplitude_error)

        return train_losses, validation_losses, angle_errors, amp_errors

    def evaluate_model(self, eval_loader, criterion):
        self._valid_step(eval_loader, criterion)

    def _calculate_metrics(self, B_demag, B_corrected):
        angle_errors = angle_error(B_demag, B_corrected)
        amp_errors = relative_amplitude_error(B_demag, B_corrected)
        return torch.mean(angle_errors), torch.mean(amp_errors)

    def correct_ansatz(self, B_reduced, prediction):
        raise NotImplementedError()

    def save(self, path: str | Path):
        torch.save(self.best_weights, path)


class FieldCorrectionNetwork(Network):
    def correct_ansatz(self, B_reduced, prediction):
        return B_reduced * prediction


class AngleAmpCorrectionNetwork(Network):
    def correct_ansatz(self, B_reduced, prediction):
        assert prediction.shape[1] == 4
        angles, amplitudes = prediction[..., :3], prediction[..., 3]
        B_corrected = angle_amp_correction(B_reduced, angles, amplitudes)
        return B_corrected


def get_num_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return (
        sum(p.numel() for p in model.parameters())
        if trainable_only
        else sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

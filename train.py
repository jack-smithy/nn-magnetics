from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from metrics import angle_error, relative_amplitude_error


def calculate_metrics(B: torch.Tensor, B_pred: torch.Tensor):
    B_demag = B[..., :3].numpy()
    B_ana = B[..., 3:].numpy()

    batch_angle_errors = angle_error(B_demag, B_pred.numpy() * B_ana)
    batch_amplitude_errors = relative_amplitude_error(B_demag, B_pred.numpy() * B_ana)
    return batch_angle_errors, batch_amplitude_errors


def calculate_metrics_baseline(B: np.ndarray) -> Tuple[np.ndarray, ...]:
    B_demag = B[..., :3]
    B_ana = B[..., 3:]

    angle_errors = angle_error(B_ana, B_demag)
    amplitude_errors = relative_amplitude_error(B_ana, B_demag)

    return angle_errors, amplitude_errors


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
) -> float:
    model.train()
    batch_losses = []

    for X, y in dataloader:
        y_pred = model(X)
        loss = criterion(y, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    return np.mean(batch_losses, axis=0)


def test_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> float:
    model.eval()
    batch_losses = []

    with torch.no_grad():
        for X, B in dataloader:
            B_pred = model(X)
            loss = criterion(B, B_pred)
            batch_losses.append(loss.item())

    return np.mean(batch_losses, axis=0)


def validate(data, model, criterion):
    _, X_test, _, B_test = data
    model.eval()

    losses = []
    avg_angle_errors_baseline = []
    avg_amp_errors_baseline = []
    avg_angle_errors = []
    avg_amp_errors = []

    with torch.no_grad():
        for X, B in zip(torch.from_numpy(X_test), torch.from_numpy(B_test)):
            B_pred = model(X)
            loss = criterion(B, B_pred)
            losses.append(loss.item())

            angle_errs_baseline, amp_errs_baseline = calculate_metrics_baseline(
                B.numpy()
            )
            avg_angle_errors_baseline.append(np.mean(angle_errs_baseline, axis=0))
            avg_amp_errors_baseline.append(np.mean(amp_errs_baseline, axis=0))

            angle_errs, amp_errs = calculate_metrics(B, B_pred)
            avg_angle_errors.append(np.mean(angle_errs, axis=0))
            avg_amp_errors.append(np.mean(amp_errs, axis=0))

    return {
        "angle_errors_baseline": avg_angle_errors_baseline,
        "amp_errors_baseline": avg_amp_errors_baseline,
        "angle_errors": avg_angle_errors,
        "amp_errors": avg_amp_errors,
    }

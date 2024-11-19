import os
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

import wandb
from nn_magnetics.dataset import ChiMode, DemagData, get_data_parallel
from nn_magnetics.model import (
    Network,
    get_loss,
)
from nn_magnetics.plotting import (
    plot_heatmaps,
    plot_histograms,
    plot_loss,
)
from nn_magnetics.utils import calculate_metrics, get_device

np.random.seed(0)
torch.manual_seed(0)

DEVICE = get_device(use_accelerators=False)


def _train_one_epoch(
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


def _test_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float, float]:
    model.eval()
    batch_losses = []
    batch_angle_errs = []
    batch_amp_errs = []

    with torch.no_grad():
        for X, B in dataloader:
            B_pred = model(X)
            loss = criterion(B, B_pred)
            batch_losses.append(loss.item())
            bang, bamp = calculate_metrics(B, B_pred)
            batch_angle_errs.append(np.mean(bang, axis=0))
            batch_amp_errs.append(np.mean(bamp, axis=0))

    return (
        np.mean(batch_losses, axis=0),
        np.mean(batch_angle_errs, axis=0),
        np.mean(batch_amp_errs, axis=0),
    )


def train_isotropic(config):
    training_stats = {
        "train_loss": [],
        "test_loss": [],
        "angle_error": [],
        "amplitude_error": [],
    }

    print("Training model for isotropic chi")
    print(f"Saving data to {config["save_path"]}")

    X_train, B_train = get_data_parallel(
        f"{config["data_dir"]}/train",
        ChiMode.ISOTROPIC,
    )

    X_test, B_test = get_data_parallel(
        f"{config["data_dir"]}/test",
        ChiMode.ISOTROPIC,
    )

    train_dataloader = DataLoader(
        dataset=DemagData(
            X=X_train.reshape(-1, 6),
            y=B_train.reshape(-1, 6),
            device=DEVICE,
        ),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    test_dataloader = DataLoader(
        dataset=DemagData(
            X=X_test.reshape(-1, 6),
            y=B_test.reshape(-1, 6),
            device=DEVICE,
        ),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    criterion = get_loss(name=config["loss_fn"])()

    model = Network(
        in_features=6,
        hidden_dim_factor=config["hidden_dim_factor"],
        out_features=1,
        activation=F.silu,
    ).to(DEVICE, dtype=torch.float64)

    if wandb.run is not None:
        wandb.watch(models=model, criterion=criterion)

    opt = Adam(params=model.parameters(), lr=config["learning_rate"])
    scheduler = lr_scheduler.ExponentialLR(optimizer=opt, gamma=config["gamma"])

    best_test_loss = np.inf
    best_weights = deepcopy(model).state_dict()
    for _ in tqdm.tqdm(range(config["epochs"]), unit="epoch", disable=False):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=opt,
        )

        test_loss, test_angle_err, test_amp_err = _test_one_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
        )

        training_stats["test_loss"].append(test_loss)
        training_stats["train_loss"].append(train_loss)
        training_stats["angle_error"].append(test_angle_err)
        training_stats["amplitude_error"].append(test_amp_err)

        scheduler.step()

        if wandb.run is not None:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "angle_error": test_angle_err,
                    "amp_err": test_amp_err,
                }
            )

        if test_loss < best_test_loss:
            best_model = deepcopy(model)
            best_weights = best_model.state_dict()

    os.makedirs(config["save_path"])
    plot_loss(training_stats, config["save_path"], show_plot=False)
    torch.save(best_weights, f"{config["save_path"]}/weights.pt")


def train_anisotropic(config):
    training_stats = {
        "train_loss": [],
        "test_loss": [],
        "angle_error": [],
        "amplitude_error": [],
    }

    print("Training model for anisotropic chi")
    print(f"Saving data to {config["save_path"]}")

    X_train, B_train = get_data_parallel(
        f"{config["data_dir"]}/train",
        ChiMode.ANISOTROPIC,
    )

    X_test, B_test = get_data_parallel(
        f"{config["data_dir"]}/test_anisotropic",
        ChiMode.ANISOTROPIC,
    )

    train_dataloader = DataLoader(
        dataset=DemagData(
            X=X_train.reshape(-1, 7),
            y=B_train.reshape(-1, 6),
            device=DEVICE,
        ),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    test_dataloader = DataLoader(
        dataset=DemagData(
            X=X_test.reshape(-1, 7),
            y=B_test.reshape(-1, 6),
            device=DEVICE,
        ),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    criterion = get_loss(name=config["loss_fn"])()

    model = Network(
        in_features=7,
        hidden_dim_factor=config["hidden_dim_factor"],
        out_features=3,
        activation=F.silu,
    ).to(DEVICE, dtype=torch.float64)

    if wandb.run is not None:
        wandb.watch(models=model, criterion=criterion)

    opt = Adam(params=model.parameters(), lr=config["learning_rate"])
    scheduler = lr_scheduler.ExponentialLR(optimizer=opt, gamma=config["gamma"])

    best_test_loss = np.inf
    best_weights = deepcopy(model).state_dict()
    for _ in tqdm.tqdm(range(config["epochs"]), unit="epoch", disable=False):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=opt,
        )

        test_loss, test_angle_err, test_amp_err = _test_one_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
        )

        training_stats["test_loss"].append(test_loss)
        training_stats["train_loss"].append(train_loss)
        training_stats["angle_error"].append(test_angle_err)
        training_stats["amplitude_error"].append(test_amp_err)

        scheduler.step()

        if wandb.run is not None:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "angle_error": test_angle_err,
                    "amp_err": test_amp_err,
                }
            )

        if test_loss < best_test_loss:
            best_model = deepcopy(model)
            best_weights = best_model.state_dict()

    os.makedirs(config["save_path"])
    plot_loss(training_stats, config["save_path"], show_plot=False)
    torch.save(best_weights, f"{config["save_path"]}/weights.pt")


def evaluate_isotropic(config):
    print("Evaluating model for isotropic chi")

    X_test, B_test = get_data_parallel(
        f"{config["data_dir"]}/test",
        ChiMode.ISOTROPIC,
    )

    model = Network(
        in_features=6,
        hidden_dim_factor=config["hidden_dim_factor"],
        out_features=1,
    ).to(DEVICE, dtype=torch.float64)

    model.load_state_dict(
        torch.load(
            f"{config["save_path"]}/weights.pt",
            weights_only=True,
        )
    )

    plot_histograms(
        X_test=X_test,
        B_test=B_test,
        model=model,
        save_path=config["save_path"],
        show_plot=False,
        tag="isotropic",
    )

    plot_heatmaps(
        model=model,
        save_path=config["save_path"],
        tag="isotropic",
        chi_mode=ChiMode.ISOTROPIC,
        eval_path=f"{config["data_dir"]}/test/data_1.npz",
    )


def evaluate_anisotropic(config):
    print("Evaluating model for anisotropic chi")

    X_test, B_test = get_data_parallel(
        f"{config["data_dir"]}/test_anisotropic",
        ChiMode.ANISOTROPIC,
    )

    model = Network(
        in_features=7,
        hidden_dim_factor=config["hidden_dim_factor"],
        out_features=3,
    ).to(DEVICE, dtype=torch.float64)

    model.load_state_dict(
        torch.load(
            f"{config["save_path"]}/weights.pt",
            weights_only=True,
        )
    )

    plot_histograms(
        X_test=X_test,
        B_test=B_test,
        model=model,
        save_path=config["save_path"],
        show_plot=False,
        tag="anisotropic",
    )

    plot_heatmaps(
        model=model,
        save_path=config["save_path"],
        tag="anisotropic",
        chi_mode=ChiMode.ANISOTROPIC,
        eval_path=f"{config["data_dir"]}/test_anisotropic/data_1.npz",
    )

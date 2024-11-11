import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

import wandb
from src.dataset import ChiMode, DemagData, get_data_parallel
from src.model import CorrectionLoss, Network, FieldLoss
from src.plotting import plot_histograms, plot_loss
from src.train import (
    test_one_epoch,
    train_one_epoch,
    validate,
)
from src.utils import get_device

torch.manual_seed(0)
np.random.seed(0)

DEVICE = get_device(use_accelerators=False)


def run(epochs, batch_size, learning_rate, data_dir, save_path, log=False):
    training_stats = {
        "train_loss": [],
        "test_loss": [],
        "angle_error": [],
        "amplitude_error": [],
        "params": {
            "lr": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        },
    }

    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="test-isotropic-correction-loss",
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": "MLP-SiLU-double-width",
                "dataset": "data/",
                "epochs": epochs,
                "batch_size": batch_size,
            },
        )

    os.makedirs(save_path)

    X_train, B_train = get_data_parallel(f"{data_dir}/train", ChiMode.ISOTROPIC)
    X_test, B_test = get_data_parallel(f"{data_dir}/test", ChiMode.ISOTROPIC)

    train_dataloader = DataLoader(
        dataset=DemagData(
            X=X_train.reshape(-1, 6),
            y=B_train.reshape(-1, 6),
            device=DEVICE,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        dataset=DemagData(
            X=X_test.reshape(-1, 6),
            y=B_test.reshape(-1, 6),
            device=DEVICE,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = FieldLoss()

    model = Network(
        in_features=6,
        hidden_dim_factor=12,
        out_features=3,
        activation=F.silu,
    )

    opt = Adam(params=model.parameters(), lr=learning_rate)

    best_model = copy.deepcopy(model)
    best_weights = best_model.state_dict()

    best_loss = np.inf
    for _ in tqdm.tqdm(range(epochs), unit="epoch", disable=False):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=opt,
        )

        test_loss, test_angle_err, test_amp_err = test_one_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
        )

        training_stats["test_loss"].append(test_loss)
        training_stats["train_loss"].append(train_loss)
        training_stats["angle_error"].append(test_angle_err)
        training_stats["amplitude_error"].append(test_amp_err)

        if log:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "angle_error": test_angle_err,
                    "amp_err": test_amp_err,
                }
            )

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = copy.deepcopy(model)
            best_weights = best_model.state_dict()

    val_stats = validate(X_test, B_test, best_model, criterion)

    plot_loss(training_stats, save_path, show_plot=False)
    plot_histograms(val_stats, save_path=save_path, show_plot=False)

    torch.save(best_weights, f"{save_path}/weights.pt")

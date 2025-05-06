import datetime
import json
import os
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import wandb
from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import (
    QuaternionNet,
    FieldCorrectionNetwork,
    AngleAmpCorrectionNetwork,
    get_num_params,
)
from nn_magnetics.utils.plotting import (
    plot_heatmaps,
    plot_histograms,
    plot_training,
)

DEVICE = "cpu"
SAVE_PATH = Path(f"results/3dof_chi_v2/{str(datetime.datetime.now())}")

config = {
    "model": "AngleAmp",
    "epochs": 45,
    "batch_size": 268,
    "learning_rate": 0.00135,
    "gamma": 0.96,
    "activation": "silu",
    "loss": "l1",
    "weight_decay": 0,
    "p": 0,
}

activations = {"tanh": F.tanh, "silu": F.silu}
losses = {"l1": F.l1_loss, "mse": F.mse_loss}


def main():
    wandb.init(project="3dof_chi_v2", config=config)

    assert wandb.run is not None
    os.makedirs(SAVE_PATH, exist_ok=True)

    train_data = AnisotropicData("data/3dof_chi_v2/train", device=DEVICE)
    valid_data = AnisotropicData("data/3dof_chi_v2/validation", device=DEVICE)

    # train_loader = train_data.get_magnets()
    # valid_loader = valid_data.get_magnets()

    train_loader = DataLoader(
        train_data,
        batch_size=wandb.config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=wandb.config.batch_size,
        shuffle=True,
    )

    model = AngleAmpCorrectionNetwork(
        save_path=SAVE_PATH,
        activation=activations[wandb.config.activation],
        save_weights=True,
        p=wandb.config.p,
    ).to(torch.float64)

    config["num_params"] = get_num_params(model=model)
    with open(f"{SAVE_PATH}/config.json", "w+") as f:
        json.dump(config, f)

    optimizer = Adam(
        params=model.parameters(),
        lr=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
    )

    model.lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=wandb.config.gamma)

    train_losses, valid_losses, angle_errs, amp_errs = model.fit(
        train_loader,
        valid_loader,
        losses[wandb.config.loss],
        optimizer,
        wandb.config.epochs,
    )

    learning = {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "angle_errs": angle_errs,
        "amp_errs": amp_errs,
    }

    with open(f"{SAVE_PATH}/learning.json", "w+") as f:
        json.dump(learning, f)

    plot_training(train_losses, valid_losses, angle_errs, amp_errs, save_path=SAVE_PATH)

    X, B = valid_data.get_magnets()

    plot_histograms(X, B, model, SAVE_PATH, tag="_valid_set")

    X_mag, B_mag = X[1], B[1]

    plot_heatmaps(model, X_mag, B_mag, SAVE_PATH, tag=f"{1}")

    wandb.finish()


if __name__ == "__main__":
    main()

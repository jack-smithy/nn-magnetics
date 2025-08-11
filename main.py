import datetime
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import wandb
from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import (
    AngleAmpCorrectionNetwork,
    BaseNetwork,
    FieldCorrectionNetwork,
    QuaternionNet,
    AdditionCorrectionNetwork,
    SphericalCorrectionNetwork,
    get_num_params,
)

from nn_magnetics.utils.plotting import (
    plot_heatmaps,
    plot_training,
    plot_histograms_with_baseline,
)

DEVICE = "cpu"
DTYPE = torch.float32
PROJECT = "3dof_chi_spherical"
SAVE_PATH = Path(f"results/{PROJECT}/{str(datetime.datetime.now())}")

config = {
    "epochs": 50,
    "batch_size": 2048,
    "learning_rate": 0.001,
    "gamma": 0.95,
    "weight_decay": 0,
    "p": 0.00,
    "do_output_activation": False,
    "activation": "gelu",
    "loss": "l1",
    "network": "spherical",
    "size": "large",
}

losses = {"l1": F.l1_loss, "mse": F.mse_loss}

activations = {
    "silu": F.silu,
    "tanh": F.tanh,
    "gelu": F.gelu,
    "sigmoid": F.sigmoid,
}


def main():
    wandb.init(project=PROJECT, config=config)

    assert wandb.run is not None
    os.makedirs(SAVE_PATH, exist_ok=True)

    train_data = AnisotropicData(
        f"data/3dof_chi_v3/{wandb.config.size}/train",
        device=DEVICE,
        dtype=DTYPE,
    )

    valid_data = AnisotropicData(
        f"data/3dof_chi_v3/{wandb.config.size}/validation",
        device=DEVICE,
        dtype=DTYPE,
    )

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

    model = SphericalCorrectionNetwork(
        save_path=SAVE_PATH,
        activation=activations[wandb.config.activation],
        save_weights=True,
        p=wandb.config.p,
        do_output_activation=wandb.config.do_output_activation,
    ).to(DTYPE)

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
        F.l1_loss,
        optimizer,
        wandb.config.epochs,
    )

    plot_training(train_losses, valid_losses, angle_errs, amp_errs, save_path=SAVE_PATH)

    X, B = valid_data.get_magnets()

    plot_histograms_with_baseline(X, B, model, SAVE_PATH, tag="_valid_set")

    X_mag, B_mag = X[1], B[1]

    plot_heatmaps(model, X_mag, B_mag, SAVE_PATH, tag=f"{1}")

    wandb.finish()


if __name__ == "__main__":
    main()

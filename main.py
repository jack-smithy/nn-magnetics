import datetime
import json
import os
from pathlib import Path

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from wakepy import keep

import wandb
from nn_magnetics.data.dataset import AnisotropicData
from nn_magnetics.model import AngleAmpCorrectionNetwork
from nn_magnetics.utils.plotting import (
    plot_heatmaps,
    plot_histograms,
    plot_loss,
)

DEVICE = "cpu"
SAVE_PATH = Path(f"results/3dof_chi/{str(datetime.datetime.now())}")


def main():
    config = {
        "epochs": 150,
        "batch_size": 128,
        "learning_rate": 0.001,
        "shuffle_train": True,
        "hidden_dim_factor": 6,
        "weight_decay": 0,
        "optimizer": "adam",
        "lr_scheduler": "exponential",
        "gamma": 1,
    }

    wandb.init(project="3dof-chi", config=config)
    assert wandb.run is not None

    config["name"] = wandb.run.name

    os.makedirs(SAVE_PATH)
    with open(f"{SAVE_PATH}/config.json", "w+") as f:
        json.dump(config, f)

    train_data = AnisotropicData("data/3dof_chi/train", device=DEVICE)
    valid_data = AnisotropicData("data/3dof_chi/validation", device=DEVICE)

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=config["shuffle_train"],
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=config["batch_size"],
    )

    model = AngleAmpCorrectionNetwork(
        in_features=8,
        hidden_dim_factor=config["hidden_dim_factor"],
        out_features=4,
        save_path=SAVE_PATH,
    ).to(DEVICE)

    loss = nn.L1Loss()

    optimizer = Adam(
        params=model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    model.lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=config["gamma"])

    train_losses, valid_losses, angle_errs, amp_errs = model.fit(
        train_loader,
        valid_loader,
        loss,
        optimizer,
        config["epochs"],
    )

    wandb.finish()

    plot_loss(
        train_losses,
        valid_losses,
        angle_errs,
        amp_errs,
        config["epochs"],
        save_path=SAVE_PATH,
    )

    X, B = valid_data.get_magnets()
    plot_histograms(X, B, model, SAVE_PATH)

    plot_heatmaps(model, X[0], B[0], SAVE_PATH)


if __name__ == "__main__":
    with keep.running():
        main()

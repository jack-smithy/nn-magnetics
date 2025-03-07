import datetime
import json
import os
from pathlib import Path

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
    get_num_params,
)
from nn_magnetics.utils.plotting import (
    plot_heatmaps,
    plot_histograms,
    plot_loss,
)

DEVICE = "cpu"
SAVE_PATH = Path(f"results/3dof_chi_quaternion/{str(datetime.datetime.now())}")

config = {
    "epochs": 45,
    "batch_size": 268,
    "learning_rate": 0.00135,
    "hidden_dim_factor": 6,
    "gamma": 0.96,
}


def main():
    wandb.init(project="3dof-chi", config=config)

    assert wandb.run is not None
    os.makedirs(SAVE_PATH, exist_ok=True)

    with open(f"{SAVE_PATH}/config.json", "w+") as f:
        json.dump(config, f)

    train_data = AnisotropicData("data/3dof_chi/train", device=DEVICE)
    valid_data = AnisotropicData("data/3dof_chi/validation", device=DEVICE)
    test_data = AnisotropicData("data/3dof_chi/test")

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

    model = QuaternionNet(
        in_features=8,
        hidden_dim_factor=wandb.config.hidden_dim_factor,
        save_path=SAVE_PATH,
        activation=F.silu,
        save_weights=True,
        do_output_activation=True,
    ).to(torch.float64)

    num_params = get_num_params(model)
    print(f"Trainable parameters: {num_params}")

    loss = nn.L1Loss()
    optimizer = Adam(params=model.parameters(), lr=wandb.config.learning_rate)

    model.lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=wandb.config.gamma)

    train_losses, valid_losses, angle_errs, amp_errs = model.fit(
        train_loader,
        valid_loader,
        loss,
        optimizer,
        wandb.config.epochs,
    )

    del model

    model = QuaternionNet.load_from_path(
        SAVE_PATH / "best_weights.pt",
        wandb.config.hidden_dim_factor,
        activation=F.silu,
        save_path=None,
        save_weights=False,
        do_output_activation=True,
    ).to(torch.float64)

    plot_loss(
        train_losses,
        valid_losses,
        angle_errs,
        amp_errs,
        wandb.config.epochs,
        save_path=SAVE_PATH,
    )

    X, B = valid_data.get_magnets()

    X_test, B_test = test_data.get_magnets()

    plot_histograms(X, B, model, SAVE_PATH, tag="_valid_set")
    plot_histograms(X_test, B_test, model, SAVE_PATH, tag="_test_set")
    plot_heatmaps(model, X[0], B[0], SAVE_PATH)

    wandb.finish()


if __name__ == "__main__":
    main()

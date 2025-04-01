import datetime
import json
import os
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import wandb
from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import (
    DivergenceLoss,
    FieldCorrectionNetwork,
    get_num_params,
)
from nn_magnetics.utils.plotting import (
    plot_heatmaps,
    plot_histograms,
    plot_loss,
)

DEVICE = "cpu"
DTYPE = torch.float64
LOG = True
SAVE_PATH = (
    Path(f"results/3dof_chi_pinn/{str(datetime.datetime.now())}") if LOG else None
)

config = {
    "model": "AngleAmp",
    "epochs": 10,
    "batch_size": 395,
    "learning_rate": 0.077,
    "hidden_dim_factor": 6,
    "gamma": 0.9,
    "lambda_": 0.1,
    "architecture": "pinn",
}


def main():
    if LOG:
        wandb.init(project="3dof-chi", config=config)

        assert wandb.run is not None
        os.makedirs(SAVE_PATH, exist_ok=True)  # type: ignore

        with open(f"{SAVE_PATH}/config.json", "w+") as f:
            json.dump(config, f)

    train_data = AnisotropicData(
        "data/3dof_chi_v2/train_fast",
        device=DEVICE,
        dtype=DTYPE,
    )

    valid_data = AnisotropicData(
        "data/3dof_chi_v2/validation_fast",
        device=DEVICE,
        dtype=DTYPE,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    model = (
        FieldCorrectionNetwork(
            in_features=8,
            hidden_dim_factor=config["hidden_dim_factor"],
            save_path=SAVE_PATH,
            save_weights=True,
        )
        .to(DTYPE)
        .to(DEVICE)
    )

    print(f"Num params: {get_num_params(model)}")

    loss = DivergenceLoss(lambda_=config["lambda_"])
    optimizer = Adam(params=model.parameters(), lr=config["learning_rate"])

    model.lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=config["gamma"])

    train_losses, valid_losses, angle_errs, amp_errs = model.fit(
        train_loader,
        valid_loader,
        loss,
        optimizer,
        config["epochs"],
    )

    model.eval()

    plot_loss(
        train_losses,
        valid_losses,
        angle_errs,
        amp_errs,
        save_path=SAVE_PATH,
    )

    X, B = valid_data.get_magnets()

    plot_histograms(X, B, model, save_path=SAVE_PATH, tag="_valid_set")

    for i in range(2):
        X_mag, B_mag = X[i], B[i]

        plot_heatmaps(model, X_mag, B_mag, save_path=SAVE_PATH, tag=f"{i}")

    if LOG:
        wandb.finish()


if __name__ == "__main__":
    main()

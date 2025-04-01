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
    DivergenceLoss,
)
from nn_magnetics.utils.plotting import (
    plot_heatmaps,
    plot_histograms,
    plot_loss,
    plot_div_loss,
)

DEVICE = "cpu"
DTYPE = torch.float64
LOG = False
SAVE_PATH = (
    Path(f"results/3dof_chi_resnet/{str(datetime.datetime.now())}") if LOG else None
)

config = {
    "model": "AngleAmp",
    "epochs": 10,
    "batch_size": 350,
    "learning_rate": 0.008,
    "hidden_dim": 6,
    "n_residual_blocks": 10,
    "gamma": 0.985,
    "p": 0.2,
    "lambda": 18,
    "do_output_activation": False,
    "activation": "silu",
    "architecture": "mlp",
}

activations = {"silu": F.silu, "tanh": F.tanh}


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
            hidden_dim_factor=config["hidden_dim"],
            save_path=SAVE_PATH,
            do_output_activation=config["do_output_activation"],
            activation=activations[config["activation"]],
            save_weights=False,
        )
        .to(DTYPE)
        .to(DEVICE)
    )

    print(f"Num params: {get_num_params(model)}")

    loss = DivergenceLoss(lambda_=config["lambda"])
    optimizer = Adam(params=model.parameters(), lr=config["learning_rate"])

    model.lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=config["gamma"])

    train_losses, valid_losses, angle_errs, amp_errs, train_div, valid_div = model.fit(
        train_loader,
        valid_loader,
        loss,
        optimizer,
        config["epochs"],
    )

    if LOG:
        with open(f"{SAVE_PATH}/learning.json", "w+") as f:
            json.dump(
                {
                    "train_losses": list(train_losses),
                    "valid_losses": list(valid_losses),
                    "angle_errs": list(angle_errs),
                    "amp_errs": list(amp_errs),
                },
                f,
            )

    if LOG:
        del model

        model = (
            QuaternionNet.load_from_path(
                path=SAVE_PATH / "best_weights.pt",  # type: ignore
                hidden_dim_factor=config["hidden_dim"],
                activation=activations[config["activation"]],
                save_path=None,
                do_output_activation=config["do_output_activation"],
                save_weights=False,
            )
            .to(DTYPE)
            .to(DEVICE)
        )

    model.eval()

    plot_loss(
        train_losses,
        valid_losses,
        angle_errs,
        amp_errs,
        save_path=SAVE_PATH,
    )

    plot_div_loss(train_div=train_div, valid_div=valid_div)

    X, B = valid_data.get_magnets()

    plot_histograms(X, B, model, save_path=SAVE_PATH, tag="_valid_set")

    for i in range(2):
        X_mag, B_mag = X[i], B[i]

        plot_heatmaps(model, X_mag, B_mag, save_path=SAVE_PATH, tag=f"{i}")

    if LOG:
        wandb.finish()


if __name__ == "__main__":
    main()

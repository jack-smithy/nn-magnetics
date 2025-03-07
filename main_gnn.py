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
from nn_magnetics.data import AnisotropicData, get_graphs, get_graphs_batched
from nn_magnetics.models import QuaternionNet, AngleAmpCorrectionNetwork, GNN
from nn_magnetics.utils.plotting import (
    plot_heatmaps,
    plot_histograms,
    plot_loss,
    plot_histograms_gnn,
)

DEVICE = "cpu"
SAVE_PATH = Path(f"results/3dof_chi_gnn/{str(datetime.datetime.now())}")

config = {
    "epochs": 5,
    "learning_rate": 0.036,
    "hidden_dim_factor": 12,
    "gamma": 0.95,
    "activation": "silu",
}

# sweep_config = {
#     "method": "bayes",
#     "metric": {"name": "validation_loss", "goal": "minimize"},
#     "parameters": {
#         "learning_rate": {"min": 0.0001, "max": 0.1},
#         "hidden_dim_factor": {"values": [6, 12, 24, 48]},
#         "epochs": {"value": 15},
#         "gamma": {"min": 0.9, "max": 1.0},
#         "do_output_activation": {"values": [True, False]},
#         "activation": {"values": ["tanh", "silu"]},
#     },
# }

activations = {"silu": F.silu, "tanh": F.tanh}

# sweep_id = wandb.sweep(sweep_config, project="3dof-chi-gnn")


def main():
    wandb.init(project="3dof-chi-gnn", config=config)
    # wandb.init()

    # assert wandb.run is not None
    # os.makedirs(SAVE_PATH, exist_ok=True)

    # with open(f"{SAVE_PATH}/config.json", "w+") as f:
    #     json.dump(config, f)

    train_loader = get_graphs(Path("data/3dof_chi_graph/train_fast"))
    valid_loader = get_graphs(Path("data/3dof_chi_graph/validation_fast"))

    model = GNN(
        in_features=8,
        out_features=4,
        hidden_dim_factor=wandb.config.hidden_dim_factor,
        save_path=None,
        activation=activations[wandb.config.activation],
        save_weights=False,
        do_output_activation=False,
    ).to(torch.float64)

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

    plot_loss(
        train_losses,
        valid_losses,
        angle_errs,
        amp_errs,
        wandb.config.epochs,
        save_path=None,
    )

    plot_histograms_gnn(valid_loader, model, None, tag="_valid_set")

    wandb.finish()


if __name__ == "__main__":
    # wandb.agent(sweep_id=sweep_id, function=main, count=50)
    main()

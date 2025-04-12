import json
from pathlib import Path

import torch.nn.functional as F
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
    plot_loss,
)

DEVICE = "cpu"
SAVE_PATH = Path("results/paper/component")


def main():
    with open(f"{SAVE_PATH}/config.json") as f:
        config = json.load(f)

    model = FieldCorrectionNetwork.load_from_path(
        SAVE_PATH / "best_weights.pt",
        hidden_dim_factor=config["hidden_dim_factor"],
        activation=F.silu,
        do_output_activation=True,
    ).to(torch.float64)

    with open(f"{SAVE_PATH}/learning.json") as f:
        data = json.load(f)

    plot_loss(
        data["train_losses"],
        data["valid_losses"],
        # data["angle_errs"],
        # data["amp_errs"],
        save_path=None,
        tag="alt1",
    )

    # valid_data = AnisotropicData("data/3dof_chi/validation", device=DEVICE)

    # X, B = valid_data.get_magnets()

    # plot_histograms(X, B, model, SAVE_PATH, tag="_alt")

    # X_mag, B_mag = X[1], B[1]

    # plot_heatmaps(model, X_mag, B_mag, SAVE_PATH, tag="_alt")


if __name__ == "__main__":
    main()

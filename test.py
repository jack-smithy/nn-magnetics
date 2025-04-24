import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import (
    AngleAmpCorrectionNetwork,
    FieldCorrectionNetwork,
    QuaternionNet,
)
from nn_magnetics.utils.plotting import (
    plot_heatmaps,
    plot_histograms,
    plot_loss,
    plot_training,
)

DEVICE = "cpu"
SAVE_PATH = Path("results/paper/quaternion")


def main():
    # with open(f"{SAVE_PATH}/config.json") as f:
    #     config = json.load(f)

    model = QuaternionNet.load_from_path(
        SAVE_PATH / "best_weights.pt",
        hidden_dim_factor=6,
        activation=F.silu,
        do_output_activation=True,
    ).to(torch.float64)

    # train_losses = []
    # validation_losses = []
    # angle_errors = []
    # amplitude_errors = []

    # with open(f"{SAVE_PATH}/component/learning.json") as f:
    #     datac = json.load(f)

    #     train_losses.append(datac["train_losses"])
    #     validation_losses.append(datac["valid_losses"])
    #     angle_errors.append(datac["angle_errs"])
    #     amplitude_errors.append(datac["amp_errs"])

    # with open(f"{SAVE_PATH}/euler/learning.json") as f:
    #     datae = json.load(f)

    #     train_losses.append(datae["train_losses"])
    #     validation_losses.append(datae["valid_losses"])
    #     angle_errors.append(datae["angle_errs"])
    #     amplitude_errors.append(datae["amp_errs"])

    # with open(f"{SAVE_PATH}/quaternion/learning.json") as f:
    #     dataq = json.load(f)

    #     train_losses.append(dataq["train_losses"])
    #     validation_losses.append(dataq["valid_losses"])
    #     angle_errors.append(dataq["angle_errs"])
    #     amplitude_errors.append(dataq["amp_errs"])

    # plot_loss(
    #     train_loss=train_losses,
    #     validation_loss=validation_losses,
    #     angle_error=angle_errors,
    #     amp_error=amplitude_errors,
    #     save_path=f"{SAVE_PATH}",
    # )

    # plot_loss(
    #     data["train_losses"],
    #     data["valid_losses"],
    #     save_path=f"{SAVE_PATH}/pdfs",
    # )

    # X, B = AnisotropicData("data/3dof_chi/validation").get_magnets()
    # plot_histograms(
    #     X=X,
    #     B=B,
    #     model=model,
    #     save_path=f"{SAVE_PATH}",
    #     figsize=(6, 4),
    #     tag="_no_baseline",
    # )

    X_mag, B_mag = AnisotropicData(
        "data/3dof_chi/one",
        device=DEVICE,
    ).get_magnets()

    print(X_mag[0, 0, :])

    # plot_heatmaps(model, X_mag[0], B_mag[0], f"{SAVE_PATH}/pdfs", tag="_no_baseline")


if __name__ == "__main__":
    main()

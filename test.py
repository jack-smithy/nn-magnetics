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
    plot_baseline_histograms,
)
from nn_magnetics.utils.metrics import (
    calculate_metrics_baseline,
    calculate_metrics_trained,
    vector_field_correlation,
)
import matplotlib.pyplot as plt

DEVICE = "cpu"
SAVE_PATH = Path(
    "/Users/jacksmith/Documents/work/nn-magnetics/results/3dof_chi_v2/2025-04-25 10:27:16.283326"
)


def main():
    # with open(f"{SAVE_PATH}/config.json") as f:
    #     config = json.load(f)

    model = AngleAmpCorrectionNetwork.load_from_path(
        SAVE_PATH / "best_weights.pt",
        activation=F.silu,
        save_path=None,
        save_weights=False,
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

    X, B = AnisotropicData("data/3dof_chi/validation_medium").get_magnets()

    angle_errs = []
    amp_errs = []
    angle_errs_baseline = []
    amp_errs_baseline = []
    for x, b in zip(X, B):
        angle_err_baseline, amp_err_baseline = calculate_metrics_baseline(b)
        angle_err, amp_err = calculate_metrics_trained(x, b, model)

        angle_errs.append(angle_err.mean())
        amp_errs.append(amp_err.mean())
        angle_errs_baseline.append(angle_err_baseline.mean())
        amp_errs_baseline.append(amp_err_baseline.mean())

    angle_errs_t = torch.stack(angle_errs)
    amp_errs_t = torch.stack(amp_errs)
    angle_errs_b_t = torch.stack(angle_errs_baseline)
    amp_errs_b_t = torch.stack(amp_errs_baseline)

    print(angle_errs_t.mean().item())
    print(amp_errs_t.mean().item())

    print(angle_errs_b_t.mean().item())
    print(amp_errs_b_t.mean().item())

    # X_mag, B_mag = AnisotropicData(
    #     "data/3dof_chi/one",
    #     device=DEVICE,
    # ).get_magnets()

    # plot_heatmaps(model, X_mag[0], B_mag[0], f"{SAVE_PATH}/pdfs", tag="_no_baseline")


if __name__ == "__main__":
    main()

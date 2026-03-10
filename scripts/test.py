from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import (
    FieldCorrectionNetwork,
    SphericalCorrectionNetwork,
    NoCorrectionNetwork,
)
from nn_magnetics.utils.plotting import (
    plot_baseline_histograms,
)
import matplotlib.pyplot as plt
from nn_magnetics.utils.metrics import (
    calculate_metrics_baseline,
    calculate_metrics_trained,
)
from torch.profiler import profile, ProfilerActivity, record_function

DEVICE = "cpu"
# SAVE_PATH = Path(
#     "/Users/jacksmith/Documents/work/nn-magnetics/results/paper_v2/spherical"
# )

SAVE_PATH = Path(
    "/Users/jacksmith/Documents/Work/nn-magnetics/results/3dof_chi_spherical/2026-02-23 15:09:51.400856"
)
torch.set_printoptions(precision=10)


def r2_score_global(y_true, y_pred):
    # y_true = y_true.reshape(-1)
    # y_pred = y_pred.reshape(-1)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def main():
    # with open(f"{SAVE_PATH}/config.json") as f:
    #     config = json.load(f)

    model = NoCorrectionNetwork.load_from_path(
        SAVE_PATH / "best_weights.pt",
        activation=F.gelu,
        save_path=None,
        save_weights=False,
        do_output_activation=False,
        p=0.0,
    )

    X, B = AnisotropicData(
        "data/3dof_chi_v3/large/validation",
        dtype=torch.float32,
    ).get_magnets()

    print(B.shape)

    angle_err_baseline, amp_err_baseline = calculate_metrics_baseline(B)
    print(angle_err_baseline.shape)

    angles, amps = [], []
    for Bi, Xi in zip(B, X):
        angle_err, amp_err = calculate_metrics_trained(Xi, Bi, model)
        angles.append(angle_err.mean())
        amps.append(amp_err.mean())

    mean_max_amp = np.mean(amps)
    mean_max_angle = np.mean(angles)

    print(mean_max_amp, mean_max_angle)

    print(
        "angle err model ",
        angle_err.mean().item(),
        "angle err ana ",
        angle_err_baseline.mean().item(),
    )
    print(
        "amp err model ",
        amp_err.mean().item(),
        "amp err ana ",
        amp_err_baseline.mean().item(),
    )

    X = X.reshape((-1, 8))
    B = B.reshape((-1, 6))

    B_trues, B_ana = B[:, :3], B[:, 3:]

    B_preds = model(X)

    mae_pred = torch.nn.functional.l1_loss(B_trues, B_preds) * 1000
    mae_ana = torch.nn.functional.l1_loss(B_trues, B_ana) * 1000

    r2_pred = r2_score_global(B_trues, B_preds)
    r2_ana = r2_score_global(B_trues, B_ana)

    print("mae model ", mae_pred.item(), "mae ana ", mae_ana.item())
    print("r2 model ", r2_pred.item(), "r2 ana ", r2_ana.item())


if __name__ == "__main__":
    main()

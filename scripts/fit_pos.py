import torch
from tqdm import tqdm
import numpy as np
from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import SphericalCorrectionNetwork
import matplotlib.pyplot as plt
from nn_magnetics.optimize.fit_lbfgs import (
    build_model,
)
from nn_magnetics.optimize.other import optimize_positions, optimize_positions_ana


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 16,
    }
)

labels = ["Component", "Euler", "Quaternion"]
colors_ = ["#784c99", "#6780be", "#e93ea5"]
markers = ["v", "o", "*"]


COMPONENT_WEIGHTS_PATH = "/Users/jacksmith/Documents/work/nn-magnetics/results/paper_v2/component/best_weights.pt"
SPHERICAL_WEIGHTS_PATH = "/Users/jacksmith/Documents/work/nn-magnetics/results/paper_v2/spherical/best_weights.pt"
DATA_PATH = (
    "/Users/jacksmith/Documents/work/nn-magnetics/data/3dof_chi_v3/small/validation"
)


def main():
    X, B = AnisotropicData(DATA_PATH).get_magnets()

    model_spherical = build_model(
        SphericalCorrectionNetwork,
        SPHERICAL_WEIGHTS_PATH,
        torch.nn.functional.gelu,
    )

    # model_component = build_model(
    #     FieldCorrectionNetwork,
    #     COMPONENT_WEIGHTS_PATH,
    #     torch.nn.functional.silu,
    # )

    true_positions = []
    # predicted_positions_component = []
    predicted_positions_spherical = []
    predicted_positions_ana = []

    offset = 4080
    steps = 15
    x, b = X[0, offset : offset + steps], B[0, offset : offset + steps]
    # for x, b in tqdm(zip(X, B)):

    print(x[:, 5:])

    for xi, bi in tqdm(zip(x, b)):
        observers = xi[5:].clone().requires_grad_(True)
        B_measured = bi[:3]

        susceptibility = x[0, 2:5]
        dimensions = x[0, :2]
        true_positions.append(observers.detach().numpy())

        predicted_pos_spherical, loss = optimize_positions(
            model=model_spherical,
            susceptibility=susceptibility,
            B_measured=B_measured,
            dimensions=dimensions,
            method="l-bfgs",
            options={"gtol": 1e-11},
            n_iter=1,
            p0=None,
        )

        predicted_pos_ana, _ = optimize_positions_ana(
            B_measured=B_measured,
            dimensions=dimensions,
            n_iter=1,
            p0=None,
        )

        predicted_positions_spherical.append(predicted_pos_spherical)
        predicted_positions_ana.append(predicted_pos_ana)

    true_positions = np.stack(true_positions)
    predicted_positions_spherical = np.stack(predicted_positions_spherical)
    predicted_positions_ana = np.stack(predicted_positions_ana)

    x_true, y_true, z_true = true_positions.T
    x_sph, y_sph, z_sph = predicted_positions_spherical.T
    x_ana, y_ana, z_ana = predicted_positions_ana.T

    # plt.plot(z_true, z_true, color=colors_[0], linestyle="dashed")

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(6, 4))

    # ax1.plot(
    #     z_true,
    #     np.abs(x_true - x_sph),
    #     color=colors_[1],
    #     marker=markers[1],
    # )

    # ax1.plot(
    #     z_true,
    #     np.abs(x_true - x_ana),
    #     color=colors_[2],
    #     marker=markers[2],
    # )

    # ax1.set_ylabel(r"$|x_{true} - x_{predicted}$ [mm]")
    # ax1.set_xlabel(r"$z_{true}$ [mm]")

    # ax2.plot(
    #     z_true,
    #     np.abs(y_true - y_sph),
    #     color=colors_[1],
    #     marker=markers[1],
    # )

    # ax2.plot(
    #     z_true,
    #     np.abs(y_true - y_ana),
    #     color=colors_[2],
    #     marker=markers[2],
    # )

    # ax2.set_ylabel(r"$|y_{true} - y_{predicted}$ [mm]")
    # ax2.set_xlabel(r"$z_{true}$ [mm]")

    ax.plot(
        z_true,
        np.abs(z_true - z_sph),
        color=colors_[1],
        marker=markers[1],
    )

    ax.plot(
        z_true,
        np.abs(z_true - z_ana),
        color=colors_[2],
        marker=markers[2],
    )

    ax.set_ylabel(r"$|z_{true} - z_{predicted}$ [mm]")
    ax.set_xlabel(r"$z_{true}$ [mm]")

    # plt.xlabel(r"$z_{true}$ [mm]")
    # plt.ylabel(r"$|z_{true} - z_{predicted}|$ [mm]")
    # plt.legend()
    plt.tight_layout()
    plt.show()

    # print(true_susc.tolist())
    # print(predicted_susc_spherical.tolist())

    # true_positions = torch.stack(true_positions)
    # predicted_positions_spherical = torch.stack(predicted_positions_spherical)
    # predicted_positions_component = torch.stack(predicted_positions_component)

    # loss_spherical = torch.nn.functional.mse_loss(
    #     true_positions,
    #     predicted_positions_spherical,
    # )
    # loss_component = torch.nn.functional.mse_loss(
    #     true_positions,
    #     predicted_positions_component,
    # )
    # print(f"Spherical Correction Loss: {loss_spherical.item()}")
    # print(f"Component Correction Loss: {loss_component.item()}")


if __name__ == "__main__":
    main()

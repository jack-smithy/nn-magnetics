from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nn_magnetics.models import SphericalCorrectionNetwork
from nn_magnetics.optimize.fit_lbfgs import (
    build_model,
    joystick_points_and_field,
    optimize_positions_joystick,
    optimize_positions_joystick_ana,
)

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
    model_spherical = build_model(
        SphericalCorrectionNetwork,
        SPHERICAL_WEIGHTS_PATH,
        torch.nn.functional.gelu,
    )

    r = 4
    d = 1
    n_points = 21
    dimensions = (1, 1, 1)
    n_magnets = 1

    susceptibilities = np.array([[0.3, 0.3, 0.05]])

    print(susceptibilities)

    true_angles_deg = np.empty((n_magnets, n_points))
    angle_errors_ana = np.empty((n_magnets, n_points))
    angle_errors_nn = np.empty((n_magnets, n_points))

    times = []

    for i, susc_i in enumerate(susceptibilities):
        true_angle = []
        predicted_angle_spherical = []
        predicted_angle_ana = []

        res = joystick_points_and_field(
            r=r,
            d=d,
            n_points=n_points,
            susceptibility=susc_i,
            dimension=dimensions,
            eps=0,
        )

        res = [torch.from_numpy(r) for r in res]
        thetas, pos, ang, obs, B_true = res
        dim = torch.tensor(dimensions[:2])
        susc = torch.tensor(susc_i)

        for theta, B in tqdm(zip(thetas, B_true)):
            true_angle.append(theta.detach().numpy())

            start = perf_counter()
            predicted_pos_spherical, loss = optimize_positions_joystick(
                model=model_spherical,
                susceptibility=susc,
                B_measured=B,
                dimensions=dim,
                r=r,
                d=d,
            )
            end = perf_counter()

            times.append((end - start))

            predicted_pos_ana, loss = optimize_positions_joystick_ana(
                B_measured=B,
                dimensions=dim,
                r=r,
                d=d,
            )

            predicted_angle_spherical.append(predicted_pos_spherical.item())
            predicted_angle_ana.append(predicted_pos_ana.item())

        true_angle = np.stack(true_angle)
        predicted_angle_spherical = np.stack(predicted_angle_spherical)
        predicted_angle_ana = np.stack(predicted_angle_ana)

        true_angle_deg = np.rad2deg(true_angle)
        angle_err = np.abs(np.rad2deg(true_angle - predicted_angle_spherical))
        angle_err_ana = np.abs(np.rad2deg(true_angle - predicted_angle_ana))

        true_angles_deg[i, :] = true_angle_deg
        angle_errors_ana[i, :] = angle_err_ana
        angle_errors_nn[i, :] = angle_err

    true_angles = true_angles_deg[0]

    mean_err_ana = np.mean(angle_errors_ana, axis=0)
    mean_err_nn = np.mean(angle_errors_nn, axis=0)

    print(mean_err_ana.mean())
    print(mean_err_nn.mean())

    print(f"Avg time for measurement NN: {np.mean(times)}")

    res = {
        "true_deg": true_angles,
        "nn_pred_deg": mean_err_nn,
        "ana_pred_deg": mean_err_ana,
    }

    df = pd.DataFrame(res)

    df.to_csv("./results/paper_v2/joystick/errors.dat", sep=" ")

    # std_err_ana = np.std(angle_errors_ana, axis=0)
    # std_err_nn = np.std(angle_errors_nn, axis=0)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6, 4),
        # gridspec_kw={
        #     "width_ratios": [2, 1],
        # },
    )
    ax.plot(
        true_angles,
        mean_err_nn,
        color=colors_[1],
        marker=markers[1],
        label="NN Correction",
    )
    ax.annotate("NN Solution", xy=(3, 0.02))
    # ax.fill_between(
    #     true_angles,
    #     mean_err_nn + std_err_nn,
    #     mean_err_nn - std_err_nn,
    #     color=colors_[1],
    #     alpha=0.5,
    # )

    ax.plot(
        true_angles,
        mean_err_ana,
        color=colors_[2],
        marker=markers[2],
        label="Analytical Model",
    )
    ax.annotate("Analytical Solution", xy=(8, 0.15))
    # ax.fill_between(
    #     true_angles,
    #     mean_err_ana + std_err_ana,
    #     mean_err_ana - std_err_ana,
    #     color=colors_[2],
    #     alpha=0.5,
    # )

    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_ylabel(r"$|\theta_{true} - \theta_{predicted}|$ (°)")
    ax.set_xlabel(r"$\theta_{true}$ (°)")
    # ax.set_ylim(top=1.2)

    ax.grid(alpha=0.5)

    # axins.plot(true_angle_deg, angle_err, color=colors_[1], marker=markers[1])
    # axins.plot(true_angle_deg, angle_err_ana, color=colors_[2], marker=markers[2])

    # axins.set_xlabel(r"$\theta_{true}$ (°)")
    # axins.set_xlim(12.5, 15)
    # axins.set_ylim(0, 0.02)

    # axins.xaxis.set_major_locator(MultipleLocator(1))
    # axins.yaxis.set_major_locator(MultipleLocator(0.005))

    # axins.grid(True, which="both", linestyle="--", linewidth=0.5)

    # axins.yaxis.tick_right()
    # axins.yaxis.set_label_position("right")

    # mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

    # plt.subplots_adjust(wspace=0.05, left=0.1, right=0.9, top=0.95, bottom=0.15)

    plt.tight_layout()

    plt.savefig("results/paper_v2/joystick/angle_err_v4.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    main()

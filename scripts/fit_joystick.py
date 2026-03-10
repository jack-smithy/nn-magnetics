from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from nn_magnetics.models import SphericalCorrectionNetwork, AnalyticalModel
from nn_magnetics.optimize.fit_lbfgs import (
    build_model,
    joystick_points_and_field,
    optimize_positions_joystick,
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


def test_susc(susceptibilities, dimensions, d, ax, r=4, n_points=21):
    model_spherical = build_model(
        SphericalCorrectionNetwork,
        SPHERICAL_WEIGHTS_PATH,
        torch.nn.functional.gelu,
    )

    model_analytical = AnalyticalModel(False)

    times = []

    true_angle = []
    predicted_angle_spherical = []
    predicted_angle_spherical_isotropic = []
    predicted_angle_ana = []

    res = joystick_points_and_field(
        r=r,
        d=d,
        n_points=n_points,
        susceptibility=susceptibilities,
        dimension=dimensions,
        eps=0,
    )

    res = [torch.from_numpy(r) for r in res]
    thetas, pos, ang, obs, B_true = res
    dim = torch.tensor(dimensions[:2])
    susc = torch.tensor(susceptibilities)

    for theta, B in tqdm(zip(thetas, B_true), total=B_true.shape[0]):
        true_angle.append(theta.detach().numpy())

        start = perf_counter()
        predicted_pos_spherical, loss = optimize_positions_joystick(
            model=model_spherical,
            susceptibility=susc,
            B_measured=B,
            dimensions=dim,
            r=r,
            d=d,
            maxiter=3,
        )
        end = perf_counter()

        predicted_pos_spherical_isotropic, loss = optimize_positions_joystick(
            model=model_spherical,
            susceptibility=torch.tensor(
                [susc[2] + 0.01, susc[2] + 0.01, susc[2] + 0.01]
            ),
            B_measured=B,
            dimensions=dim,
            r=r,
            d=d,
            maxiter=3,
        )

        times.append((end - start))

        predicted_pos_ana, loss = optimize_positions_joystick(
            model=model_analytical,
            susceptibility=torch.tensor(
                [susc[2] + 0.01, susc[2] + 0.01, susc[2] + 0.01]
            ),
            B_measured=B,
            dimensions=dim,
            r=r,
            d=d,
            maxiter=3,
            options={
                "gtol": 0.001,
                "xtol": 0.0001,
            },
        )

        predicted_angle_spherical.append(predicted_pos_spherical.item())
        predicted_angle_spherical_isotropic.append(
            predicted_pos_spherical_isotropic.item()
        )
        predicted_angle_ana.append(predicted_pos_ana.item())

        true_angle_np = np.stack(true_angle)
        predicted_angle_spherical_np = np.stack(predicted_angle_spherical)
        predicted_angle_spherical_isotropic_np = np.stack(
            predicted_angle_spherical_isotropic
        )
        predicted_angle_ana_np = np.stack(predicted_angle_ana)

        true_angle_deg = np.rad2deg(true_angle_np)
        angle_err = np.abs(np.rad2deg(true_angle_np - predicted_angle_spherical_np))
        angle_err_ana = np.abs(np.rad2deg(true_angle_np - predicted_angle_ana_np))
        angle_err_isotropic = np.abs(
            np.rad2deg(true_angle_np - predicted_angle_spherical_isotropic_np)
        )

    mean_err_ana = np.mean(angle_err_ana, axis=0)
    mean_err_nn = np.mean(angle_err, axis=0)
    mean_err_isotropic = np.mean(angle_err_isotropic, axis=0)

    print(f"ana error = {mean_err_ana:.6f}")
    print(f"aniso error = {mean_err_nn:.6f}")
    print(f"iso error = {mean_err_isotropic:.6f}")

    print(f"Avg time for measurement NN: {np.mean(times)}")

    ax.plot(
        true_angle_deg,
        angle_err,
        color=colors_[1],
        marker=markers[1],
        label="NN Correction (Anisotropic)",
    )

    ax.plot(
        true_angle_deg,
        angle_err_isotropic,
        color=colors_[0],
        marker=markers[0],
        label="NN Correction (Isotropic)",
    )

    ax.plot(
        true_angle_deg,
        angle_err_ana,
        color=colors_[2],
        marker=markers[2],
        label="Analytical Model",
    )


def main():
    susceptibilities = np.array([0.3, 0.3, 0.05])

    fig, ax = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    d = 1
    dimensions = (3, 3, 1)
    test_susc(
        susceptibilities=susceptibilities,
        dimensions=dimensions,
        d=d,
        ax=ax[0],
    )
    ax[0].set_title(f"dimensions={tuple(d * 5 for d in dimensions)}")

    d = 1
    dimensions = (1, 1, 1)
    test_susc(
        susceptibilities=susceptibilities,
        dimensions=dimensions,
        d=d,
        ax=ax[1],
    )
    ax[1].set_title(f"dimensions={tuple(d * 5 for d in dimensions)}")

    d = 1
    dimensions = (0.3, 0.3, 1)
    test_susc(
        susceptibilities=susceptibilities,
        dimensions=dimensions,
        d=d,
        ax=ax[2],
    )
    ax[2].set_title(f"dimensions={tuple(d * 5 for d in dimensions)}")

    for axs in ax:
        axs.set_xlabel(r"$\theta_{true}$ (°)")
        axs.grid(alpha=0.5)

    ax[2].legend(frameon=False)
    ax[0].set_ylabel(r"$|\theta_{true} - \theta_{predicted}|$ (°)")

    plt.tight_layout()
    plt.savefig(
        f"results/paper_v2/joystick/angle_err_with_isotropic_vary_dims_v2.pdf",
        format="pdf",
    )
    plt.show()


if __name__ == "__main__":
    main()

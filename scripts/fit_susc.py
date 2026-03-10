import torch

from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import SphericalCorrectionNetwork
from nn_magnetics.optimize.fit_lbfgs import build_model
from tqdm import tqdm

from nn_magnetics.optimize.other import optimize, optimize_isotropic

COMPONENT_WEIGHTS_PATH = "/Users/jacksmith/Documents/work/nn-magnetics/results/paper_v2/component/best_weights.pt"
SPHERICAL_WEIGHTS_PATH = "/Users/jacksmith/Documents/work/nn-magnetics/results/paper_v2/spherical/best_weights.pt"
DATA_PATH = (
    "/Users/jacksmith/Documents/work/nn-magnetics/data/3dof_chi_v3/small/validation"
)


def fit_isotropic():
    X, B = AnisotropicData(DATA_PATH).get_magnets()

    model_spherical = build_model(
        SphericalCorrectionNetwork,
        SPHERICAL_WEIGHTS_PATH,
        torch.nn.functional.gelu,
    )

    true_suscs = []
    predicted_suscs_spherical = []

    for x, b in zip(X, B):
        observers = x[:, 5:]
        B_measured = b[:, :3]

        true_susc = x[0, 2:5].clone().requires_grad_(True)
        dimensions = x[0, :2]
        true_suscs.append(true_susc)

        predicted_susc_spherical, loss = optimize_isotropic(
            model=model_spherical,
            observers=observers,
            B_measured=B_measured,
            dimensions=dimensions,
            method="l-bfgs",
            options={"gtol": 1e-11},
            n_iter=1,
            x0=None,
        )

        predicted_suscs_spherical.append(torch.tile(predicted_susc_spherical, (3,)))

        print(list(float(f"{s:.4f}") for s in true_susc.tolist()))
        print(list(float(f"{s:.4f}") for s in predicted_susc_spherical.tolist()))
        print("\n\n")

    true_suscs = torch.stack(true_suscs)
    predicted_suscs_spherical = torch.stack(predicted_suscs_spherical)

    susc_err = (
        ((true_suscs - predicted_suscs_spherical) * 100 / true_suscs).mean(0).abs()
    )

    susc_error_abs = (
        ((true_suscs - predicted_suscs_spherical) / true_suscs).mean(0).abs()
    )
    print(susc_err)
    print(susc_error_abs)

    loss_spherical = torch.nn.functional.mse_loss(true_suscs, predicted_suscs_spherical)

    print(f"Spherical Correction Loss: {loss_spherical.item()}")


def main():
    X, B = AnisotropicData(DATA_PATH).get_magnets()

    model_spherical = build_model(
        SphericalCorrectionNetwork,
        SPHERICAL_WEIGHTS_PATH,
        torch.nn.functional.gelu,
    )

    true_suscs = []
    predicted_suscs_spherical = []

    for x, b in zip(X, B):
        observers = x[:, 5:]
        B_measured = b[:, :3]

        true_susc = x[0, 2:5].clone().requires_grad_(True)
        dimensions = x[0, :2]
        true_suscs.append(true_susc)

        predicted_susc_spherical, loss = optimize(
            model=model_spherical,
            observers=observers,
            B_measured=B_measured,
            dimensions=dimensions,
            method="l-bfgs",
            options={"gtol": 1e-11},
            n_iter=1,
            x0=None,
        )

        predicted_suscs_spherical.append(predicted_susc_spherical)

        print(list(float(f"{s:.4f}") for s in true_susc.tolist()))
        print(list(float(f"{s:.4f}") for s in predicted_susc_spherical.tolist()))
        print("\n\n")

    true_suscs = torch.stack(true_suscs)
    predicted_suscs_spherical = torch.stack(predicted_suscs_spherical)

    susc_err = (
        ((true_suscs - predicted_suscs_spherical) * 100 / true_suscs).mean(0).abs()
    )

    susc_error_abs = (
        ((true_suscs - predicted_suscs_spherical) / true_suscs).mean(0).abs()
    )
    print(susc_err)
    print(susc_error_abs)

    loss_spherical = torch.nn.functional.mse_loss(true_suscs, predicted_suscs_spherical)

    print(f"Spherical Correction Loss: {loss_spherical.item()}")


if __name__ == "__main__":
    fit_isotropic()

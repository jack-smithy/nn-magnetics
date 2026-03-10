from typing import Type

import numpy as np
import torch
from magpylib import magnet
from magpylib_material_response import demag, meshing
from scipy.optimize import differential_evolution
from torch import Tensor
from torch.nn import functional as F
from torchmin import minimize

from nn_magnetics.models.base import BaseNetwork
from nn_magnetics.utils.physics import batch_rotation_matrices


def joystick_points_and_field(r, d, n_points, susceptibility, dimension, eps=1e-3):
    thetas = np.linspace(eps, np.pi / 12, n_points)

    pos = np.array([[r * np.sin(t), 0, -1 * r * np.cos(t)] for t in thetas])
    ang = np.array([[0, t, 0] for t in thetas])
    observers = np.array([[0, 0, -r - d] for _ in range(n_points)])

    cuboid = magnet.Cuboid(dimension=dimension, polarization=(0, 0, 1))
    mesh = meshing.mesh_Cuboid(cuboid=cuboid, target_elems=100)
    demag.apply_demag(mesh, susceptibility=susceptibility, inplace=True)

    B = B_full_domain_MoM(
        observers=observers,
        angles=ang,
        positions=pos,
        mesh=mesh,
    )

    return thetas, pos, ang, observers, B


def joystick_v2(
    theta: Tensor,
    r: float,
    d: float,
    model: BaseNetwork,
    dimensions: Tensor,
    susceptibilities: Tensor,
):
    position = torch.stack(
        [
            r * torch.sin(theta),
            torch.zeros_like(theta),
            -1 * r * torch.cos(theta),
        ],
        dim=1,
    ).squeeze()

    angle = torch.stack(
        [
            torch.zeros_like(theta),
            theta,
            torch.zeros_like(theta),
        ],
        dim=1,
    ).squeeze()

    observer = torch.stack(
        [
            torch.zeros_like(theta),
            torch.zeros_like(theta),
            -1 * (r + d) * torch.ones_like(theta),
        ],
        dim=1,
    ).squeeze()

    B = B_full_domain(
        observers=observer,
        dimensions=dimensions,
        susceptibilities=susceptibilities,
        model=model,
        positions=position,
        angles=angle,
    )

    return B


def joystick(
    theta: Tensor,
    r: float,
    d: float,
    model: BaseNetwork,
    dimensions: Tensor,
    susceptibilities: Tensor,
):
    print(torch.rad2deg(theta))
    position = torch.tensor([r * torch.sin(theta), 0, -1 * r * torch.cos(theta)])
    angle = torch.tensor([0, theta, 0])
    observer = torch.tensor([0, 0, -r - d])

    print(position)
    print(angle)
    print(observer)

    B = B_full_domain(
        observers=observer,
        dimensions=dimensions,
        susceptibilities=susceptibilities,
        model=model,
        positions=position,
        angles=angle,
    )

    return B


def joystick_ana(
    theta: Tensor,
    r: float,
    d: float,
    dimensions: Tensor,
):
    position = np.array([r * np.sin(theta[0]), 0, -1 * r * np.cos(theta[0])])

    angle = np.array([0, theta[0], 0])

    observer = np.array([0, 0, -r - d])

    cuboid = magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))

    B = B_full_domain_MoM(
        observers=observer,
        positions=position,
        angles=angle,
        mesh=cuboid,
    )

    return B


def B_full_domain_MoM(observers, positions, angles, mesh):
    if positions is not None:
        # Shift observers into the magnet's frame
        observers = observers - positions

    if angles is not None:
        # Rotation angles: angles = (alpha, beta, gamma) in radians
        Rs = batch_rotation_matrices(torch.from_numpy(np.deg2rad(angles))).numpy()

        if len(observers.shape) == 1:
            observers = np.array([[observers]])

        observers = np.matmul(observers[:, None, :], Rs.transpose(0, 2, 1)).squeeze(1)

    return mesh.getB(observers)


def B_full_domain(
    observers: Tensor,
    dimensions: Tensor,
    susceptibilities: Tensor,
    model: BaseNetwork,
    positions: Tensor | None = None,
    angles: Tensor | None = None,
) -> Tensor:
    if positions is not None:
        # Shift observers into the magnet's frame
        observers = observers - positions

    if angles is not None:
        # Rotation angles: angles = (alpha, beta, gamma) in radians
        Rs = batch_rotation_matrices(torch.deg2rad(angles))
        R_T = Rs.transpose(1, 2).to(torch.float64)
        observers = torch.bmm(observers.unsqueeze(0).unsqueeze(0), R_T).squeeze(1)

    signs = torch.sign(observers)
    mask = signs[:, 2] == -1
    signs[mask] *= -1

    abs_observers = observers.abs()
    n_samples = observers.shape[0]

    dims = dimensions.expand((n_samples, -1))
    susc = susceptibilities.expand((n_samples, -1))

    feature = torch.cat([dims, susc, abs_observers], dim=1)
    return signs * model(feature)


def build_model(model_cls: Type[BaseNetwork], path: str, activation) -> BaseNetwork:
    """
    Load a trained model from saved weights. We can't do this beforehand as the model
    can't be shared between threads

    Args:
        model_cls (Type[BaseNetwork]): Model class
        path (str): Path to saved weights

    Returns:
        BaseNetwork: Model with loaded weights
    """
    model = model_cls.load_from_path(
        path,
        activation=activation,
        save_path=None,
        save_weights=False,
        do_output_activation=False,
        p=0,
    ).to(torch.float64)

    torch.set_grad_enabled(True)
    model.eval()

    return model


def optimize_positions_joystick(
    model: BaseNetwork,
    B_measured: Tensor,
    susceptibility: Tensor,
    dimensions: Tensor,
    r: float,
    d: float,
    maxiter=None,
    options=None,
) -> tuple[Tensor, float]:
    B_measured = B_measured.unsqueeze(0)

    def objective(theta: Tensor):
        B_pred = joystick_v2(
            theta=theta,
            r=r,
            d=d,
            model=model,
            dimensions=dimensions,
            susceptibilities=susceptibility,
        )

        loss = F.l1_loss(B_measured, B_pred)

        return loss

    x0 = torch.tensor([0.1], dtype=torch.float64)
    result = minimize(
        objective,
        x0,
        "l-bfgs",
        max_iter=maxiter,
        options=options,
    )

    return result.x, result.fun


def optimize_positions_joystick_ana(
    B_measured: Tensor,
    dimensions: Tensor,
    r: float,
    d: float,
) -> tuple[Tensor, float]:
    def objective(theta: Tensor):
        # assert observers.requires_grad
        B_pred = joystick_ana(
            theta=theta,
            r=r,
            d=d,
            dimensions=dimensions,
        )

        B_pred = torch.from_numpy(B_pred)
        # print(f"B pred shape: {B_pred.shape}")
        # print(f"B measured shape: {B_measured.shape}")
        loss = F.l1_loss(B_measured, B_pred)

        return loss.item()

    bounds = [(0, np.pi)]
    result = differential_evolution(objective, bounds=bounds, maxiter=10)

    return result.x, result.fun

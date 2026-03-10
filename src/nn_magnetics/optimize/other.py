import numpy as np
from scipy.optimize import differential_evolution
from nn_magnetics.models.base import BaseNetwork
from torchmin import minimize
from nn_magnetics.optimize.fit_lbfgs import B_full_domain
from nn_magnetics.utils.physics import Bfield_homogeneous
from magpylib_material_response import meshing, demag
from magpylib import magnet

import torch
from torch import Tensor
from torch.nn import functional as F


def _calc_loss_analytical(
    B_measured: Tensor,
    observers: Tensor,
    dimension: Tensor,
) -> Tensor:
    """
    Calculate the loss for the predicted field vs the true field

    Args:
        model (BaseNetwork): Trained predictive model
        B_measured (Tensor): Ground truth B field
        observers (Tensor): Evaluation points
        susceptibilities (Tensor): Magnet susceptibility
        dimensions (Tensor): Magnet dimensions

    Returns:
        Tensor: Loss value
    """
    observers = torch.from_numpy(observers)
    B_measured = torch.from_numpy(B_measured)

    n_samples = observers.shape[0]

    cc = torch.ones((n_samples, 1))
    dimensions = dimension.expand((n_samples, -1))
    dimensions = torch.cat([dimensions, cc], dim=1)

    polarizations = torch.tensor([0, 0, 1]).expand((n_samples, -1))

    B_predicted = Bfield_homogeneous(
        observers=observers,
        dimensions=dimensions,
        polarizations=polarizations,
    )

    loss = F.l1_loss(B_measured, B_predicted)

    return loss


def _calc_loss(
    model: BaseNetwork,
    B_measured: Tensor,
    observers: Tensor,
    susceptibility: Tensor,
    dimension: Tensor,
) -> Tensor:
    """
    Calculate the loss for the predicted field vs the true field

    Args:
        model (BaseNetwork): Trained predictive model
        B_measured (Tensor): Ground truth B field
        observers (Tensor): Evaluation points
        susceptibilities (Tensor): Magnet susceptibility
        dimensions (Tensor): Magnet dimensions

    Returns:
        Tensor: Loss value
    """
    if isinstance(observers, np.ndarray):
        observers = torch.from_numpy(observers)

    if isinstance(B_measured, np.ndarray):
        B_measured = torch.from_numpy(B_measured)

    n_samples = observers.shape[0]

    susceptibilities = susceptibility.expand((n_samples, -1))
    dimensions = dimension.expand((n_samples, -1))

    B_predicted = B_full_domain(
        observers=observers,
        dimensions=dimensions,
        susceptibilities=susceptibilities,
        model=model,
    )

    loss = F.l1_loss(B_measured, B_predicted)

    return loss


def optimize_isotropic(
    model: BaseNetwork,
    observers: Tensor,
    B_measured: Tensor,
    dimensions: Tensor,
    method: str,
    options: dict | None,
    n_iter: int = 10,
    x0: Tensor | None = None,
) -> tuple[Tensor, float]:
    def objective(susc):
        loss = _calc_loss(
            model=model,
            B_measured=B_measured,
            observers=observers,
            susceptibility=torch.tile(susc, (3,)),
            dimension=dimensions,
        )

        return loss

    best_loss = torch.inf
    best_params = None

    for _ in range(n_iter):
        if x0 is None:
            x0 = torch.rand(1, dtype=torch.float64, requires_grad=True)

        result = minimize(
            objective,
            x0=x0,
            method=method,
            options=options,
        )

        params = result.x
        loss = result.fun

        if loss < best_loss:
            best_params = params
            best_loss = loss

    if best_params is None:
        raise RuntimeError("No runs converged")

    return best_params, best_loss


def optimize(
    model: BaseNetwork,
    observers: Tensor,
    B_measured: Tensor,
    dimensions: Tensor,
    method: str,
    options: dict | None,
    n_iter: int = 10,
    x0: Tensor | None = None,
) -> tuple[Tensor, float]:
    def objective(susc):
        loss = _calc_loss(
            model=model,
            B_measured=B_measured,
            observers=observers,
            susceptibility=susc,
            dimension=dimensions,
        )

        return loss

    best_loss = torch.inf
    best_params = None

    for _ in range(n_iter):
        if x0 is None:
            x0 = torch.rand(3, dtype=torch.float64, requires_grad=True)

        result = minimize(
            objective,
            x0=x0,
            method=method,
            options=options,
        )

        params = result.x
        loss = result.fun

        if loss < best_loss:
            best_params = params
            best_loss = loss

    if best_params is None:
        raise RuntimeError("No runs converged")

    return best_params, best_loss


def optimize_positions(
    model: BaseNetwork,
    B_measured: Tensor,
    susceptibility: Tensor,
    dimensions: Tensor,
    method: str,
    options: dict | None,
    n_iter: int = 10,
    p0: Tensor | None = None,
) -> tuple[Tensor, float]:
    def objective(observers: Tensor):
        # assert observers.requires_grad

        loss = _calc_loss(
            model=model,
            B_measured=np.expand_dims(B_measured, axis=0),  # type: ignore
            observers=np.expand_dims(observers, axis=0),  # type: ignore
            susceptibility=susceptibility,
            dimension=dimensions,
        )

        return loss.item()

    best_loss = torch.inf
    best_params = None

    for _ in range(n_iter):
        # if p0 is None:
        a, b = dimensions[0], dimensions[1]
        xmin, xmax = 0, 2.5 * a
        ymin, ymax = 0, 2.5 * b
        zmin, zmax = 0, 2.5
        # x0 = xmin + torch.rand(1) * (xmax - xmin)
        # y0 = ymin + torch.rand(1) * (ymax - ymin)
        # z0 = zmin + torch.rand(1) * (zmax - zmin)

        # p0 = torch.cat([x0, y0, z0])

        # result = minimize(
        #     objective,
        #     x0=p0,
        #     method=method,
        #     options=options,
        # )

        bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        result = differential_evolution(objective, bounds=bounds, maxiter=10)

        params = result.x
        loss = result.fun

        if loss < best_loss:
            best_params = params
            best_loss = loss

    if best_params is None:
        raise RuntimeError("No runs converged")

    return best_params, best_loss


def optimize_positions_ana(
    B_measured: Tensor,
    dimensions: Tensor,
    n_iter: int = 10,
    p0: Tensor | None = None,
) -> tuple[Tensor, float]:
    def objective(observers: Tensor):
        # assert observers.requires_grad

        loss = _calc_loss_analytical(
            B_measured=np.expand_dims(B_measured, axis=0),  # type: ignore
            observers=np.expand_dims(observers, axis=0),  # type: ignore
            dimension=dimensions,
        )

        return loss.item()

    best_loss = torch.inf
    best_params = None

    for _ in range(n_iter):
        # if p0 is None:
        a, b = dimensions[0], dimensions[1]
        xmin, xmax = 0, 2.5 * a
        ymin, ymax = 0, 2.5 * b
        zmin, zmax = 0, 2.5
        # x0 = xmin + torch.rand(1) * (xmax - xmin)
        # y0 = ymin + torch.rand(1) * (ymax - ymin)
        # z0 = zmin + torch.rand(1) * (zmax - zmin)

        # p0 = torch.cat([x0, y0, z0])

        # result = minimize(
        #     objective,
        #     x0=p0,
        #     method=method,
        #     options=options,
        # )

        bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        result = differential_evolution(objective, bounds=bounds, maxiter=10)

        params = result.x
        loss = result.fun

        if loss < best_loss:
            best_params = params
            best_loss = loss

    if best_params is None:
        raise RuntimeError("No runs converged")

    return best_params, best_loss


def format_result(x, message, fun):
    res_str = f"""
    Message: {message}
    Loss: {fun}
    Dimensions=({x[0]:.4f}, {x[1]:.4f}, 1.0)
    Susceptibility=({x[2]:.4f}, {x[3]:.4f}, {x[4]:.4f})
    Polarization=(0, 0, 1)
    """
    print(res_str)


def format_position(x, message, fun):
    res_str = f"""
    Message: {message}
    Loss: {fun}
    Position=({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})
    """
    print(res_str)


def format_position_errors(observers_true, observers_pred):
    abs_err = (observers_true - observers_pred).abs()
    rel_err = (observers_true - observers_pred).abs() / observers_true * 100

    res_str = f"""
    Absolute Position Error=({abs_err[0]:.4f}mm, {abs_err[1]:.4f}mm, {abs_err[2]:.4f}mm)
    Relative Position Error=({rel_err[0]:.4f}%, {rel_err[1]:.4f}%, {rel_err[2]:.4f}%)
    """
    print(res_str)


def format_true(feature, fun):
    a, b = feature[0], feature[1]
    chi = feature[2:5]

    res_str = f"""
    True Values:
    Loss: {fun}
    Dimensions=({a:.4f}, {b:.4f}, 1.0)
    Susceptibility=({chi[0]:.4f}, {chi[1]:.4f}, {chi[2]:.4f})
    Polarization=(0, 0, 1)
    """
    print(res_str)


def format_errors(true_params, predicted_params):
    dims_true, dims_predicted = true_params[:2], predicted_params[:2]
    susc_true, susc_predicted = true_params[2:5], predicted_params[2:5]

    dim_error = ((dims_true - dims_predicted).abs() / dims_true * 100).mean()
    susc_error = ((susc_true - susc_predicted).abs() / susc_true * 100).mean()

    res_str = f"""
    Dimension Error={dim_error:.4f}%
    Susceptibility Error={susc_error:.4f}%
    """
    print(res_str)

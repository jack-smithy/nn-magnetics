from typing import Type

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torchmin import minimize
from nn_magnetics.utils.physics import Bfield_homogeneous
from nn_magnetics.models.base import BaseNetwork
from scipy.optimize import differential_evolution, basinhopping
from nn_magnetics.utils.physics import batch_rotation_matrices
from magpylib.core import magnet_cuboid_Bfield
from nn_magnetics.utils.metrics import angle_error, relative_amplitude_error
from magpylib import magnet
from magpylib_material_response import meshing, demag


def joystick_points_and_field(r, d, n_points, susceptibility, eps=1e-3):
    thetas = np.linspace(eps, np.pi / 12, n_points)

    pos = np.array([[r * np.sin(t), 0, -1 * r * np.cos(t)] for t in thetas])
    ang = np.array([[0, t, 0] for t in thetas])
    observers = np.array([[0, 0, -r - d] for _ in range(n_points)])

    cuboid = magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    mesh = meshing.mesh_Cuboid(cuboid=cuboid, target_elems=100)
    demag.apply_demag(mesh, susceptibility=susceptibility, inplace=True)

    B = B_full_domain_MoM(
        observers=observers,
        angles=ang,
        positions=pos,
        mesh=mesh,
    )

    return thetas, pos, ang, observers, B


def joystick(
    theta: Tensor,
    r: float,
    d: float,
    model: BaseNetwork,
    dimensions: Tensor,
    susceptibilities: Tensor,
):
    theta = torch.tensor(theta)
    position = torch.tensor([r * torch.sin(theta), 0, -1 * r * torch.cos(theta)])
    angle = torch.tensor([0, theta, 0])
    observer = torch.tensor([0, 0, -r - d])

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


def optimize_positions_joystick(
    model: BaseNetwork,
    B_measured: Tensor,
    susceptibility: Tensor,
    dimensions: Tensor,
    r: float,
    d: float,
) -> tuple[Tensor, float]:
    B_measured = B_measured.unsqueeze(0)

    def objective(theta: Tensor):
        # assert observers.requires_grad
        B_pred = joystick(
            theta=theta,
            r=r,
            d=d,
            model=model,
            dimensions=dimensions,
            susceptibilities=susceptibility,
        )

        loss = F.l1_loss(B_measured, B_pred)

        return loss.item()

    bounds = [(0, np.pi / 12)]
    result = differential_evolution(objective, bounds=bounds, maxiter=10)

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


if __name__ == "__main__":
    from magpylib import magnet
    from magpylib_material_response import meshing, demag
    from nn_magnetics.models import SphericalCorrectionNetwork
    from scipy.spatial.transform import Rotation as R

    SPHERICAL_WEIGHTS_PATH = "/Users/jacksmith/Documents/work/nn-magnetics/results/paper_v2/spherical/best_weights.pt"

    model_spherical = build_model(
        SphericalCorrectionNetwork,
        SPHERICAL_WEIGHTS_PATH,
        torch.nn.functional.gelu,
    )

    dimensions = (1, 1, 1)
    susceptibilities = (0.1, 0.2, 0.3)
    position = [[0, y * -0.1, 0] for y in range(16)]
    angle = [[0, a, 0] for a in range(16)]

    cuboid = magnet.Cuboid(
        dimension=dimensions,
        polarization=(0, 0, 1),
    )

    res = joystick_points_and_field(
        4,
        1,
        16,
        susceptibility=susceptibilities,
    )

    res = [torch.from_numpy(r) for r in res]
    thetas, pos, ang, obs, B = res
    dim = torch.tensor(dimensions[:2])
    susc = torch.tensor(susceptibilities)
    # print(pos)
    # print(ang)
    # print(obs)
    # print(thetas)
    print(B)

    B_pred = B_full_domain(
        observers=obs,
        dimensions=dim,
        susceptibilities=susc,
        model=model_spherical,
        positions=pos,
        angles=ang,
    )

    print(B_pred)

    print(angle_error(B, B_pred).mean().item())

    # grid = np.array([[1.0, 1.0, 1.0] for _ in range(16)])

    # mesh = meshing.mesh_Cuboid(cuboid, 100)
    # demag.apply_demag(mesh, susceptibility=susceptibilities, inplace=True)

    # B_true = torch.from_numpy(
    #     B_full_domain_MoM(
    #         observers=grid,
    #         positions=position,
    #         angles=angle,
    #         mesh=mesh,
    #     )
    # )

    # observers = torch.from_numpy(grid)
    # dims = torch.tensor(dimensions[:2])
    # susc = torch.tensor(susceptibilities)
    # pos = torch.tensor(position)
    # ang = torch.tensor(angle)

    # B_pred = B_full_domain(
    #     observers=observers,
    #     dimensions=dims,
    #     susceptibilities=susc,
    #     positions=pos,
    #     model=model_spherical,
    #     angles=ang,
    # )

    # angle_err = angle_error(B_true, B_pred)
    # amp_err = relative_amplitude_error(B_true, B_pred, return_abs=True)
    # print(
    #     f"\nAngle err: {angle_err.mean().item():.4f}, Amp err: {amp_err.mean().item():.4f}"
    # )

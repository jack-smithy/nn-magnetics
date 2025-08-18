from nn_magnetics.models import SphericalCorrectionNetwork
import torch
from torch import Tensor
import matplotlib.pyplot as plt


def rotate(arr: Tensor, start: int, end: int):
    if start == end:
        return arr

    _arr = arr.clone()

    _arr[:, end] = arr[:, start]
    _arr[:, start] = arr[:, end]

    return _arr


def _B(
    observers: Tensor,
    dimensions: Tensor,
    susceptibilities: Tensor,
    model: SphericalCorrectionNetwork,
) -> Tensor:
    signs = torch.sign(observers)
    mask = signs[:, 2] == -1
    signs[mask] *= -1

    abs_observers = torch.abs(observers)
    a, b, c = dimensions.T

    _a = (a / c).unsqueeze(1)
    _b = (b / c).unsqueeze(1)
    _c = c.unsqueeze(1)

    _dimensions = torch.cat([_a, _b], dim=1)
    _observers = abs_observers / _c

    feature = torch.cat([_dimensions, susceptibilities, _observers], dim=1)
    return signs * model(feature)


def _B_rotated(
    observers: Tensor,
    dimensions: Tensor,
    susceptibilities: Tensor,
    model: SphericalCorrectionNetwork,
    axis: int,
):
    observers_rotated = rotate(observers, 2, axis)
    dimensions_rotated = rotate(dimensions, 2, axis)
    susceptibilities_rotated = rotate(susceptibilities, 2, axis)

    B_rot = _B(
        observers=observers_rotated,
        dimensions=dimensions_rotated,
        susceptibilities=susceptibilities_rotated,
        model=model,
    )

    return rotate(B_rot, axis, 2)


def Bfield_nn(
    observers: Tensor,
    dimensions: Tensor,
    polarizations: Tensor,
    susceptibilities: Tensor,
    model: SphericalCorrectionNetwork,
) -> Tensor:
    B = torch.zeros_like(observers)
    for axis in range(3):
        Ji = polarizations[:, axis].unsqueeze(1)
        Bi = _B_rotated(
            observers=observers,
            dimensions=dimensions,
            susceptibilities=susceptibilities,
            model=model,
            axis=axis,
        )
        B += Ji * Bi

    return B


if __name__ == "__main__":
    from magpylib import magnet
    from magpylib_material_response import meshing, demag
    from nn_magnetics.utils.metrics import angle_error, relative_amplitude_error
    from nn_magnetics.data.create_data import generate_points_grid

    dimension = (3, 6, 2)
    susceptibility = (0.2, 0.2, 0.2)
    polarization = (0, 0, 1.1)

    PATH = "/Users/jacksmith/Documents/work/nn-magnetics/results/3dof_chi_spherical/2025-05-13 16:36:35.129285/best_weights.pt"
    model = SphericalCorrectionNetwork.load_from_path(
        PATH,
        activation=torch.nn.functional.silu,
        save_path=None,
        save_weights=False,
        do_output_activation=False,
    ).to(torch.float64)
    model.eval()

    observers = generate_points_grid(26, *dimension)
    # observers[:, 1] *= -1
    observers[:, 2] *= -1
    # observers[:, 0] *= -1
    n_points = observers.shape[0]

    cuboid = magnet.Cuboid(dimension=dimension, polarization=polarization)
    mesh = meshing.mesh_Cuboid(cuboid=cuboid, target_elems=100)
    demag.apply_demag(mesh, susceptibility=susceptibility, inplace=True)
    B_true = torch.from_numpy(mesh.getB(observers))

    observers_t = torch.from_numpy(observers)
    polarization_t = torch.tensor(polarization).expand((n_points, -1))
    susceptibility_t = torch.tensor(susceptibility).expand((n_points, -1))
    dimension_t = torch.tensor(dimension).expand((n_points, -1))

    B_pred = Bfield_nn(
        observers=observers_t,
        dimensions=dimension_t,
        polarizations=polarization_t,
        susceptibilities=susceptibility_t,
        model=model,
    )

    angle_err = angle_error(B_true, B_pred)
    amp_err = relative_amplitude_error(B_true, B_pred, return_abs=True)

    avg_angle_err = angle_err.mean().item()
    avg_amp_err = amp_err.mean().item()

    print(f"Angle Error: {avg_angle_err:.4f}, Amp Error: {avg_amp_err:.4f}")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.hist(angle_err.numpy(force=True))
    ax2.hist(amp_err.numpy(force=True))

    plt.show()

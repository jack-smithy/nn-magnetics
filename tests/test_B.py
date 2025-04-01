import magpylib
import torch
import numpy as np
from nn_magnetics.utils.physics import Bfield_homogeneous
from nn_magnetics.data.create_data import generate_points_grid

torch.set_printoptions(precision=8)


def get_magnet_characterization():
    a, b, c, Jx, Jy, Jz = 1, 1, 1, 0, 0, 1
    return a, b, c, Jx, Jy, Jz


def get_magnet_characterization_many(n_samples=100):
    torch.manual_seed(199)
    dimensions = 0.3 + 2.7 * torch.rand((n_samples, 3))
    polarizations = torch.rand((n_samples, 3))
    observers = 3.0 + 1.5 * torch.rand((n_samples, 3))

    return dimensions, polarizations, observers


def get_grid():
    a, b, _, _, _, _ = get_magnet_characterization()
    return generate_points_grid(26, a, b)


def test_B_one_magnet():
    a, b, c, Jx, Jy, Jz = get_magnet_characterization()
    cuboid = magpylib.magnet.Cuboid(dimension=(a, b, c), polarization=(Jx, Jy, Jz))

    grid = get_grid()
    n_samples = grid.shape[0]

    B_magpylib = torch.from_numpy(cuboid.getB(grid))

    B_calculation = Bfield_homogeneous(
        observers=torch.from_numpy(grid),
        dimensions=torch.tensor([[a, b, c] for _ in range(n_samples)]),
        polarizations=torch.tensor([[Jx, Jy, Jz] for _ in range(n_samples)]),
    )

    assert torch.allclose(B_magpylib, B_calculation)


def test_B_multiple_magnets():
    dimensions, polarizations, observers = get_magnet_characterization_many()

    B_calculation = Bfield_homogeneous(
        observers=observers,
        polarizations=polarizations,
        dimensions=dimensions,
    )

    B_magpy = []
    for d, p, o in zip(dimensions, polarizations, observers):
        cuboid = magpylib.magnet.Cuboid(dimension=d.numpy(), polarization=p.numpy())
        B_magpy.append(cuboid.getB(o.numpy()))
    B_magpy = torch.from_numpy(np.array(B_magpy)).to(torch.float)

    assert torch.allclose(B_calculation, B_magpy, atol=1e-7)

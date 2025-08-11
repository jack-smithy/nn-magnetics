import magpylib
from magpylib_material_response import meshing, demag
import torch
from torch import Tensor
from nn_magnetics.data.create_data import generate_points_grid, generate_points_random


def get_mock_measurements(
    a: float,
    b: float,
    susceptibility: tuple,
    Pz: float,
    axis_coarseness: int,
) -> tuple[Tensor, Tensor]:
    X = generate_points_grid(axis_coarseness=axis_coarseness, a=a, b=b)
    cuboid = magpylib.magnet.Cuboid(dimension=(a, b, 1), polarization=(0, 0, Pz))
    mesh = meshing.mesh_Cuboid(cuboid=cuboid, target_elems=100)
    demag.apply_demag(mesh, susceptibility=susceptibility, inplace=True)

    B = mesh.getB(X)
    return torch.from_numpy(X).to(torch.float64), torch.from_numpy(B).to(torch.float64)

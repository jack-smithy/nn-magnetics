import magpylib
import numpy as np
import torch
from magpylib_material_response import demag, meshing

from nn_magnetics.utils.physics import Dz_cuboid, Nz_elementwise


def get_dimensions_and_N():
    return (
        [1.0, 1.5, 1.0, 3.1],
        [1.0, 1.0, 2.5, 2.3],
        [0.333333333333, 0.3754077108449708, 0.4177737334929503, 0.5607320394291083],
    )


def get_susceptibilities():
    return (
        [0, 0, 0],
        [0.1, 0.1, 0.1],
        [0.9, 0.87, 0.6789],
    )


def test_Nz_tensor():
    aa, bb, N_true = get_dimensions_and_N()
    N = Dz_cuboid(torch.tensor(aa), torch.tensor(bb))
    assert torch.allclose(N, torch.tensor(N_true))


def test_Nz_numpy():
    aa, bb, N_true = get_dimensions_and_N()
    for a, b, Ni in zip(aa, bb, N_true):
        N = Nz_elementwise(a, b, 1)
        assert np.allclose(N, Ni)


def test_Nz_magnetization():
    aa, bb, N_true = get_dimensions_and_N()
    chis = get_susceptibilities()

    for a, b, N, chi in zip(aa, bb, N_true, chis):
        print(a, b, N, chi)
        cuboid = magpylib.magnet.Cuboid(dimension=(a, b, 1), polarization=(0, 0, 1))

        mesh = meshing.mesh_Cuboid(cuboid=cuboid, target_elems=100)
        demag.apply_demag(mesh, susceptibility=chi, inplace=True)

        M = mesh.getM([m.position for m in mesh])
        J = magpylib.mu_0 * np.mean(M, axis=0)

        Jz = 1 / (1 + N * chi[2])

        assert np.allclose(J[2], Jz, rtol=1e-4, atol=1e-2)

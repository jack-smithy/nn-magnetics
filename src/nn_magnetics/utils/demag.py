import numpy as np
from collections.abc import Callable
from numpy import ndarray as Array


NewellFunc = Callable[[Array, Array, Array], Array]


def f(x: Array, y: Array, z: Array) -> Array:
    x, y, z = np.abs(x), np.abs(y), np.abs(z)
    x2, y2, z2 = x**2, y**2, z**2
    r = np.sqrt(x2 + y2 + z2)
    res = 1.0 / 6.0 * (2 * x2 - y2 - z2) * r

    res += np.nan_to_num(
        y / 2.0 * (z2 - x2) * np.asinh(y / np.sqrt(x2 + z2)),
        posinf=0,
        neginf=0,
    )

    res += np.nan_to_num(
        z / 2.0 * (y2 - x2) * np.asinh(z / np.sqrt(x2 + y2)),
        posinf=0,
        neginf=0,
    )

    res -= np.nan_to_num(
        x * y * z * np.atan(y * z / (x * r)),
        posinf=0,
        neginf=0,
    )

    return res


def g(x: Array, y: Array, z: Array) -> Array:
    z = np.abs(z)
    x2, y2, z2 = x**2, y**2, z**2
    r = np.sqrt(x2 + y2 + z2)

    res = -x * y * r / 3.0

    res += np.nan_to_num(
        x * y * z * np.asinh(z / np.sqrt(x2 + y2)),
        posinf=0,
        neginf=0,
    )
    res += np.nan_to_num(
        y / 6.0 * (3.0 * z2 - y2) * np.asinh(x / np.sqrt(y2 + z2)),
        posinf=0,
        neginf=0,
    )

    res += np.nan_to_num(
        x / 6.0 * (3.0 * z2 - x2) * np.asinh(y / np.sqrt(x2 + z2)),
        posinf=0,
        neginf=0,
    )

    res -= np.nan_to_num(
        z**3 / 6.0 * np.atan(x * y / (z * r)),
        posinf=0,
        neginf=0,
    )

    res -= np.nan_to_num(
        z * y2 / 2.0 * np.atan(x * z / (y * r)),
        posinf=0,
        neginf=0,
    )

    res -= np.nan_to_num(
        z * x2 / 2.0 * np.atan(y * z / (x * r)),
        posinf=0,
        neginf=0,
    )

    return res


def F1(func: NewellFunc, x: Array, y: Array, z: Array, dz: float) -> Array:
    return func(x, y, z + dz) - 2 * func(x, y, z) + func(x, y, z - dz)


def F0(func: NewellFunc, x: Array, y: Array, z: Array, dy: float, dz: float) -> Array:
    return (
        F1(func, x, y + dy, z, dz)
        - 2 * F1(func, x, y, z, dz)
        + F1(func, x, y - dy, z, dz)
    )


def newell(
    func: NewellFunc, x: Array, y: Array, z: Array, dx: float, dy: float, dz: float
) -> Array:
    res = (
        2 * F0(func, x, y, z, dy, dz)
        - F0(func, x - dx, y, z, dy, dz)
        - F0(func, x + dx, y, z, dy, dz)
    )
    return -res / (4.0 * np.pi * dx * dy * dz)


def dipole_f(x: Array, y: Array, z: Array, dx: float, dy: float, dz: float) -> Array:
    res = (2.0 * x**2 - y**2 - z**2) * pow(x**2 + y**2 + z**2, -5.0 / 2.0)
    res[0, 0, 0] = 0.0
    return res * dx * dy * dz / (4.0 * np.pi)


def dipole_g(x: Array, y: Array, z: Array, dx: float, dy: float, dz: float) -> Array:
    res = 3.0 * x * y * pow(x**2 + y**2 + z**2, -5.0 / 2.0)
    res[0, 0, 0] = 0.0
    return res * dx * dy * dz / (4.0 * np.pi)


def demag_f(x: Array, y: Array, z: Array, dx: float, dy: float, dz: float, p: int):
    with np.errstate(divide="ignore", invalid="ignore"):
        res = dipole_f(x, y, z, dx, dy, dz)
        near = (x**2 + y**2 + z**2) / (dx**2 + dy**2 + dz**2) < p**2
        res[near] = newell(f, x[near], y[near], z[near], dx, dy, dz)

    return res


def demag_g(
    x: Array, y: Array, z: Array, dx: float, dy: float, dz: float, p: int
) -> Array:
    with np.errstate(divide="ignore", invalid="ignore"):
        res = dipole_g(x, y, z, dx, dy, dz)
        near = (x**2 + y**2 + z**2) / (dx**2 + dy**2 + dz**2) < p**2
        res[near] = newell(g, x[near], y[near], z[near], dx, dy, dz)
    return res


def _shape(n: tuple) -> tuple:
    """Return padded FFT grid shape (2*N) for each axis."""
    if len(n) != 3:
        raise ValueError("_shape expects a 3-tuple")
    return tuple(2 * int(ni) for ni in n)


def _init_N_component(n: tuple, dx: Array, perm: tuple, func: Callable, p=20) -> Array:
    dx /= dx.min()

    shape = _shape(n=n)

    ij = [np.fft.fftfreq(n, 1 / n) for n in shape]  # local indices
    ij = np.meshgrid(*ij, indexing="ij")

    x, y, z = [ij[ind] * dx[ind] for ind in perm]
    dx_perm = [dx[ind] for ind in perm]

    Nc = func(x, y, z, *dx_perm, p)

    axes = [i for i in range(3) if n[i] > 1]

    if len(axes) > 0:
        Nc = np.fft.fftn(Nc, axes=axes)

    return Nc.real.copy()


def _init_N(n: tuple, dx: Array) -> list[list[Array]]:
    Nxx = _init_N_component(n, dx, (0, 1, 2), demag_f)
    Nxy = _init_N_component(n, dx, (0, 1, 2), demag_g)
    Nxz = _init_N_component(n, dx, (0, 2, 1), demag_g)
    Nyy = _init_N_component(n, dx, (1, 2, 0), demag_f)
    Nyz = _init_N_component(n, dx, (1, 2, 0), demag_g)
    Nzz = _init_N_component(n, dx, (2, 0, 1), demag_f)

    return [[Nxx, Nxy, Nxz], [Nxy, Nyy, Nyz], [Nxz, Nyz, Nzz]]


def h(n: tuple, dx: tuple, m: Array) -> Array:
    N = _init_N(n, np.array(dx))

    axes = [i for i in range(3) if n[i] > 1]
    shape = _shape(n)
    s = [shape[i] for i in axes]

    hx = np.zeros(N[0][0].shape, dtype=np.complex128)
    hy = np.zeros(N[0][0].shape, dtype=np.complex128)
    hz = np.zeros(N[0][0].shape, dtype=np.complex128)

    for ax in range(3):
        m_pad_fft1D = np.fft.fftn(m[:, :, :, (ax,)], axes=axes, s=s).squeeze(-1)

        hx += N[0][ax] * m_pad_fft1D
        hy += N[1][ax] * m_pad_fft1D
        hz += N[2][ax] * m_pad_fft1D

    hx = np.fft.ifftn(hx, axes=axes)
    hy = np.fft.ifftn(hy, axes=axes)
    hz = np.fft.ifftn(hz, axes=axes)

    return np.stack(
        [
            hx[: n[0], : n[1], : n[2]].real,
            hy[: n[0], : n[1], : n[2]].real,
            hz[: n[0], : n[1], : n[2]].real,
        ],
        axis=3,
    )


if __name__ == "__main__":
    from magnumnp import Mesh, State, DemagField
    import torch

    n = (500, 500, 10)
    dx = np.array((0.1, 0.1, 0.1))
    m = torch.tensor((1 / np.sqrt(2), 1 / np.sqrt(2), 0))
    Ms = torch.linalg.norm(m)

    mesh = Mesh(n, dx)
    coords = torch.stack(mesh.SpatialCoordinate(), dim=-1).numpy()
    state = State(mesh)

    state.material = {"Ms": Ms}
    demag = DemagField()
    state.m = state.Constant(np.array(m) / Ms.numpy())

    h_magnumnp = demag.h(state)

    m = (Ms * state.m).numpy()
    h_mine = h(n, tuple(dx), m)

    # np.savez("h_n1001003_d0101001_m1sqrt21sqrt20.npz", h=h_mine)

    np.testing.assert_allclose(h_magnumnp, h_mine, rtol=1e-9, atol=1e-9)


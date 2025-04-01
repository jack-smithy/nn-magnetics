import numpy as np
import torch
from torch import Tensor


def Dz_cuboid(dimensions: Tensor) -> Tensor:
    assert dimensions.shape[1] == 3
    assert torch.allclose(
        dimensions[..., 2], torch.ones(dimensions.shape[0], dtype=torch.double)
    )

    a, b = dimensions[..., 0], dimensions[..., 1]

    norm = torch.sqrt(a * a + b * b + 1)
    normab = torch.sqrt(a * a + b * b)
    normac = torch.sqrt(a * a + 1)
    normbc = torch.sqrt(b * b + 1)

    return (1 / np.pi) * (
        (b**2 - 1) / (2 * b) * torch.log((norm - a) / (norm + a))
        + (a**2 - 1) / (2 * a) * torch.log((norm - b) / (norm + b))
        + b / (2) * torch.log((normab + a) / (normab - a))
        + a / (2) * torch.log((normab + b) / (normab - b))
        + 1 / (2 * a) * torch.log((normbc - b) / (normbc + b))
        + 1 / (2 * b) * torch.log((normac - a) / (normac + a))
        + 2 * torch.arctan((a * b) / (norm))
        + (a**3 + b**3 - 2) / (3 * a * b)
        + (a**2 + b**2 - 2) / (3 * a * b) * norm
        + 1 / (a * b) * (torch.sqrt(a**2 + 1) + torch.sqrt(b**2 + 1))
        - ((a**2 + b**2) ** (3 / 2) + (b**2 + 1) ** (3 / 2) + (1 + a**2) ** (3 / 2))
        / (3 * a * b)
    )


def Dz_cuboid_elementwise(a: float, b: float, c: float) -> float:
    norm = np.sqrt(a * a + b * b + c * c)
    normab = np.sqrt(a * a + b * b)
    normac = np.sqrt(a * a + c * c)
    normbc = np.sqrt(b * b + c * c)

    return (1 / np.pi) * (
        (b**2 - c**2) / (2 * b * c) * np.log((norm - a) / (norm + a))
        + (a**2 - c**2) / (2 * a * c) * np.log((norm - b) / (norm + b))
        + b / (2 * c) * np.log((normab + a) / (normab - a))
        + a / (2 * c) * np.log((normab + b) / (normab - b))
        + c / (2 * a) * np.log((normbc - b) / (normbc + b))
        + c / (2 * b) * np.log((normac - a) / (normac + a))
        + 2 * np.arctan((a * b) / (c * norm))
        + (a**3 + b**3 - 2 * c**3) / (3 * a * b * c)
        + (a**2 + b**2 - 2 * c**2) / (3 * a * b * c) * norm
        + c / (a * b) * (np.sqrt(a**2 + c**2) + np.sqrt(b**2 + c**2))
        - (
            (a**2 + b**2) ** (3 / 2)
            + (b**2 + c**2) ** (3 / 2)
            + (c**2 + a**2) ** (3 / 2)
        )
        / (3 * a * b * c)
    )


def batch_rotation_matrices(angles: Tensor) -> Tensor:
    assert angles.shape[0] > 0 and angles.shape[1] == 3

    alpha = angles.T[0]
    beta = angles.T[1]
    gamma = angles.T[2]

    # Calculate cosines and sines of the angles
    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)
    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    # Construct rotation matrices for each axis in batch form
    Rx = torch.stack(
        [
            torch.ones_like(alpha),
            torch.zeros_like(alpha),
            torch.zeros_like(alpha),
            torch.zeros_like(alpha),
            cos_alpha,
            -sin_alpha,
            torch.zeros_like(alpha),
            sin_alpha,
            cos_alpha,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    Ry = torch.stack(
        [
            cos_beta,
            torch.zeros_like(beta),
            sin_beta,
            torch.zeros_like(beta),
            torch.ones_like(beta),
            torch.zeros_like(beta),
            -sin_beta,
            torch.zeros_like(beta),
            cos_beta,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    Rz = torch.stack(
        [
            cos_gamma,
            -sin_gamma,
            torch.zeros_like(gamma),
            sin_gamma,
            cos_gamma,
            torch.zeros_like(gamma),
            torch.zeros_like(gamma),
            torch.zeros_like(gamma),
            torch.ones_like(gamma),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    # # Combine rotations by matrix multiplication: Rz * Ry * Rx
    return Rz @ Ry @ Rx


def normalize_quaternion(q: Tensor):
    return q / q.norm(p=2, dim=1, keepdim=True)


def invert_quaternion(q: Tensor):
    _q = q.clone()
    _q[:, 1:] *= -1
    return _q / torch.sum(torch.pow(_q, 2), dim=1, keepdim=True)


def multiply_quaternions(q1: Tensor, q2: Tensor) -> Tensor:
    a1, b1, c1, d1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    a2, b2, c2, d2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    a3 = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b3 = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c3 = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d3 = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    return torch.cat(
        [
            a3.unsqueeze(0),
            b3.unsqueeze(0),
            c3.unsqueeze(0),
            d3.unsqueeze(0),
        ]
    ).T


def Bfield_homogeneous(
    observers: Tensor,
    dimensions: Tensor,
    polarizations: Tensor,
) -> Tensor:
    """B-field of homogeneously magnetized cuboids in Cartesian Coordinates.

    The cuboids sides are parallel to the coordinate axes. The geometric centers of the
    cuboids lie in the origin. The output is proportional to the polarization magnitude
    and independent of the length units chosen for observers and dimensions.

    Parameters
    ----------
    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates.

    dimensions: ndarray, shape (n,3)
        Length of cuboids sides.

    polarizations: ndarray, shape (n,3)
        Magnetic polarization vectors.

    Returns
    -------
    B-Field: ndarray, shape (n,3)
        B-field generated by Cuboids at observer positions.

    Notes
    -----
    Field computations via magnetic surface charge density. Published
    several times with similar expressions:

    Yang: Superconductor Science and Technology 3(12):591 (1990)

    Engel-Herbert: Journal of Applied Physics 97(7):074504 - 074504-4 (2005)

    Camacho: Revista Mexicana de Fisica E 59 (2013) 8-17

    Avoiding indeterminate forms:

    In the above implementations there are several indeterminate forms
    where the limit must be taken. These forms appear at positions
    that are extensions of the edges in all xyz-octants except bottQ4.
    In the vicinity of these indeterminate forms the formula becomes
    numerically instable.

    Chosen solution: use symmetries of the problem to change all
    positions to their bottQ4 counterparts. see also

    Cichon: IEEE Sensors Journal, vol. 19, no. 7, April 1, 2019, p.2509
    """
    pol_x, pol_y, pol_z = polarizations.T
    a, b, c = dimensions.T / 2
    x, y, z = observers.T.clone()

    # avoid indeterminate forms by evaluating in bottQ4 only --------
    # basic masks
    maskx = x < 0
    masky = y > 0
    maskz = z > 0

    # change all positions to their bottQ4 counterparts
    x = torch.where(maskx, -x, x)
    y = torch.where(masky, -y, y)
    z = torch.where(maskz, -z, z)

    # create sign flips for position changes
    qsigns = torch.ones((len(pol_x), 3, 3))
    qs_flipx = torch.tensor([[1, -1, -1], [-1, 1, 1], [-1, 1, 1]])
    qs_flipy = torch.tensor([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
    qs_flipz = torch.tensor([[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
    # signs flips can be applied subsequently

    qsigns[maskx] = qsigns[maskx] * qs_flipx
    qsigns[masky] = qsigns[masky] * qs_flipy
    qsigns[maskz] = qsigns[maskz] * qs_flipz

    # field computations --------------------------------------------
    # Note: in principle the computation for all three polarization-components can be
    #   vectorized itself using symmetries. However, tiling the three
    #   components will cost more than is gained by the vectorized evaluation

    # Note: making the following computation steps is not necessary
    #   as mkl will cache such small computations
    xma, xpa = x - a, x + a
    ymb, ypb = y - b, y + b
    zmc, zpc = z - c, z + c

    xma2, xpa2 = xma**2, xpa**2
    ymb2, ypb2 = ymb**2, ypb**2
    zmc2, zpc2 = zmc**2, zpc**2

    mmm = torch.sqrt(xma2 + ymb2 + zmc2)
    pmp = torch.sqrt(xpa2 + ymb2 + zpc2)
    pmm = torch.sqrt(xpa2 + ymb2 + zmc2)
    mmp = torch.sqrt(xma2 + ymb2 + zpc2)
    mpm = torch.sqrt(xma2 + ypb2 + zmc2)
    ppp = torch.sqrt(xpa2 + ypb2 + zpc2)
    ppm = torch.sqrt(xpa2 + ypb2 + zmc2)
    mpp = torch.sqrt(xma2 + ypb2 + zpc2)

    ff2x = torch.log((xma + mmm) * (xpa + ppm) * (xpa + pmp) * (xma + mpp)) - torch.log(
        (xpa + pmm) * (xma + mpm) * (xma + mmp) * (xpa + ppp)
    )

    ff2y = torch.log(
        (-ymb + mmm) * (-ypb + ppm) * (-ymb + pmp) * (-ypb + mpp)
    ) - torch.log((-ymb + pmm) * (-ypb + mpm) * (ymb - mmp) * (ypb - ppp))

    ff2z = torch.log(
        (-zmc + mmm) * (-zmc + ppm) * (-zpc + pmp) * (-zpc + mpp)
    ) - torch.log((-zmc + pmm) * (zmc - mpm) * (-zpc + mmp) * (zpc - ppp))

    ff2x = torch.nan_to_num(ff2x, nan=0.0, posinf=1e10, neginf=-1e10)
    ff2y = torch.nan_to_num(ff2y, nan=0.0, posinf=1e10, neginf=-1e10)
    ff2z = torch.nan_to_num(ff2z, nan=0.0, posinf=1e10, neginf=-1e10)

    ff1x = (
        torch.arctan2((ymb * zmc), (xma * mmm))
        - torch.arctan2((ymb * zmc), (xpa * pmm))
        - torch.arctan2((ypb * zmc), (xma * mpm))
        + torch.arctan2((ypb * zmc), (xpa * ppm))
        - torch.arctan2((ymb * zpc), (xma * mmp))
        + torch.arctan2((ymb * zpc), (xpa * pmp))
        + torch.arctan2((ypb * zpc), (xma * mpp))
        - torch.arctan2((ypb * zpc), (xpa * ppp))
    )

    ff1y = (
        torch.arctan2((xma * zmc), (ymb * mmm))
        - torch.arctan2((xpa * zmc), (ymb * pmm))
        - torch.arctan2((xma * zmc), (ypb * mpm))
        + torch.arctan2((xpa * zmc), (ypb * ppm))
        - torch.arctan2((xma * zpc), (ymb * mmp))
        + torch.arctan2((xpa * zpc), (ymb * pmp))
        + torch.arctan2((xma * zpc), (ypb * mpp))
        - torch.arctan2((xpa * zpc), (ypb * ppp))
    )

    ff1z = (
        torch.arctan2((xma * ymb), (zmc * mmm))
        - torch.arctan2((xpa * ymb), (zmc * pmm))
        - torch.arctan2((xma * ypb), (zmc * mpm))
        + torch.arctan2((xpa * ypb), (zmc * ppm))
        - torch.arctan2((xma * ymb), (zpc * mmp))
        + torch.arctan2((xpa * ymb), (zpc * pmp))
        + torch.arctan2((xma * ypb), (zpc * mpp))
        - torch.arctan2((xpa * ypb), (zpc * ppp))
    )

    # contributions from x-polarization
    #    the 'missing' third sign is hidden in ff1x
    bx_pol_x = pol_x * ff1x * qsigns[:, 0, 0]
    by_pol_x = pol_x * ff2z * qsigns[:, 0, 1]
    bz_pol_x = pol_x * ff2y * qsigns[:, 0, 2]
    # contributions from y-polarization
    bx_pol_y = pol_y * ff2z * qsigns[:, 1, 0]
    by_pol_y = pol_y * ff1y * qsigns[:, 1, 1]
    bz_pol_y = -pol_y * ff2x * qsigns[:, 1, 2]
    # contributions from z-polarization
    bx_pol_z = pol_z * ff2y * qsigns[:, 2, 0]
    by_pol_z = -pol_z * ff2x * qsigns[:, 2, 1]
    bz_pol_z = pol_z * ff1z * qsigns[:, 2, 2]

    # summing all contributions
    bx_tot = bx_pol_x + bx_pol_y + bx_pol_z
    by_tot = by_pol_x + by_pol_y + by_pol_z
    bz_tot = bz_pol_x + bz_pol_y + bz_pol_z

    B = torch.stack((bx_tot, by_tot, bz_tot), dim=0).T

    B /= 4 * torch.pi
    return B


def divB(B: Tensor, observers: Tensor):
    assert observers.requires_grad
    assert B.requires_grad

    n_samples = B.shape[0]

    div = torch.zeros(n_samples)

    for i in range(3):
        grad_Bi = torch.autograd.grad(
            B[:, i],
            observers,
            grad_outputs=torch.ones_like(B[:, i]),
            retain_graph=True,
        )[0]
        div += grad_Bi[:, i]

    return div

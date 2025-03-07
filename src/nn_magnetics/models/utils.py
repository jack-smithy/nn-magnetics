import torch
from torch import Tensor


def magnet_cuboid_Bfield_torch(
    observers: Tensor,  # shape (3,)
    dimensions: Tensor,  # shape (3,)
    polarizations: Tensor,  # shape (3,)
):
    observers = observers.unsqueeze(0)
    n_observers = observers.shape[0]

    dimensions = torch.tile(dimensions, (n_observers, 1))
    polarizations = torch.tile(polarizations, (n_observers, 1))

    pol_x, pol_y, pol_z = polarizations.T
    a, b, c = dimensions.T / 2

    x, y, z = torch.clone(observers).T

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

    qsigns = torch.where(maskx[:, None], qsigns * qs_flipx, qsigns)
    qsigns = torch.where(masky[:, None], qsigns * qs_flipy, qsigns)
    qsigns = torch.where(maskz[:, None], qsigns * qs_flipz, qsigns)

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

    # B = torch.c_[bx_tot, by_tot, bz_tot]      # faster for 10^5 and more evaluations
    B = torch.stack((bx_tot, by_tot, bz_tot), dim=0).T

    B /= 4 * torch.pi
    return B


def get_num_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return (
        sum(p.numel() for p in model.parameters())
        if trainable_only
        else sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

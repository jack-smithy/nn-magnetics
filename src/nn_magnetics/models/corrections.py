from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from scipy.spatial.transform import Rotation as R

ArrayLike = np.ndarray | Tensor


def batch_rotation_matrices(angles: Tensor) -> Tensor:
    assert angles.ndim == 2 and angles.shape[0] > 0 and angles.shape[1] == 3

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


def field_correction(
    B: Float[ArrayLike, "batch 6"],
    preds: Float[ArrayLike, "batch 3"],
) -> Tuple[Float[ArrayLike, "batch 3"], Float[ArrayLike, "batch 3"]]:
    """
    Correct analytical solution using NN predictions.

    Args:
        B (Float[np.ndarray, "batch 6"]): Array containing true B field (...,:3) and analytical B field (...,3:) for 'batch' points
        preds (Float[np.ndarray, "batch 3"]): NN correction factors for the 'batch points'

    Returns:
        (Tuple[Float[np.ndarray, "batch 3"], Float[np.ndarray, "batch 3"]]): Tuple containing the unmodified true B field and corrected analytical field.
    """
    assert preds.shape[1] == 3
    B_demag, B_reduced = B[..., :3], B[..., 3:]
    return B_demag, B_reduced * preds


def no_op(B: ArrayLike) -> Tuple[ArrayLike, ...]:
    return B[..., :3], B[..., 3:]


def amplitude_correction(B: ArrayLike, preds: ArrayLike) -> Tuple[ArrayLike, ...]:
    assert preds.shape[1] == 1
    B_demag, B_reduced = B[..., :3], B[..., 3:]
    return B_demag, preds * B_reduced


def angle_amp_correction(
    B_reduced: Tensor, angles: Tensor, amplitudes: Tensor
) -> Tensor:
    Rs = torch.from_numpy(R.from_rotvec(angles, degrees=False).as_matrix())
    return amplitudes[:, None] * torch.einsum("nij,nj->ni", Rs, B_reduced)


def quaternion_multiplication(q: Tensor, r: Tensor) -> Tensor:
    assert q.shape[1] == r.shape[1]
    # q and r are tensors of shape (..., 4)
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)

import numpy as np
import torch
from torch import Tensor


def demagnetizing_factor(a: float, b: float, c: float) -> float:
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


if __name__ == "__main__":
    q1 = 10 * torch.rand((10, 4), dtype=torch.float64)
    print(q1)
    q1_inv = invert_quaternion(q1)
    q3 = multiply_quaternions(q1, q1_inv)
    print(q3)

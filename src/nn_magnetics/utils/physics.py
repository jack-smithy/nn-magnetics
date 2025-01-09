import numpy as np
import torch

ArrayLike = torch.Tensor | np.ndarray


def demagnetizing_factor(a, b, c):
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


def batch_rotation_matrices(angles: ArrayLike) -> ArrayLike:
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


def split_tensor(B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return B[..., :3], B[..., 3:]

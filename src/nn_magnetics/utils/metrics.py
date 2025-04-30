from typing import Tuple, overload

import torch
import torch.linalg as TLA
from torch import Tensor
import numpy as np
from torch_geometric.data import Data


@overload
def relative_amplitude_error(
    v1: np.ndarray,
    v2: np.ndarray,
    return_abs: bool,
) -> np.ndarray: ...


@overload
def relative_amplitude_error(
    v1: Tensor,
    v2: Tensor,
    return_abs: bool,
) -> Tensor: ...


def relative_amplitude_error(
    v1: Tensor | np.ndarray,
    v2: Tensor | np.ndarray,
    return_abs: bool = True,
) -> Tensor | np.ndarray:
    """
    CalcuTLAtes the reTLAtive amplitude error between two vectors in %.

    Parameters:
    - v1 (array_like): First input vector.
    - v2 (array_like): Second input vector.

    Returns:
    - float: The reTLAtive amplitude error between v1 and v2 as a percentage.
    """
    is_tensor = isinstance(v1, Tensor) and isinstance(v2, Tensor)

    if not is_tensor:
        v1, v2 = torch.tensor(v1), torch.tensor(v2)

    v1_norm = TLA.norm(v1, axis=1)
    v2_norm = TLA.norm(v2, axis=1)

    errors = (v2_norm - v1_norm) / v1_norm * 100

    if return_abs:
        errors = torch.abs(errors)

    if not is_tensor:
        errors = errors.numpy()

    return errors


@overload
def angle_error(v1: Tensor, v2: Tensor) -> Tensor: ...


@overload
def angle_error(v1: np.ndarray, v2: np.ndarray) -> np.ndarray: ...


def angle_error(
    v1: Tensor | np.ndarray,
    v2: Tensor | np.ndarray,
) -> Tensor | np.ndarray:
    """
    CalcuTLAtes the angle error between two vectors in °.

    Parameters:
    - v1 (array_like): First input vector.
    - v2 (array_like): Second input vector.

    Returns:
    - float: The angle error between v1 and v2 in °.
    """
    is_tensor = isinstance(v1, Tensor) and isinstance(v2, Tensor)

    if not is_tensor:
        v1, v2 = torch.tensor(v1), torch.tensor(v2)

    v1_norm = TLA.norm(v1, axis=-1)
    v2_norm = TLA.norm(v2, axis=-1)
    v1tv2 = torch.sum(v1 * v2, dim=-1)  # type: ignore
    arg = v1tv2 / v1_norm / v2_norm
    arg[arg > 1] = 1
    arg[arg < -1] = -1

    errors = torch.rad2deg(torch.arccos(arg))

    if not is_tensor:
        errors = errors.numpy()

    return errors


def calculate_metrics_baseline(
    B: Tensor,
    return_abs: bool = True,
) -> Tuple[Tensor, Tensor]:
    B_demag, B_reduced = B[..., :3], B[..., 3:]
    angle_errors = angle_error(B_demag, B_reduced)
    amp_errors = relative_amplitude_error(B_demag, B_reduced, return_abs)

    return angle_errors, amp_errors


def calculate_metrics_trained(
    X: Tensor,
    B: Tensor,
    model,
    return_abs: bool = True,
) -> Tuple[Tensor, Tensor]:
    B_demag = B[..., :3]

    with torch.no_grad():
        B_corrected = model(X)

    angle_errors = angle_error(B_demag, B_corrected)
    amp_errors = relative_amplitude_error(B_demag, B_corrected, return_abs)

    return angle_errors, amp_errors


def calculate_metrics_trained_gnn(
    data: Data,
    model,
    return_abs: bool = True,
) -> Tuple[Tensor, Tensor]:
    B_demag, B_reduced = data.y[..., :3], data.y[..., 3:]  # type: ignore

    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
        B_corrected = model.correct_ansatz(B_reduced, predictions)

    angle_errors = angle_error(B_demag, B_corrected)
    amp_errors = relative_amplitude_error(B_demag, B_corrected, return_abs)

    return angle_errors, amp_errors


def vector_field_correlation(B1, B2):
    """
    Calculate vector correlation between vector fields B1 and B2 according to:
                     Σ B1,i · B2,i
    CV_vec = -------------------------------
             (Σ |B1,i|² · Σ |B2,i|²)^(1/2)

    Args:
        B1 (torch.Tensor): First vector field with shape (N, 3)
        B2 (torch.Tensor): Second vector field with shape (N, 3)

    Returns:
        torch.Tensor: Scalar correlation value
    """
    # Check if shapes match and are of the expected form
    if B1.shape != B2.shape:
        raise ValueError(
            f"Input vector fields must have the same shape. Got {B1.shape} and {B2.shape}"
        )

    if B1.shape[1] != 3:
        raise ValueError(f"Expected vector fields with shape (N, 3), got {B1.shape}")

    # Calculate dot product at each point
    # (B1 * B2).sum(dim=1) gives the dot product for each point
    point_dot_products = torch.sum(B1 * B2, dim=1)

    # Sum all dot products for the numerator
    numerator = torch.sum(point_dot_products)

    # Calculate the squared magnitudes at each point
    B1_squared_magnitudes = torch.sum(B1 * B1, dim=1)
    B2_squared_magnitudes = torch.sum(B2 * B2, dim=1)

    # Sum the squared magnitudes
    sum_B1_squared = torch.sum(B1_squared_magnitudes)
    sum_B2_squared = torch.sum(B2_squared_magnitudes)

    # Calculate denominator: sqrt of product of sum of squared magnitudes
    denominator = torch.sqrt(sum_B1_squared * sum_B2_squared)

    # Calculate correlation coefficient
    correlation = numerator / denominator

    return correlation

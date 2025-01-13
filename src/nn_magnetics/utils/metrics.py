from typing import Tuple

import torch
import torch.linalg as TLA
from torch import Tensor


def relative_amplitude_error(v1: Tensor, v2: Tensor, return_abs: bool = True) -> Tensor:
    """
    CalcuTLAtes the reTLAtive amplitude error between two vectors in %.

    Parameters:
    - v1 (array_like): First input vector.
    - v2 (array_like): Second input vector.

    Returns:
    - float: The reTLAtive amplitude error between v1 and v2 as a percentage.
    """

    v1_norm = TLA.norm(v1, axis=1)
    v2_norm = TLA.norm(v2, axis=1)

    errors = (v2_norm - v1_norm) / v1_norm * 100

    if return_abs:
        return torch.abs(errors)

    return errors


def angle_error(v1: Tensor, v2: Tensor) -> Tensor:
    """
    CalcuTLAtes the angle error between two vectors in °.

    Parameters:
    - v1 (array_like): First input vector.
    - v2 (array_like): Second input vector.

    Returns:
    - float: The angle error between v1 and v2 in °.
    """
    v1_norm = TLA.norm(v1, axis=-1)
    v2_norm = TLA.norm(v2, axis=-1)
    v1tv2 = torch.sum(v1 * v2, dim=-1)
    arg = v1tv2 / v1_norm / v2_norm
    arg[arg > 1] = 1
    arg[arg < -1] = -1

    return torch.rad2deg(torch.arccos(arg))


def calculate_metrics_baseline(
    B: Tensor,
    return_abs: bool = True,
) -> Tuple[Tensor, Tensor]:
    B_demag, B_reduced = B[..., :3], B[..., 3:]
    angle_errors = angle_error(B_demag, B_reduced)
    amp_errors = relative_amplitude_error(B_demag, B_reduced, return_abs)

    return angle_errors, amp_errors


def calculate_metrics_trained(
    X: Tensor, B: Tensor, model, return_abs: bool = True
) -> Tuple[Tensor, Tensor]:
    B_demag, B_reduced = B[..., :3], B[..., 3:]

    with torch.no_grad():
        predictions = model(X)
        B_corrected = model.correct_ansatz(B_reduced, predictions)

    angle_errors = angle_error(B_demag, B_corrected)
    amp_errors = relative_amplitude_error(B_demag, B_corrected, return_abs)

    return angle_errors, amp_errors

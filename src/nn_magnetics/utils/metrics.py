from typing import Tuple
import numpy as np
import torch

from nn_magnetics.pytorch.corrections import (
    field_correction,
    no_op,
    angle_amp_correction,
    ArrayLike,
)

eps = 1e-10


def relative_amplitude_error(v1, v2, return_abs=True):
    """
    Calculates the relative amplitude error between two vectors in %.

    Parameters:
    - v1 (array_like): First input vector.
    - v2 (array_like): Second input vector.

    Returns:
    - float: The relative amplitude error between v1 and v2 as a percentage.
    """

    v1_norm = np.linalg.norm(v1, axis=1)
    v2_norm = np.linalg.norm(v2, axis=1)

    errors = (v2_norm - v1_norm) / v1_norm * 100

    if return_abs:
        return np.abs(errors)

    return errors


def angle_error(v1, v2):
    """
    Calculates the angle error between two vectors in °.

    Parameters:
    - v1 (array_like): First input vector.
    - v2 (array_like): Second input vector.

    Returns:
    - float: The angle error between v1 and v2 in °.
    """
    if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        v1, v2 = v1.numpy(), v2.numpy()

    v1_norm = np.linalg.norm(v1, axis=-1)
    v2_norm = np.linalg.norm(v2, axis=-1)
    v1tv2 = np.sum(v1 * v2, axis=-1)
    arg = v1tv2 / v1_norm / v2_norm
    arg[arg > 1] = 1
    arg[arg < -1] = -1

    return np.rad2deg(np.arccos(arg))


def calculate_metrics_baseline(
    B: np.ndarray,
    return_abs=True,
) -> Tuple[np.ndarray, ...]:
    B_true, B_reduced = no_op(B)

    angle_errors = angle_error(B_true, B_reduced)
    amplitude_errors = relative_amplitude_error(
        B_true,
        B_reduced,
        return_abs=return_abs,
    )

    return angle_errors, amplitude_errors


def calculate_metrics(B: ArrayLike, B_pred: ArrayLike, return_abs=True):
    angles, amplitudes = B_pred[..., :3], B_pred[..., 3]
    B_true, B_corrected = angle_amp_correction(B, angles, amplitudes)

    batch_angle_errors = angle_error(B_true, B_corrected)
    batch_amplitude_errors = relative_amplitude_error(
        B_true,
        B_corrected,
        return_abs=return_abs,
    )
    return batch_angle_errors, batch_amplitude_errors

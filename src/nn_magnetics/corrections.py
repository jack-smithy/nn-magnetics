from typing import Tuple
import numpy as np


def field_correction(B: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, ...]:
    B_demag, B_reduced = B[..., :3], B[..., 3:]
    return B_demag, B_reduced * preds


def no_op(B: np.ndarray) -> Tuple[np.ndarray, ...]:
    return B[..., :3], B[..., 3:]


def amplitude_correction(B: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, ...]:
    B_demag, B_reduced = B[..., :3], B[..., 3:]
    return B_demag, preds * B_reduced

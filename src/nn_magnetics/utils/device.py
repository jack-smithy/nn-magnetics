import torch
from torch.utils.data import Dataset
from typing import TypeVar, Any
import functools
import jax


def get_device(use_accelerators: bool = True) -> str:
    if not use_accelerators:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"

    return "cpu"

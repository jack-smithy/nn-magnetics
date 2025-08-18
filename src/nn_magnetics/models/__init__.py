from nn_magnetics.models.base import BaseNetwork
from nn_magnetics.models.networks import (
    AngleAmpCorrectionNetwork,
    FieldCorrectionNetwork,
    AdditionCorrectionNetwork,
    SphericalCorrectionNetwork,
    QuaternionNet,
)
from nn_magnetics.models.utils import get_num_params

__all__ = [
    "BaseNetwork",
    "AngleAmpCorrectionNetwork",
    "FieldCorrectionNetwork",
    "QuaternionNet",
    "get_num_params",
    "AdditionCorrectionNetwork",
]

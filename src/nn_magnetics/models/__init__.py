from nn_magnetics.models.base import Network, GNN
from nn_magnetics.models.networks import (
    AngleAmpCorrectionNetwork,
    FieldCorrectionNetwork,
    AmpCorrectionNetwork,
    QuaternionNet,
)
from nn_magnetics.models.utils import get_num_params, DivergenceLoss

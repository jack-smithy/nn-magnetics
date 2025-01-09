from nn_magnetics.utils.physics import (
    demagnetizing_factor,
    split_tensor,
)
from nn_magnetics.utils.device import get_device
from nn_magnetics.utils.metrics import (
    relative_amplitude_error,
    angle_error,
    calculate_metrics,
    calculate_metrics_baseline,
)

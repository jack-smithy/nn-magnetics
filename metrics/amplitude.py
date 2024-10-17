import numpy as np


def relative_amplitude_error(v1, v2):
    """
    Calculates the relative amplitude error between two vectors in %.

    Parameters:
    - v1 (array_like): First input vector.
    - v2 (array_like): Second input vector.

    Returns:
    - float: The relative amplitude error between v1 and v2 as a percentage.
    """

    v1_norm = np.linalg.norm(v1, axis=-1)
    v2_norm = np.linalg.norm(v2, axis=-1)

    return (v2_norm - v1_norm) / v1_norm * 100


if __name__ == "__main__":
    v1 = np.ones((5, 3))
    v2 = np.ones((5, 3)) * 1.1

    res = relative_amplitude_error(v1, v2)
    print(res)

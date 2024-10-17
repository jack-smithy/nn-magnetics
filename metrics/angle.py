import numpy as np
import torch


def angle_error(v1, v2):
    """
    Calculates the angle error between two vectors in °.

    Parameters:
    - v1 (array_like): First input vector.
    - v2 (array_like): Second input vector.

    Returns:
    - float: The angle error between v1 and v2 in °.
    """

    v1_norm = np.linalg.norm(v1, axis=-1)
    v2_norm = np.linalg.norm(v2, axis=-1)
    v1tv2 = np.sum(v1 * v2, axis=-1)
    arg = v1tv2 / v1_norm / v2_norm
    arg[arg > 1] = 1
    arg[arg < -1] = -1

    return np.rad2deg(np.arccos(arg))


def angle_error_torch(v1, v2) -> np.ndarray:
    v1_norm = torch.linalg.norm(v1, axis=-1)
    v2_norm = torch.linalg.norm(v2, axis=-1)
    v1tv2 = torch.sum(v1 * v2, dim=-1)
    arg = v1tv2 / v1_norm / v2_norm
    arg[arg > 1] = 1
    arg[arg < -1] = -1

    return torch.mean(torch.rad2deg(torch.arccos(arg)), dim=-1).numpy(force=True)


if __name__ == "__main__":
    v1 = np.random.random((5, 3))
    v2 = np.random.random((5, 3))

    v1t = torch.tensor(v1)
    v2t = torch.tensor(v2)

    res = angle_error(v1, v2)
    res_torch = angle_error_torch(v1t, v2t)

    print(res)
    print(res_torch)

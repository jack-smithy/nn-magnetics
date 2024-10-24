import numpy as np
import torch


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

    return np.abs((v2_norm - v1_norm) / v1_norm * 100)


def relative_amplitude_error_torch(v1: torch.Tensor, v2: torch.Tensor) -> np.ndarray:
    v1_norm = torch.linalg.norm(v1, dim=-1)
    v2_norm = torch.linalg.norm(v2, dim=-1)

    rel_error = torch.mean(
        torch.abs(
            torch.mul(
                torch.div(torch.sub(v2_norm, v1_norm), v1_norm),
                torch.tensor([100.0]),
            ),
        ),
        dim=-1,
    ).numpy(force=True)

    return rel_error


if __name__ == "__main__":
    v1 = np.ones((5, 3))
    v2 = np.ones((5, 3)) * 1.1

    res = relative_amplitude_error(v1, v2)
    print(res)

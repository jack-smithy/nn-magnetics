from typing import Type

import magpylib as mp
import numpy as np
import torch

from nn_magnetics.models.networks import Network
from nn_magnetics.utils.physics import Dz_cuboid_elementwise

torch.set_default_dtype(torch.float64)

axis_to_int = {"x": 0, "y": 1, "z": 2}


def rotate(array, initial_axis, final_axis, abs):
    assert array.ndim == 2 and array.shape[1] == 3, "wrong dimensions in input array"

    initial_axis = axis_to_int[initial_axis]
    final_axis = axis_to_int[final_axis]

    if initial_axis == final_axis:
        return array
    else:
        array_new = np.copy(array)
        array_new[:, final_axis] = array[:, initial_axis]
        if abs:
            array_new[:, initial_axis] = array[:, final_axis]
        else:
            array_new[:, initial_axis] = -array[:, final_axis]

    return array_new


def check_points_outside_magnet(dimension, position, points):
    """
    Check if points lie outside a cuboid defined by its center and dimensions.
    Uses vectorized operations for efficient computation.

    Args:
        center (np.ndarray): Shape (3,) array of (dx, dy, dz) coordinates of cuboid center
        dimensions (np.ndarray): Shape (3,) array of (a, b, c) side lengths of cuboid
        points (np.ndarray): Shape (N, 3) array of N points to check

    Returns:
        bool: True if all points are outside the cuboid, False otherwise
    """
    # Calculate relative coordinates of all points at once
    relative_coords = np.abs(points - position)
    # Compare with half-dimensions
    half_dimensions = dimension / 2
    # A point is inside if it's within bounds in all dimensions
    # This creates a boolean array of shape (N, 3)
    within_bounds = relative_coords <= half_dimensions
    # A point is inside only if it's within bounds in all dimensions
    # This reduces the (N, 3) array to (N,) array
    points_inside = np.all(within_bounds, axis=1)
    # Return True if no points are inside
    return not np.any(points_inside)


def B(
    position: np.ndarray,
    dimension: np.ndarray,
    polarization: np.ndarray,
    susceptibility: np.ndarray,
    points: np.ndarray,
    model_path: str,
    hidden_dim_factor: int,
    network_type: Type[Network],
) -> np.ndarray:
    """_summary_

    Args:
        dimension (np.ndarray): _description_
        polarization (np.ndarray): _description_
        susceptibility (np.ndarray): _description_
        points (np.ndarray): _description_
        model_path (str): _description_

    Returns:
        np.ndarray: _description_
    """

    model = network_type.load_from_path(model_path, hidden_dim_factor=hidden_dim_factor)
    Px, Py, Pz = polarization
    points = points - position

    # don't need to do anything special for Bz
    Bz = eval_nn(
        dimension=dimension,
        polarization=Pz,
        susceptibility=susceptibility,
        points=points,
        model=model,
    )

    # rotate for X
    dimension_rotated_X = np.expand_dims(dimension, axis=0)
    dimension_rotated_X = rotate(dimension_rotated_X, "x", "z", abs=True)
    dimension_rotated_X = np.squeeze(dimension_rotated_X)

    susceptibility_rotated_X = np.expand_dims(susceptibility, axis=0)
    susceptibility_rotated_X = rotate(susceptibility_rotated_X, "x", "z", abs=True)
    susceptibility_rotated_X = np.squeeze(susceptibility_rotated_X)

    points_rotated_X = rotate(points, "x", "z", abs=True)

    Bx = eval_nn(
        dimension=dimension_rotated_X,
        polarization=Px,
        susceptibility=susceptibility_rotated_X,
        points=points_rotated_X,
        model=model,
    )

    Bx = rotate(Bx, "z", "x", abs=True)

    # rotate for Y
    dimension_rotated_Y = np.expand_dims(dimension, axis=0)
    dimension_rotated_Y = rotate(dimension_rotated_Y, "y", "z", abs=True)
    dimension_rotated_Y = np.squeeze(dimension_rotated_Y)

    susceptibility_rotated_Y = np.expand_dims(susceptibility, axis=0)
    susceptibility_rotated_Y = rotate(susceptibility_rotated_Y, "y", "z", abs=True)
    susceptibility_rotated_Y = np.squeeze(susceptibility_rotated_Y)

    points_rotated_Y = rotate(points, "y", "z", abs=True)

    By = eval_nn(
        dimension=dimension_rotated_Y,
        polarization=Py,
        susceptibility=susceptibility_rotated_Y,
        points=points_rotated_Y,
        model=model,
    )

    By = rotate(By, "z", "y", abs=True)

    return Bx + By + Bz


def get_feature_vector(dimension, susceptibility, points):
    n_points = points.shape[0]
    characterization = torch.zeros((n_points, 5), dtype=torch.float64)
    characterization[:, 0] = dimension[0]
    characterization[:, 1] = dimension[1]
    characterization[:, 2] = susceptibility[0]
    characterization[:, 3] = susceptibility[1]
    characterization[:, 4] = susceptibility[2]

    points = torch.from_numpy(points)

    feature = torch.cat((characterization, points), dim=1)
    return feature.type(torch.float64)


def eval_nn(
    dimension: np.ndarray,
    polarization: float,
    susceptibility: np.ndarray,
    points: np.ndarray,
    model: Network,
) -> np.ndarray:
    """
    calculate corrected B-field for polarization=(0, 0, p) at any point outside magnet in (+, +, +) quadrant.

    Args:
        dimension (np.ndarray): _description_
        polarization (float): _description_
        susceptibility (np.ndarray): _description_
        points (np.ndarray): _description_
        model (AngleAmpCorrectionNetwork): _description_

    Returns:
        np.ndarray: _description_
    """

    # 1. normalize dimension and evaluation points
    # 2. record quadrant signs for evaluation points

    points[..., 0] /= dimension[0]
    points[..., 1] /= dimension[1]

    signs = np.sign(points)
    mask = signs[:, 2] == -1
    signs[mask] *= -1

    points = np.abs(points / dimension[2])
    dimension = dimension / dimension[2]

    # 3. calculate reduced polarization
    P_reduced = 1 / (1 + susceptibility[2] * Dz_cuboid_elementwise(*dimension))

    # 4. calculate analytical B-field
    magnet = mp.magnet.Cuboid(dimension=dimension, polarization=(0, 0, P_reduced))
    B_reduced = magnet.getB(points)
    B_reduced = torch.from_numpy(B_reduced)

    # 5. construct feature vector
    feature = get_feature_vector(
        dimension=dimension,
        susceptibility=susceptibility,
        points=points,
    )

    # 6. calculate correction factors & correct analytical field
    predictions = model(feature)
    B_corrected = model.correct_ansatz(B_reduced=B_reduced, prediction=predictions)
    B_corrected = B_corrected.detach().numpy()
    B_corrected = B_corrected * signs * polarization

    return B_corrected

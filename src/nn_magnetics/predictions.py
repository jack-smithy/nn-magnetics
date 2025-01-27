import magpylib as mp
import numpy as np
import torch
from magpylib_material_response import demag, meshing

from nn_magnetics.model import AngleAmpCorrectionNetwork
from nn_magnetics.utils.physics import demagnetizing_factor

torch.set_default_dtype(torch.float32)


def axis_string_to_int(s):
    if s == "x":
        return 0
    elif s == "y":
        return 1
    elif s == "z":
        return 2
    else:
        raise Exception('Sorry, input needs to be "x", "y" or "z"')


axis_to_int = {"x": 0, "y": 1, "z": 2}


def rotate(array, initial_axis, final_axis, abs):
    assert array.ndim == 2 and array.shape[1] == 3, "wrong dimensions in input array"

    initial_axis = axis_string_to_int(initial_axis)
    final_axis = axis_string_to_int(final_axis)

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


def D_cuboid(dimension):
    a, b, c = dimension / 2
    Dx = demagnetizing_factor(c, a, b)
    Dy = demagnetizing_factor(a, c, b)
    Dz = demagnetizing_factor(a, b, c)
    return np.array((Dx, Dy, Dz))


def process_inputs(X: np.ndarray, dimensions: np.ndarray):
    signs = np.sign(X)
    mask = signs[:, 2] == -1
    signs[mask] *= -1

    X_rotated = np.abs(X)
    X_normalized = X_rotated / dimensions[2]
    dimensions_normalized = dimensions / dimensions[2]

    return X_normalized, dimensions_normalized, signs


def _B(
    dimension: np.ndarray,
    polarization: np.ndarray,
    susceptibility: np.ndarray,
    points: np.ndarray,
    model_path: str,
) -> np.ndarray:
    # assume polarization=(0,0,p) only, but arbitrary susceptibility and dimensions

    # normalize the dimensions of the magnet so c=1
    # this is fine as the demag tensor is scale invariant

    points_normalized, dimension_normalized, signs = process_inputs(points, dimension)

    print(points_normalized)

    D = D_cuboid(dimension=dimension_normalized)
    Pz_norm = polarization[2]

    # calculate reduced polarization for analytical ansatz
    Pz_reduced = 1 - susceptibility[2] * D[2]

    # calculate for Z
    magnet_reduced = mp.magnet.Cuboid(
        dimension=dimension_normalized,
        polarization=(0, 0, Pz_reduced),
    )

    B_reduced = torch.from_numpy(magnet_reduced.getB(points_normalized)).float()

    # build feature matrix for nn calculation
    n_points = points.shape[0]
    characterization = torch.zeros((n_points, 5), dtype=torch.float32)
    characterization[:, 0] = dimension_normalized[0]
    characterization[:, 1] = dimension_normalized[1]
    characterization[:, 2] = susceptibility[0]
    characterization[:, 3] = susceptibility[1]
    characterization[:, 4] = susceptibility[2]

    feature = torch.cat(
        (characterization, torch.from_numpy(points_normalized)),
        dim=1,
    ).type(torch.float32)

    # load model and make predictions
    model = AngleAmpCorrectionNetwork.load_from_path(model_path)

    prediction = model(feature)
    B_corrected = model.correct_ansatz(B_reduced=B_reduced, prediction=prediction)
    B_corrected = B_corrected.detach().numpy()
    Bz = Pz_norm * B_corrected * signs

    return Bz


def B(
    dimension: np.ndarray,
    polarization: np.ndarray,
    susceptibility: np.ndarray,
    points: np.ndarray,
    model_path: str,
) -> np.ndarray:
    model = AngleAmpCorrectionNetwork.load_from_path(model_path)
    Pz = polarization[2]

    Bz = eval_nn(
        dimension=dimension,
        polarization=Pz,
        susceptibility=susceptibility,
        points=points,
        model=model,
    )

    return Bz


def get_feature_vector(dimension, susceptibility, points):
    n_points = points.shape[0]
    characterization = torch.zeros((n_points, 5), dtype=torch.float32)
    characterization[:, 0] = dimension[0]
    characterization[:, 1] = dimension[1]
    characterization[:, 2] = susceptibility[0]
    characterization[:, 3] = susceptibility[1]
    characterization[:, 4] = susceptibility[2]

    points = torch.from_numpy(points)

    feature = torch.cat((characterization, points), dim=1)
    return feature.type(torch.float32)


def eval_nn(
    dimension: np.ndarray,
    polarization: float,
    susceptibility: np.ndarray,
    points: np.ndarray,
    model: AngleAmpCorrectionNetwork,
) -> np.ndarray:
    """
    calculate corrected B-field for polarization=(0, 0, p) at any point outside magnet.

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

    signs = np.sign(points)
    mask = signs[:, 2] == -1
    signs[mask] *= -1

    points = np.abs(points / dimension[2])
    dimension = dimension / dimension[2]

    # 3. calculate reduced polarization
    P_reduced = 1 - susceptibility[2] * demagnetizing_factor(*dimension)

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


def test_B_demag(dimension, polarization, susceptibility, X):
    magnet = mp.magnet.Cuboid(dimension=dimension, polarization=polarization)
    magnet.susceptibility = susceptibility  # type: ignore
    mesh = meshing.mesh_Cuboid(magnet, target_elems=100)
    demag.apply_demag(mesh, inplace=True)
    return mesh.getB(X)


def test_B_ana(dimension, polarization, X):
    magnet = mp.magnet.Cuboid(dimension=dimension, polarization=polarization)
    return magnet.getB(X)


if __name__ == "__main__":
    dimension = np.array((1, 1, 0.5))
    polarization = np.array((0, 0, 2))
    susceptibility = np.array((0.1, 0.2, 0.3))
    X = np.array([[1.4, 2, 2.5], [-1.4, 2, -2.5]])

    B_demag = test_B_demag(
        dimension=dimension,
        polarization=polarization,
        susceptibility=susceptibility,
        X=X,
    )
    # X = rotate(X, "y", "z", abs=True)

    # B_true = test_B_demag(
    #     dimension=dimension,
    #     polarization=polarization,
    #     susceptibility=susceptibility,
    #     X=X,
    # )

    # B_ana = rotate(B_ana, "z", "y", abs=True)
    # B_nn = _B(
    #     dimension=dimension,
    #     polarization=polarization,
    #     susceptibility=susceptibility,
    #     points=X,
    #     model_path="/Users/jacksmith/Documents/PhD/nn-magnetics/results/3dof_chi/2025-01-15 12:23:07.808007/best_weights.pt",
    # )
    B_nn = B(
        dimension=dimension,
        polarization=polarization,
        susceptibility=susceptibility,
        points=X,
        model_path="/Users/jacksmith/Documents/PhD/nn-magnetics/results/3dof_chi/2025-01-15 12:23:07.808007/best_weights.pt",
    )

    print("True solution")
    print(B_demag)
    print("\n\nNN solution")
    print(B_nn)

    nn_error = np.mean((B_demag - B_nn) / B_demag * 100, axis=-1)
    # ana_error = np.mean((B_demag - B_ana) / B_demag * 100, axis=-1)

    # # print(f"Error analytical solution: {ana_error}")
    print(f"\n\nError nn solution: {nn_error}")

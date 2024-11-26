from typing import Dict, List, Tuple

import h5py
import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution as de

from nn_magnetics.optimize import define_movement_side, define_movement_under


def read_hdf5(path: str = "scans") -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(f"{path}/Hippie_Under_scan.hdf5", "r") as f:
        field_measured1 = np.array(f["Hippie_scan_under_run_0"])

    with h5py.File(f"{path}/Hippie_Side_scan.hdf5", "r") as f:
        field_measured2 = np.array(f["Hippie_scan_side_run_0"])

    return field_measured1, field_measured2


def prepare_measurements(path: str = "scans"):
    field_measured1, field_measured2 = read_hdf5(path)

    field_measured1[:, 2] = -field_measured1[:, 2]
    field_measured2[:, 2] = -field_measured2[:, 2]

    field_measured1 /= 1000  # in T
    field_measured2 /= 1000  # in T

    field_measured2 = field_measured2.reshape((4, -1, 3))

    _, s1 = define_movement_under()
    _, s2 = define_movement_side()

    positions1 = s1.position
    positions2 = s2.position.reshape((4, -1, 3))

    positions1 /= 1000  # in m
    positions2 /= 1000  # in m

    field_measured1 /= 2  # wrong LSB -> mT conversion
    field_measured2 /= 2  # wrong LSB -> mT conversion

    field_measured2_rotated = np.copy(field_measured2)
    positions2_rotated = np.copy(positions2)

    field_measured2_rotated[1, :, 0] = field_measured2[1, :, 1]
    field_measured2_rotated[1, :, 1] = -field_measured2[1, :, 0]
    field_measured2_rotated[2, :, 0] = -field_measured2[2, :, 0]
    field_measured2_rotated[2, :, 1] = -field_measured2[2, :, 1]
    field_measured2_rotated[3, :, 0] = -field_measured2[3, :, 1]
    field_measured2_rotated[3, :, 1] = field_measured2[3, :, 0]

    positions2_rotated[1, :, 0] = positions2[1, :, 1]
    positions2_rotated[1, :, 1] = -positions2[1, :, 0]
    positions2_rotated[2, :, 0] = -positions2[2, :, 0]
    positions2_rotated[2, :, 1] = -positions2[2, :, 1]
    positions2_rotated[3, :, 0] = -positions2[3, :, 1]
    positions2_rotated[3, :, 1] = positions2[3, :, 0]

    return positions1, positions2_rotated, field_measured1, field_measured2_rotated


def cost_function(
    x,
    positions1,
    positions2_rotated,
    field_measured1,
    field_measured2_rotated,
    plot,
):
    (
        polarization_magnitude,
        polarization_phi,
        polarization_theta,
        magnet_angle,
        magnet_position_x,
        magnet_position_y,
        magnet_position_z,
        magnet_dim_tol_x,
        magnet_dim_tol_y,
        magnet_dim_tol_z,
        sensitivity_x,
        sensitivity_y,
        sensitivity_z,
        offset_x,
        offset_y,
        offset_z,
        chi_perp,
        chi_long,
    ) = x

    dimension = np.array(
        (
            5e-3 + magnet_dim_tol_x,
            5e-3 + magnet_dim_tol_y,
            5e-3 + magnet_dim_tol_z,
        )
    )

    polarization = polarization_magnitude * np.array(
        (
            np.cos(polarization_phi) * np.sin(polarization_theta),
            np.sin(polarization_phi) * np.sin(polarization_theta),
            np.cos(polarization_theta),
        )
    )

    magnet = magpy.magnet.Cuboid(
        position=(magnet_position_x, magnet_position_y, magnet_position_z),
        dimension=dimension,
        polarization=polarization,
        susceptibility=(chi_perp, chi_perp, chi_long),  # type: ignore
    )

    magnet.rotate_from_angax(magnet_angle, "z", degrees=False)

    field_simulated1 = magnet.getB(positions1)
    field_simulated2 = magnet.getB(positions2_rotated)

    ###########
    # sensor characterization
    field_simulated1[:, 0] *= sensitivity_x
    field_simulated1[:, 1] *= sensitivity_y
    field_simulated1[:, 2] *= sensitivity_z
    field_simulated1[:, 0] += offset_x
    field_simulated1[:, 1] += offset_y
    field_simulated1[:, 2] -= offset_z

    field_simulated2[0, :, 0] *= sensitivity_x
    field_simulated2[0, :, 1] *= sensitivity_y
    field_simulated2[0, :, 2] *= sensitivity_z
    field_simulated2[0, :, 0] += offset_x
    field_simulated2[0, :, 1] += offset_y
    field_simulated2[0, :, 2] -= offset_z

    field_simulated2[1, :, 0] *= sensitivity_y
    field_simulated2[1, :, 1] *= sensitivity_x
    field_simulated2[1, :, 2] *= sensitivity_z
    field_simulated2[1, :, 0] += offset_y
    field_simulated2[1, :, 1] -= offset_x
    field_simulated2[1, :, 2] -= offset_z

    field_simulated2[2, :, 0] *= sensitivity_x
    field_simulated2[2, :, 1] *= sensitivity_y
    field_simulated2[2, :, 2] *= sensitivity_z
    field_simulated2[2, :, 0] -= offset_x
    field_simulated2[2, :, 1] -= offset_y
    field_simulated2[2, :, 2] -= offset_z

    field_simulated2[3, :, 0] *= sensitivity_y
    field_simulated2[3, :, 1] *= sensitivity_x
    field_simulated2[3, :, 2] *= sensitivity_z
    field_simulated2[3, :, 0] -= offset_y
    field_simulated2[3, :, 1] += offset_x
    field_simulated2[3, :, 2] -= offset_z

    if plot:
        plt.plot(field_measured1[:, 0], label="measured x")
        plt.plot(field_measured1[:, 1], label="measured y")
        plt.plot(field_measured1[:, 2], label="measured z")

        plt.plot(field_simulated1[:, 0], label="simulated x")
        plt.plot(field_simulated1[:, 1], label="simulated y")
        plt.plot(field_simulated1[:, 2], label="simulated z")
        plt.legend()
        plt.show()

        for i in range(4):
            plt.plot(field_measured2_rotated[i, :, 0], label="measured x")
            plt.plot(field_measured2_rotated[i, :, 1], label="measured y")
            plt.plot(field_measured2_rotated[i, :, 2], label="measured z")

            plt.plot(field_simulated2[i, :, 0], label="simulated x")
            plt.plot(field_simulated2[i, :, 1], label="simulated y")
            plt.plot(field_simulated2[i, :, 2], label="simulated z")
            plt.legend()
            plt.show()

        np.savez(
            "analytic_results_relative_extended_sensorcharacterization.npz",
            positions1=positions1,
            positions2_rotated=positions2_rotated,
            field_measured1=field_measured1,
            field_measured2_rotated=field_measured2_rotated,
            field_simulated1=field_simulated1,
            field_simulated2=field_simulated2,
        )

    result1 = (
        np.linalg.norm(field_simulated1 - field_measured1, axis=-1) ** 2
        / np.linalg.norm(field_measured1, axis=-1) ** 2
    )
    result2 = (
        np.linalg.norm(field_simulated2 - field_measured2_rotated, axis=-1) ** 2
        / np.linalg.norm(field_measured2_rotated, axis=-1) ** 2
    )

    result = np.sum(result1) + np.sum(result2)

    return result


def fit(
    polarization_magnitude,
    positions1,
    positions2_rotated,
    field_measured1,
    field_measured2_rotated,
    maxiter,
    popsize=40,
) -> OptimizeResult:
    result = de(
        func=cost_function,
        bounds=(
            (polarization_magnitude, polarization_magnitude),  # polarization magnitude
            (np.pi / 2 - np.pi / 10, np.pi / 2 + np.pi / 10),  # polarization phi
            (np.pi / 2 - np.pi / 10, np.pi / 2 + np.pi / 10),  # polarization theta
            (-np.pi / 10, np.pi / 10),  # magnet angle
            (-1e-3, 1e-3),  # magnet pos x
            (-1e-3, 1e-3),  # magnet pos y
            (-1e-3, 1e-3),  # magnet pos z
            (-1e-3, 1e-3),  # magnet dim tol x
            (-1e-3, 1e-3),  # magnet dim tol y
            (-1e-3, 1e-3),  # magnet dim tol z
            (0.7, 1.5),  # sensitivity x
            (0.7, 1.5),  # sensitivity y
            (0.7, 1.5),  # sensitivity z
            (-1e-3, 1e-3),  # offset x
            (-1e-3, 1e-3),  # offset y
            (-1e-3, 1e-3),  # offset z
            (0.0, 1.0),  # chi_perp
            (0.0, 1.0),  # chi_long
        ),
        args=(
            positions1,
            positions2_rotated,
            field_measured1,
            field_measured2_rotated,
            False,
        ),
        maxiter=maxiter,
        popsize=popsize,
        workers=1,
        disp=True,
    )

    return result


def evaluate(
    x,
    positions1,
    positions2_rotated,
    field_measured1,
    field_measured2_rotated,
):
    (
        polarization_magnitude,
        polarization_phi,
        polarization_theta,
        magnet_angle,
        magnet_position_x,
        magnet_position_y,
        magnet_position_z,
        magnet_dim_tol_x,
        magnet_dim_tol_y,
        magnet_dim_tol_z,
        sensitivity_x,
        sensitivity_y,
        sensitivity_z,
        offset_x,
        offset_y,
        offset_z,
        chi_perp,
        chi_long,
    ) = x

    dimension = np.array(
        (
            5e-3 + magnet_dim_tol_x,
            5e-3 + magnet_dim_tol_y,
            5e-3 + magnet_dim_tol_z,
        )
    )

    polarization = polarization_magnitude * np.array(
        (
            np.cos(polarization_phi) * np.sin(polarization_theta),
            np.sin(polarization_phi) * np.sin(polarization_theta),
            np.cos(polarization_theta),
        )
    )

    magnet = magpy.magnet.Cuboid(
        position=(magnet_position_x, magnet_position_y, magnet_position_z),
        dimension=dimension,
        polarization=polarization,
        susceptibility=(chi_perp, chi_perp, chi_long),  # type: ignore
    )

    magnet.rotate_from_angax(magnet_angle, "z", degrees=False)

    field_simulated1 = magnet.getB(positions1)
    field_simulated2 = magnet.getB(positions2_rotated)

    ###########
    # sensor characterization
    field_simulated1[:, 0] *= sensitivity_x
    field_simulated1[:, 1] *= sensitivity_y
    field_simulated1[:, 2] *= sensitivity_z
    field_simulated1[:, 0] += offset_x
    field_simulated1[:, 1] += offset_y
    field_simulated1[:, 2] -= offset_z

    field_simulated2[0, :, 0] *= sensitivity_x
    field_simulated2[0, :, 1] *= sensitivity_y
    field_simulated2[0, :, 2] *= sensitivity_z
    field_simulated2[0, :, 0] += offset_x
    field_simulated2[0, :, 1] += offset_y
    field_simulated2[0, :, 2] -= offset_z

    field_simulated2[1, :, 0] *= sensitivity_y
    field_simulated2[1, :, 1] *= sensitivity_x
    field_simulated2[1, :, 2] *= sensitivity_z
    field_simulated2[1, :, 0] += offset_y
    field_simulated2[1, :, 1] -= offset_x
    field_simulated2[1, :, 2] -= offset_z

    field_simulated2[2, :, 0] *= sensitivity_x
    field_simulated2[2, :, 1] *= sensitivity_y
    field_simulated2[2, :, 2] *= sensitivity_z
    field_simulated2[2, :, 0] -= offset_x
    field_simulated2[2, :, 1] -= offset_y
    field_simulated2[2, :, 2] -= offset_z

    field_simulated2[3, :, 0] *= sensitivity_y
    field_simulated2[3, :, 1] *= sensitivity_x
    field_simulated2[3, :, 2] *= sensitivity_z
    field_simulated2[3, :, 0] -= offset_y
    field_simulated2[3, :, 1] += offset_x
    field_simulated2[3, :, 2] -= offset_z

    plt.plot(field_measured1[:, 0], label="measured x")
    plt.plot(field_measured1[:, 1], label="measured y")
    plt.plot(field_measured1[:, 2], label="measured z")

    plt.plot(field_simulated1[:, 0], label="simulated x")
    plt.plot(field_simulated1[:, 1], label="simulated y")
    plt.plot(field_simulated1[:, 2], label="simulated z")
    plt.legend()
    plt.show()

    for i in range(4):
        plt.plot(field_measured2_rotated[i, :, 0], label="measured x")
        plt.plot(field_measured2_rotated[i, :, 1], label="measured y")
        plt.plot(field_measured2_rotated[i, :, 2], label="measured z")

        plt.plot(field_simulated2[i, :, 0], label="simulated x")
        plt.plot(field_simulated2[i, :, 1], label="simulated y")
        plt.plot(field_simulated2[i, :, 2], label="simulated z")
        plt.legend()
        plt.show()


def result_to_dict(result: OptimizeResult) -> Dict[str, float]:
    optimized_parameters = result.x
    parameter_names: List[str] = [
        "polarization_magnitude",
        "polarization_phi",
        "polarization_theta",
        "magnet_angle",
        "magnet_pos_x",
        "magnet_pos_y",
        "magnet_pos_z",
        "magnet_dim_tol_x",
        "magnet_dim_tol_y",
        "magnet_dim_tol_z",
        "sensitivity_x",
        "sensitivity_y",
        "sensitivity_z",
        "offset_x",
        "offset_y",
        "offset_z",
        "chi_perp",
        "chi_long",
    ]

    results_dict = {}
    for k, v in zip(parameter_names, optimized_parameters):
        results_dict[k] = v

    return results_dict

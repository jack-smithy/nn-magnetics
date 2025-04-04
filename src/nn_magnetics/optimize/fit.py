import json
from typing import Dict, List, Tuple

import h5py
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import torch
from magpylib.magnet import Cuboid
from magpylib import Collection
from magpylib_material_response import demag, meshing
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution as de

from nn_magnetics.optimize import define_movement_side, define_movement_under
from nn_magnetics.predictions import B
from nn_magnetics.utils.metrics import relative_amplitude_error, angle_error


def read_hdf5(path: str = "scans") -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(f"{path}/Hippie_Under_scan.hdf5", "r") as f:
        field_measured1 = np.array(f["Hippie_scan_under_run_0"])

    with h5py.File(f"{path}/Hippie_Side_scan.hdf5", "r") as f:
        field_measured2 = np.array(f["Hippie_scan_side_run_0"])

    return field_measured1, field_measured2


def prepare_measurements_mock(path: str = "scans"):
    _, s1 = define_movement_under()
    _, s2 = define_movement_side()

    positions1 = s1.position
    positions2 = s2.position.reshape((4, -1, 3))

    magnet = Cuboid(dimension=(5, 5, 5), polarization=(0, 0, 1))
    magnet.susceptibility = (0, 0, 0.5)  # type: ignore
    mesh = meshing.mesh_Cuboid(magnet, target_elems=100)
    demag.apply_demag(mesh, inplace=True)

    field_measured1, field_measured2 = mesh.getB(positions1), mesh.getB(positions2)

    field_measured1[:, 2] = -field_measured1[:, 2]
    field_measured2[:, 2] = -field_measured2[:, 2]

    field_measured2 = field_measured2.reshape((4, -1, 3))

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


def prepare_measurements(path: str = "scans"):
    _, s1 = define_movement_under()
    _, s2 = define_movement_side()

    positions1 = s1.position
    positions2 = s2.position.reshape((4, -1, 3))

    positions1 /= 1000  # in m
    positions2 /= 1000  # in m

    field_measured1, field_measured2 = read_hdf5(path)

    field_measured1[:, 2] = -field_measured1[:, 2]
    field_measured2[:, 2] = -field_measured2[:, 2]

    field_measured1 /= 1000  # in T
    field_measured2 /= 1000  # in T

    field_measured2 = field_measured2.reshape((4, -1, 3))

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
    model_path,
):
    (
        polarization_magnitude,
        polarization_phi,
        polarization_theta,
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
        chi_x,
        chi_y,
        chi_z,
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

    position = np.array((magnet_position_x, magnet_position_y, magnet_position_z))

    susceptibility = np.array((chi_x, chi_y, chi_z))

    cuboid = Cuboid(position=position, dimension=dimension, polarization=polarization)
    cuboid.susceptibility = susceptibility  # type: ignore
    mesh = meshing.mesh_Cuboid(cuboid=cuboid, target_elems=10)

    # field_simulated1 = B(
    #     position=position,
    #     dimension=dimension,
    #     polarization=polarization,
    #     susceptibility=susceptibility,
    #     points=positions1,
    #     model_path=model_path,
    # )

    field_simulated1 = mesh.getB(positions1)

    field_simulated2 = []
    for direction in positions2_rotated:
        #     field_simulated2_direction = B(
        #         position=position,
        #         dimension=dimension,
        #         polarization=polarization,
        #         susceptibility=susceptibility,
        #         points=direction,
        #         model_path=model_path,
        #     )
        #     field_simulated2.append(field_simulated2_direction)
        field_simulated2_direction = mesh.getB(direction)
        field_simulated2.append(field_simulated2_direction)

    field_simulated2 = np.array(field_simulated2)

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
    positions1,
    positions2_rotated,
    field_measured1,
    field_measured2_rotated,
    maxiter,
    save_path,
    model_path,
    popsize=40,
) -> OptimizeResult:
    def _save_intermediate(x, val):
        with open(f"{save_path}/best_params.json", "w+") as f:
            json.dump(arr_to_dict(x), f)

        return

    result = de(
        func=cost_function,
        bounds=(
            (0, 2),  # polarization magnitude
            (np.pi / 2 - np.pi / 10, np.pi / 2 + np.pi / 10),  # polarization phi
            (np.pi / 2 - np.pi / 10, np.pi / 2 + np.pi / 10),  # polarization theta
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
            (0.0, 1.0),  # chi_x
            (0.0, 1.0),  # chi_y
            (0.0, 1.0),  # chi_z
        ),
        args=(
            positions1,
            positions2_rotated,
            field_measured1,
            field_measured2_rotated,
            model_path,
        ),
        maxiter=maxiter,
        popsize=popsize,
        workers=-1,
        disp=True,
        callback=_save_intermediate,
        polish=True,
    )

    return result


def evaluate(
    x,
    positions1,
    positions2_rotated,
    field_measured1,
    field_measured2_rotated,
    model_path,
    save_dir=None,
):
    (
        polarization_magnitude,
        polarization_phi,
        polarization_theta,
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
        chi_x,
        chi_y,
        chi_z,
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

    position = np.array((magnet_position_x, magnet_position_y, magnet_position_z))

    susceptibility = np.array((chi_x, chi_y, chi_z))

    # field_simulated1 = B(
    #     position=position,
    #     dimension=dimension,
    #     polarization=polarization,
    #     susceptibility=susceptibility,
    #     points=positions1,
    #     model_path=model_path,
    # )

    # field_simulated2 = []
    # for direction in positions2_rotated:
    #     field_simulated2_direction = B(
    #         position=position,
    #         dimension=dimension,
    #         polarization=polarization,
    #         susceptibility=susceptibility,
    #         points=direction,
    #         model_path=model_path,
    #     )
    #     field_simulated2.append(field_simulated2_direction)

    # field_simulated2 = np.array(field_simulated2)

    cuboid = Cuboid(position=position, dimension=dimension, polarization=polarization)
    cuboid.susceptibility = susceptibility  # type: ignore
    mesh = meshing.mesh_Cuboid(cuboid=cuboid, target_elems=10)

    # field_simulated1 = B(
    #     position=position,
    #     dimension=dimension,
    #     polarization=polarization,
    #     susceptibility=susceptibility,
    #     points=positions1,
    #     model_path=model_path,
    # )

    field_simulated1 = mesh.getB(positions1)

    field_simulated2 = []
    for direction in positions2_rotated:
        #     field_simulated2_direction = B(
        #         position=position,
        #         dimension=dimension,
        #         polarization=polarization,
        #         susceptibility=susceptibility,
        #         points=direction,
        #         model_path=model_path,
        #     )
        #     field_simulated2.append(field_simulated2_direction)
        field_simulated2_direction = mesh.getB(direction)
        field_simulated2.append(field_simulated2_direction)

    field_simulated2 = np.array(field_simulated2)

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

    ###############################
    ### Plot Measurement Values ###
    ###############################
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
    ax1: matplotlib.axes.Axes
    ax2: matplotlib.axes.Axes
    ax3: matplotlib.axes.Axes

    ax1.plot(field_measured1[:, 0], label="Measurement")
    ax1.plot(field_simulated1[:, 0], label="Simulation")
    ax1.legend()
    ax1.set_xlabel("Point")
    ax1.set_title("X")
    ax1.set_ylabel("B Field")

    ax2.plot(field_measured1[:, 1], label="Measurement")
    ax2.plot(field_simulated1[:, 1], label="Measurement")
    ax2.legend()
    ax2.set_title("Y")
    ax2.set_xlabel("Point")

    ax3.plot(field_measured1[:, 2], label="Measurement")
    ax3.plot(field_simulated1[:, 2], label="Simulation")
    ax3.legend()
    ax3.set_title("Z")
    ax3.set_xlabel("Point")

    fig.suptitle("Field Values for NN Solution and Measurements")

    if save_dir is not None:
        plt.savefig(f"{save_dir}/measurements.png", format="png")
    else:
        plt.show()

    ############################
    ### Plot Relative Errors ###
    ############################
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
    ax1.plot(
        np.abs((field_measured1[:, 0] - field_simulated1[:, 0]) / field_measured1[:, 0])
        * 100
    )
    ax1.set_title("X")
    ax1.set_ylabel("Relative Error (%)")
    ax1.set_xlabel("Point")

    ax2.plot(
        np.abs((field_measured1[:, 1] - field_simulated1[:, 1]) / field_measured1[:, 0])
        * 100,
    )
    ax2.set_title("Y")
    ax2.set_xlabel("Point")

    ax3.plot(
        np.abs((field_measured1[:, 2] - field_simulated1[:, 2]) / field_measured1[:, 2])
        * 100,
    )
    ax3.set_title("Z")
    ax3.set_xlabel("Point")

    fig.suptitle("Relative Error of NN Solution")

    if save_dir is not None:
        plt.savefig(f"{save_dir}/relative-errors.png", format="png")
    else:
        plt.show()

    ###########################
    ### Plot Overall Errors ###
    ###########################

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 10))
    angle_err = angle_error(
        torch.from_numpy(field_measured1),
        torch.from_numpy(field_simulated1),
    )
    amp_err = relative_amplitude_error(
        torch.from_numpy(field_measured1),
        torch.from_numpy(field_simulated1),
        return_abs=True,
    )

    ax1.plot(angle_err)
    ax2.plot(amp_err)

    fig.suptitle("Errors of NN Solution")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/angle-amp-errors.png", format="png")
    else:
        plt.show()

    ############################
    ### Plot Absolute Errors ###
    ############################
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
    ax1.plot(
        np.abs((field_measured1[:, 0] - field_simulated1[:, 0])),
    )
    ax1.set_title("X")
    ax1.set_ylabel("Absolute Error")

    ax2.plot(
        np.abs((field_measured1[:, 1] - field_simulated1[:, 1])),
    )
    ax2.set_title("Y")
    ax2.set_xlabel("Point")

    ax3.plot(
        np.abs((field_measured1[:, 2] - field_simulated1[:, 2])),
    )
    ax3.set_title("Z")
    ax3.set_xlabel("Point")

    fig.suptitle("Absolute Error of NN Solution")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/absolute-errors.png", format="png")
    else:
        plt.show()

    ########################################
    ### Plot Some Other Measurements (?) ###
    ########################################
    for i in range(4):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
        ax1.plot(field_measured2_rotated[i, :, 0], label="Measured")
        ax1.plot(field_simulated2[i, :, 0], label="Simulated")
        ax1.set_title("X")
        ax1.legend()

        ax2.plot(field_measured2_rotated[i, :, 1], label="Measured")
        ax2.plot(field_simulated2[i, :, 1], label="Simulated y")
        ax2.set_title("Y")
        ax2.legend()

        ax3.plot(field_measured2_rotated[i, :, 2], label="Measured")
        ax3.plot(field_simulated2[i, :, 2], label="Simulated")
        ax3.set_title("Z")
        ax3.legend()
        if save_dir is not None:
            plt.savefig(f"{save_dir}/measurements_rotated_{i}.png", format="png")
        else:
            plt.show()

    ############################
    ### Plot Some Histograms ###
    ############################
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

    mean = np.mean(
        np.abs((field_measured1 - field_simulated1) / field_measured1) * 100,
        axis=0,
    )

    ax1.hist(
        np.abs((field_measured1[:, 0] - field_simulated1[:, 0]) / field_measured1[:, 0])
        * 100,
        bins=20,
    )
    ax1.set_title(f"X: Mean={round(mean[0], 2)}")
    ax1.set_ylabel("Count")
    ax2.set_xlabel("Relative Error (%)")

    ax2.hist(
        np.abs((field_measured1[:, 1] - field_simulated1[:, 1]) / field_measured1[:, 0])
        * 100,
        bins=20,
    )
    ax2.set_title(f"Y: Mean={round(mean[1], 2)}")
    ax2.set_xlabel("Relative Error (%)")

    ax3.hist(
        np.abs((field_measured1[:, 2] - field_simulated1[:, 2]) / field_measured1[:, 2])
        * 100,
        bins=20,
    )
    ax3.set_title(f"Z: Mean={round(mean[2], 3)}")
    ax3.set_xlabel("Relative Error (%)")

    fig.suptitle("Relative Error Frequency of NN Solution")

    if save_dir is not None:
        plt.savefig(f"{save_dir}/relative-error-histograms.png", format="png")
    else:
        plt.show()


def result_to_dict(result: OptimizeResult) -> Dict[str, float]:
    optimized_parameters = result.x
    parameter_names: List[str] = [
        "polarization_magnitude",
        "polarization_phi",
        "polarization_theta",
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
        "chi_x",
        "chi_y",
        "chi_z",
    ]

    results_dict = {}
    for k, v in zip(parameter_names, optimized_parameters):
        results_dict[k] = v

    return results_dict


def arr_to_dict(xs):
    parameter_names: List[str] = [
        "polarization_magnitude",
        "polarization_phi",
        "polarization_theta",
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
        "chi_x",
        "chi_y",
        "chi_z",
    ]

    results_dict = {}
    for k, v in zip(parameter_names, xs):
        results_dict[k] = v

    return results_dict

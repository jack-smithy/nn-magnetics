import magpylib as magpy
import numpy as np
from magpylib import Collection
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid
from nn_magnetics.utils.physics import demagnetizing_factor
import matplotlib.pyplot as plt
from nn_magnetics.graph.create_graphs import compute_edge_index_fast

eps = 1e-6


def generate_points_random(axis_coarseness, a, b, scale_factor=3.5):
    """
    Generate points (x,y,z) in the first octant (x,y,z>=0) that lie outside a box
    centered at the origin with side lengths (a/2, b/2, c/2), with a higher density near the box.

    Parameters:
      num_points: number of points to generate.
      a, b, c: parameters defining the box side lengths.
      scale_factor: controls how tightly points cluster around the box (smaller = more concentrated).

    Returns:
      A list of tuples (x, y, z) that are outside the box.
    """
    total_points = axis_coarseness**3
    points = []
    x_thresh, y_thresh, z_thresh = (
        a / 2,
        b / 2,
        1 / 2,
    )  # Box boundaries in the positive octant

    while len(points) < total_points:
        # Generate exponentially distributed random numbers, shifting them by the threshold
        x = np.random.exponential(scale_factor * x_thresh)
        y = np.random.exponential(scale_factor * y_thresh)
        z = np.random.exponential(scale_factor * z_thresh)

        outside_box = x > a / 2 or y > b / 2 or z > 1 / 2
        inside_domain = x <= 2.5 * a and y <= 2.5 * b and z <= 2.5
        if outside_box and inside_domain:
            points.append((x, y, z))

    return np.array(points)


def generate_points_grid(axis_coarseness, a, b):
    _grid = []
    for xx in np.linspace(eps, a * 2.5, axis_coarseness):
        for yy in np.linspace(eps, b * 2.5, axis_coarseness):
            for zz in np.linspace(eps, 2.5, axis_coarseness):
                if not (
                    0 <= xx <= a / 2 + eps
                    and 0 <= yy <= b / 2 + eps
                    and 0 <= zz <= 1 / 2 + eps
                ):
                    _grid.append([xx, yy, zz])

    return np.array(_grid)


def simulate_demag(
    a: float,
    b: float,
    chi: tuple,
    axis_coarseness: int = 26,
    points: str = "grid",
    display: bool = False,
    calculate_edge_index: bool = False,
) -> dict:
    print("=" * 100)
    assert len(chi) == 3

    print("Creating measurement grid")
    if points == "grid":
        grid = generate_points_grid(axis_coarseness=axis_coarseness, a=a, b=b)
    elif points == "random":
        grid = generate_points_random(axis_coarseness=axis_coarseness, a=a, b=b)
    else:
        raise ValueError(f"{points} is not a strategy")

    edge_index = None
    if calculate_edge_index:
        edge_index = compute_edge_index_fast(grid)

    if display:
        display_points(grid, a, b, 1)

    ######### calculate demag field ###############
    magnet = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(a, b, 1))
    magnet.susceptibility = chi  # type: ignore

    print("Meshing magnet and applying demag effects")
    cm = mesh_Cuboid(magnet, target_elems=int(100))
    cm_demag: Collection = apply_demag(cm, inplace=False)  # type: ignore
    grid_field = cm_demag.getB(grid)

    print("Calculating reduced field")
    cell_pos_all = np.array([cell.position for cell in cm])

    cell_field = cm_demag.getM(cell_pos_all)
    mean_magnetization = np.mean(cell_field, axis=0)
    reduced_polarization = magpy.mu_0 * mean_magnetization

    magnet_reduced = magpy.magnet.Cuboid(
        polarization=reduced_polarization,
        dimension=(a, b, 1),
    )
    grid_field_reduced = magnet_reduced.getB(grid)

    # grid_field_reduced_scaled = grid_field_ana * Pz_reduced

    data = {
        "a": a,
        "b": b,
        "chi_x": chi[0],
        "chi_y": chi[1],
        "chi_z": chi[2],
        "grid": grid,
        "grid_field": grid_field,
        "grid_field_reduced": grid_field_reduced,
        # "grid_field_reduced_calc": grid_field_reduced_calc,
        # "reduced_polarization": (0, 0, Pz_reduced),
        "reduced_polarization_calc": reduced_polarization,
        # "grid_field_reduced_scaled": grid_field_reduced_scaled,
        "edge_index": edge_index,
    }

    return data


def display_points(filtered_points, a, b, c):
    # Plot the filtered points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the remaining grid points
    ax.scatter(
        filtered_points[:, 0],
        filtered_points[:, 1],
        filtered_points[:, 2],
        color="b",
        s=10,  # type: ignore
        label="Grid Points",
    )

    # Plot the excluded cuboid (wireframe)
    cuboid_x = np.array([0, a / 2, a / 2, 0, 0, a / 2, a / 2, 0])
    cuboid_y = np.array([0, 0, b / 2, b / 2, 0, 0, b / 2, b / 2])
    cuboid_z = np.array([0, 0, 0, 0, c / 2, c / 2, c / 2, c / 2])

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for edge in edges:
        ax.plot(
            [cuboid_x[edge[0]], cuboid_x[edge[1]]],
            [cuboid_y[edge[0]], cuboid_y[edge[1]]],
            [cuboid_z[edge[0]], cuboid_z[edge[1]]],
            color="r",
        )

    ax.set_xlim((0, 2.5 * a))
    ax.set_ylim((0, 2.5 * b))
    ax.set_zlim((0, 2.5 * c))  # type: ignore
    # Labels and view settings
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")  # type: ignore
    ax.set_title("Non-Uniform 3x3x3 Grid with Cuboid Excluded")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    simulate_demag(
        1,
        1,
        (0.1, 0.1, 0.1),
        display=True,
        axis_coarseness=8,
        calculate_edge_index=True,
    )

import magpylib as magpy
import numpy as np
from magpylib import Collection
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid

eps = 1e-6


def simulate_demag(a: float, b: float, chi: tuple, axis_coarseness: int = 26) -> dict:
    print("=" * 100)
    assert len(chi) == 3

    magnet = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(a, b, 1))
    # magnet.susceptibility = (chi_perp, chi_perp, chi_long)  # type: ignore
    magnet.susceptibility = chi  # type: ignore

    print("Meshing magnet and applying demag effects")
    cm = mesh_Cuboid(magnet, target_elems=int(100))
    cm_demag: Collection = apply_demag(cm, inplace=False)  # type: ignore

    cell_pos_all = np.array([cell.position for cell in cm])

    print("Creating measurement grid")
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

    grid = np.array(_grid)

    print("Calculating analytical B-field")
    grid_field_ana = cm.getB(grid)

    print("Calculating demag B-field")
    grid_field = cm_demag.getB(grid)

    print("Calculating reduced field")
    cell_field = cm_demag.getM(cell_pos_all)
    mean_magnetization = np.mean(cell_field, axis=0)
    demag_factor = magpy.mu_0 * mean_magnetization

    magnet_reduced = magpy.magnet.Cuboid(
        polarization=demag_factor,
        dimension=(a, b, 1),
    )
    grid_field_reduced = magnet_reduced.getB(grid)

    return {
        "a": a,
        "b": b,
        "chi_x": chi[0],
        "chi_y": chi[1],
        "chi_z": chi[2],
        "grid": grid,
        "grid_field": grid_field,
        "grid_field_ana": grid_field_ana,
        "grid_field_reduced": grid_field_reduced,
        "demagnetization_factor": demag_factor[2],
    }

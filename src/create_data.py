import magpylib as magpy
import numpy as np
from magpylib import Collection
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid

eps = 1e-6


def simulate_demag(a: float, b: float, chi: float) -> dict:
    print("=" * 100)

    magnet = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(a, b, 1))
    # magnet.susceptibility = (chi_perp, chi_perp, chi_long)  # type: ignore
    magnet.susceptibility = chi  # type: ignore

    print("Meshing magnet and applying demag effects")
    cm = mesh_Cuboid(magnet, target_elems=int(100))
    cm_demag: Collection = apply_demag(cm, inplace=False)  # type: ignore

    cell_pos_all = np.array([cell.position for cell in cm])

    print("Creating measurement grid")
    _grid = []
    for xx in np.linspace(0, a * 2.5, 26):
        for yy in np.linspace(0, b * 2.5, 26):
            for zz in np.linspace(0, 2.5, 26):
                if not (0 <= xx <= a / 2 and 0 <= yy <= b / 2 and 0 <= zz <= 1 / 2):
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
        # chi_perp=chi_perp,
        # chi_long=chi_perp,
        "chi": chi,
        # cell_pos=cell_pos,
        # cell_field=cell_field,
        # cell_H=cell_H,
        "grid": grid,
        "grid_field": grid_field,
        "grid_field_ana": grid_field_ana,
        "grid_field_reduced": grid_field_reduced,
        "demagnetization_factor": demag_factor,
        # magnetization_reduced=magnetization_reduced,
        # grid_field_reduced=grid_field_reduced,
        # cell_field_reduced=cell_field_reduced,
        # cell_phi=cell_phi,
        # cell_phi_reduced=cell_phi_reduced,
        # grid_phi=grid_phi,
        # grid_phi_reduced=grid_phi_reduced,
    }


def simulate_task(index):
    # chi_perp = np.random.uniform(0.0, 1.0)
    # chi_long = np.random.uniform(0.0, 1.0)
    # chi = (chi_perp, chi_perp, chi_long)

    a = np.random.uniform(low=0.3, high=3.0)
    b = np.random.uniform(low=0.3, high=3.0)
    chi = np.random.uniform(0, 1)

    if a > b:
        a, b = b, a

    print(f"Starting simuation: {index}")
    data = simulate_demag(a, b, chi)
    path = f"data/isotropic_chi_v2/data_{index}.npz"
    np.savez(path, **data)


if __name__ == "__main__":
    for idx in range(101, 501):
        simulate_task(idx)

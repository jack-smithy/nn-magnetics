from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid
import magpylib as magpy
import numpy as np

eps = 1e-6


def simulate_demag(a, b, chi, index):
    print("=" * 100)
    print(f"Starting simulation {index}")

    magnet = magpy.magnet.Cuboid(magnetization=(0, 0, 1), dimension=(a, b, 1))
    volume = a * b
    magnet.susceptibility = chi  # type: ignore

    print("Meshing magnet and applying demag effects")
    cm = mesh_Cuboid(magnet, target_elems=int(100 * volume))
    apply_demag(cm, inplace=True)

    # cell_pos_all = np.array([cell.position for cell in cm])

    # cell_pos = []
    # for cell in cm:
    #     pos = cell.position
    #     if pos[0] >= -eps and pos[1] >= -eps and pos[2] >= -eps:
    #         cell_pos.append(pos)

    # cell_pos = np.array(cell_pos)

    print("Creating measurement grid")
    x_grid = np.linspace(0, a * 2.5, 26)
    y_grid = np.linspace(0, b * 2.5, 26)
    z_grid = np.linspace(0, 2.5, 26)

    grid = []
    for x in x_grid:
        for y in y_grid:
            for z in z_grid:
                if not (x <= a / 2 + eps and y <= b / 2 + eps and z <= 0.5 + eps):
                    grid.append([x, y, z])

    grid = np.array(grid)

    # cell_field_all = cm.getB(cell_pos_all)
    # cell_H_all = cm.getH(cell_pos_all)
    # cell_magnetization_all = cell_field_all - cell_H_all * 4 * np.pi / 10
    # magnetization_mean = np.mean(cell_magnetization_all, axis=0)

    # cell_field = cm.getB(cell_pos)
    # cell_H = cm.getH(cell_pos)
    # cell_magnetization = cell_field - cell_H * 4 * np.pi / 10
    # magnetization_reduced = cell_magnetization - magnetization_mean

    print("Calculating demag B-field")
    grid_field = cm.getB(grid)
    # magnet_reduced = magpy.magnet.Cuboid(
    #     magnetization=magnetization_mean, dimension=(a, b, 1)
    # )
    # grid_field_magnet_reduced = magnet_reduced.getB(grid)
    # grid_field_reduced = grid_field - grid_field_magnet_reduced

    print("Calculating analytical B-field")
    magnet_ana = magpy.magnet.Cuboid(magnetization=(0, 0, 1), dimension=(a, b, 1))
    grid_field_ana = magnet_ana.getB(grid)

    # cell_field_magnet_reduced = magnet_reduced.getB(cell_pos)
    # cell_field_reduced = cell_field - cell_field_magnet_reduced

    # # for scalar potential
    # cell_phi = scalar_potential.getPhi_collection(cm, cell_pos)
    # cell_phi_magnet_reduced = scalar_potential.getPhi(
    #     cell_pos, np.array((a, b, 1)), magnetization_mean, np.array((0, 0, 0))
    # )
    # cell_phi_reduced = cell_phi - cell_phi_magnet_reduced

    # grid_phi = scalar_potential.getPhi_collection(cm, grid)
    # grid_phi_magnet_reduced = scalar_potential.getPhi(
    #     grid, np.array((a, b, 1)), magnetization_mean, np.array((0, 0, 0))
    # )
    # grid_phi_reduced = grid_phi - grid_phi_magnet_reduced

    print("Saving data")
    np.savez(
        f"data/isotropic_chi/data_{index}",
        a=a,
        b=b,
        # chi_perp=chi[0],
        # chi_long=chi[-1],
        chi=chi,
        # cell_pos=cell_pos,
        # cell_field=cell_field,
        # cell_H=cell_H,
        grid=grid,
        grid_field=grid_field,
        grid_field_ana=grid_field_ana,
        # demagnetization_factor=magnetization_mean[2],
        # magnetization_reduced=magnetization_reduced,
        # grid_field_reduced=grid_field_reduced,
        # cell_field_reduced=cell_field_reduced,
        # cell_phi=cell_phi,
        # cell_phi_reduced=cell_phi_reduced,
        # grid_phi=grid_phi,
        # grid_phi_reduced=grid_phi_reduced,
    )


if __name__ == "__main__":
    for index in range(0, 100):
        # chi_perp = np.random.uniform(0.1, 0.9)
        # chi_long = np.random.uniform(0.1, 0.9)
        # chi = (chi_perp, chi_perp, chi_long)
        chi = np.random.uniform(low=0, high=1)
        a = np.random.uniform(low=0.3, high=3.0)
        b = np.random.uniform(low=0.3, high=3.0)

        if a > b:
            a, b = b, a

        simulate_demag(a, b, chi, index)

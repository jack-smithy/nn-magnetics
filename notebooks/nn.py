import magpylib as magpy
import pyvista as pv
import numpy as np

# Create a magnet with Magpylib
magnet = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.01, 0.01, 0.01))

# Create a 3D grid with Pyvista
grid = pv.ImageData(
    dimensions=(41, 41, 41),
    spacing=(0.002, 0.002, 0.002),
    origin=(-0.04, -0.04, -0.04),
)

pl = pv.Plotter()

# Add magnet to scene - streamlines units are assumed to be meters
magpy.show(magnet, canvas=pl, units_length="m", backend="pyvista")
offset = 0.005

pl.remove_legend(pl)

start_points = np.array(
    [
        [0.01, 0.01, 0.01],
        [-0.01, 0.01, 0.01],
        [0.01, -0.01, 0.01],
        [-0.01, -0.01, 0.01],
        [0.01 + offset, 0.01 + offset, -0.01 - offset],
        [-0.01 - offset, 0.01 + offset, -0.01 - offset],
        [0.01 + offset, -0.01 - offset, -0.01 - offset],
        [-0.01 - offset, -0.01 - offset, -0.01 - offset],
    ]
)

# Corresponding direction vectors
directions = 0.5 * np.array(
    [
        # (+, +, +)
        [0.01, 0.01, 0.01],
        # (-, +, +)
        [-0.01, 0.01, 0.01],
        # (+, -, +)
        [0.01, -0.01, 0.01],
        # (-, -, +)
        [-0.01, -0.01, 0.01],
        # (+, +, -)
        [-0.01, -0.01, 0.01],
        # (-, +, -)
        [0.01, -0.01, 0.01],
        # (+, -, -)
        [-0.01, 0.01, 0.01],
        # (-, -, -)
        [0.01, 0.01, 0.01],
    ]
)

# Add arrows to the plot
pl.add_arrows(
    start_points,
    directions,
    mag=1,  # scale factor for visualization
    color="red",
)

# Add transparent coordinate planes
plane_opacity = 0.25
plane_resolution = (10, 10)
plane_size = 0.03  # Half-width of the plane in each direction
colours = ["lightblue", "lightgreen", "lightcoral", "black"]

# XY plane at z=0
xy_plane = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    i_size=plane_size * 2,
    j_size=plane_size * 2,
    i_resolution=plane_resolution[0],
    j_resolution=plane_resolution[1],
)
pl.add_mesh(xy_plane, color=colours[0], opacity=plane_opacity, show_edges=False)

# YZ plane at x=0
yz_plane = pv.Plane(
    center=(0, 0, 0),
    direction=(1, 0, 0),
    i_size=plane_size * 2,
    j_size=plane_size * 2,
    i_resolution=plane_resolution[0],
    j_resolution=plane_resolution[1],
)
pl.add_mesh(yz_plane, color=colours[1], opacity=plane_opacity, show_edges=False)

# XZ plane at y=0
xz_plane = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 1, 0),
    i_size=plane_size * 2,
    j_size=plane_size * 2,
    i_resolution=plane_resolution[0],
    j_resolution=plane_resolution[1],
)
pl.add_mesh(xz_plane, color=colours[2], opacity=plane_opacity, show_edges=False)

axis_length = plane_size

r = 0.01 + offset / 2
width = 4
point_size = 12
point_opacity = 1

# X axis (red)
pl.add_lines(
    np.array([[r, -r, r], [r, r, r]]),
    color=colours[2],
    width=width,
)

pl.add_points(
    np.array([r, 0, r]),
    render_points_as_spheres=True,
    point_size=point_size,
    color=colours[3],
    opacity=point_opacity,
)


pl.add_lines(
    np.array([[r, -r, r], [-r, -r, r]]),
    color=colours[1],
    width=width,
)

pl.add_points(
    np.array([0, -r, r]),
    render_points_as_spheres=True,
    point_size=point_size,
    color=colours[3],
    opacity=point_opacity,
)

pl.add_lines(
    np.array([[r, -r, r], [r, -r, -r]]),
    color=colours[0],
    width=width,
)

pl.add_points(
    np.array([r, -r, 0]),
    render_points_as_spheres=True,
    point_size=point_size,
    color=colours[3],
    opacity=point_opacity,
)

pl.add_lines(
    np.array([[0, 0, -0.03], [0, 0, 0.03]]),
    color="black",
    width=width,
)

pl.add_lines(
    np.array([[0, -0.03, 0], [0, 0.03, 0]]),
    color="black",
    width=width,
)

pl.add_lines(
    np.array([[-0.03, 0, 0], [0.03, 0, 0]]),
    color="black",
    width=width,
)

# Show scene

# # Prepare and show scene
pl.camera.position = (0.1, 0.02, 0.02)
pl.show()
pl.screenshot("B-field_symmetry.png", return_img=False)

import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# Create figure with 2D and 3D canvas
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)

# Model of magnetization tool
tool1 = magpy.magnet.Cuboid(
    dimension=(5, 3, 3), polarization=(1, 0, 0), position=(9, 0, 0)
).rotate_from_angax(50, "z", anchor=0)
tool2 = tool1.copy(polarization=(-1, 0, 0)).rotate_from_angax(-100, "z", 0)
tool3 = tool1.copy().rotate_from_angax(180, "z", 0)
tool4 = tool2.copy().rotate_from_angax(180, "z", 0)
tool = magpy.Collection(tool1, tool2, tool3, tool4)

# Model of Quadrupole Cylinder
cyl = magpy.magnet.Cuboid(
    dimension=(1, 1, 1),
    polarization=(0, 0, 0),
    style_magnetization_show=False,
)

# Plot 3D model on ax1
magpy.show(
    cyl, tool, canvas=ax1, style_legend_show=False, style_magnetization_mode="color"
)
ax1.view_init(90, -90)

# Compute and plot tool-field on grid
grid = np.mgrid[-6:6:50j, -6:6:50j, 0:0:1j].T[0]
X, Y, _ = np.moveaxis(grid, 2, 0)

B = tool.getB(grid)
Bx, By, Bz = np.moveaxis(B, 2, 0)

ax2.streamplot(
    X,
    Y,
    Bx,
    By,
    color=np.linalg.norm(B, axis=2),
    cmap="autumn",
    density=1.5,
    linewidth=1,
)

# Outline magnet boundary
ts = np.linspace(0, 2 * np.pi, 200)
ax2.plot(2 * np.sin(ts), 2 * np.cos(ts), color="k", lw=2)
ax2.plot(5 * np.sin(ts), 5 * np.cos(ts), color="k", lw=2)

# Plot styling
ax2.set(
    title="B-field in xy-plane",
    xlabel="x-position",
    ylabel="y-position",
    aspect=1,
)

plt.tight_layout()
plt.show()

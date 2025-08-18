import os.path

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from pyvista import cartesian_to_spherical, spherical_to_cartesian

# changing all fonts
plt.rcParams.update(
    {
        # "font.family": "serif",
        # "font.serif": ["Times New Roman"],
        "font.size": 16,
        # "mathtext.fontset": "custom",
        # "mathtext.rm": "Times New Roman",
        # "mathtext.it": "Times New Roman:italic",
        # "mathtext.bf": "Times New Roman:bold",
    }
)

file_path = "notebooks/data.npz"
print(os.path.exists(file_path))

x_lim = [0.5, 1.0]
z_lim = [0.48, 0.7]

# additional lineplots
x_start = 0.5
x_end = 1.0
z_value = 0.59

# Create a Matplotlib figure
fig = plt.figure()
subfigs = fig.subfigures(2, 1)

ax1, ax2 = subfigs[0].subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
ax3, ax4 = subfigs[1].subplots(2, 1, sharex=True)
# gs = fig.add_gridspec(3,2)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[1, :])
# ax4 = fig.add_subplot(gs[2, :])


# Create an observer grid in the xz-symmetry plane
ts = np.linspace(-1.5, 1.5, 40)
grid = np.array([[(x, 0, z) for x in ts] for z in ts])
X, _, Z = np.moveaxis(grid, 2, 0)

# Compute the B-field of a cube magnet on the grid
cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
B = cube.getB(grid)
Bx, _, Bz = np.moveaxis(B, 2, 0)

# Display the B-field with streamplot using log10-scaled
# color function and linewidth
splt = ax1.streamplot(
    X,
    Z,
    Bx,
    Bz,
    density=1.5,
    color="y",
)


# # Outline magnet boundary
# ax.plot(
#     [0.5, 0.5, -0.5, -0.5, 0.5],
#     [0.5, -0.5, -0.5, 0.5, 0.5],
#     "k--",
#     lw=2,
# )

x_lim = [0.5, 1.0]
z_lim = [0.48, 0.7]

# Outline zoom
ax1.plot(
    [x_lim[0], x_lim[1], x_lim[1], x_lim[0], x_lim[0]],
    [z_lim[0], z_lim[0], z_lim[1], z_lim[1], z_lim[0]],
    lw=3,
    color="darkviolet",
)

# Figure styling
ax1.set(
    xlabel="X (a.u.)",
    ylabel="Z (a.u.)",
)


my_gradient = LinearSegmentedColormap.from_list(
    "my_gradient", ((0, "g"), (1 / 3, "w"), (2 / 3, "w"), (1, "r"))
)


xs_fine = np.linspace(-0.5, 0.5, 1000)
zs_fine = np.linspace(-0.5, 0.5, 1000)
values = np.array([[z for x in xs_fine] for z in zs_fine])

ax1.contourf(xs_fine, zs_fine, values, levels=900, cmap=my_gradient, zorder=10)

ax1.set_aspect("equal", adjustable="box")


if os.path.exists(file_path):
    data = np.load(file_path)
    X = data["X"]
    Z = data["Z"]
    Bx = data["Bx"]
    Bz = data["Bz"]
    norm_B = data["norm_B"]
    Bx_demag = data["Bx_demag"]
    Bz_demag = data["Bz_demag"]
    norm_B_demag = data["norm_B_demag"]
    X_fine = data["X_fine"]
    Z_fine = data["Z_fine"]
    Bx_fine = data["Bx_fine"]
    Bz_fine = data["Bz_fine"]
    Bx_demag_fine = data["Bx_demag_fine"]
    Bz_demag_fine = data["Bz_demag_fine"]
    grid_lineplot = data["grid_lineplot"]
    B_lineplot = data["B_lineplot"]
    B_lineplot_demag = data["B_lineplot_demag"]

else:
    polarization = 1
    magnet = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, polarization),
    )

    susceptibility = 10
    magnet_demag = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, polarization * (1 + 1 / 3 * susceptibility)),
    )
    magnet_demag = mesh_Cuboid(magnet_demag, (10, 10, 10))
    magnet_demag = apply_demag(magnet_demag, susceptibility=susceptibility)

    # Create an observer grid in the xz-symmetry plane
    xs = np.linspace(*x_lim, 10)
    zs = np.linspace(*z_lim, 10)
    grid = np.array([[(x, 0, z) for x in xs] for z in zs])
    X, _, Z = np.moveaxis(grid, 2, 0)

    # Compute the B-field of a cube magnet on the grid
    B = magnet.getB(grid)
    B_demag = magnet_demag.getB(grid)

    Bx, _, Bz = np.moveaxis(B, 2, 0)
    norm_B = np.linalg.norm(B, axis=2)

    Bx_demag, _, Bz_demag = np.moveaxis(B_demag, 2, 0)
    norm_B_demag = np.linalg.norm(B_demag, axis=2)

    # Display the B-field with streamplot using log10-scaled
    # color function and linewidth
    # splt = ax.streamplot(X, Z, Bx, Bz,
    #     density=1.5,
    #     color=log10_norm_B,
    #     linewidth=log10_norm_B,
    #     cmap="autumn",
    # )

    # fine grid for cotourf
    # Create an observer grid in the xz-symmetry plane
    xs_fine = np.linspace(*x_lim, 250)
    zs_fine = np.linspace(*z_lim, 250)
    grid_fine = np.array([[(x, 0, z) for x in xs_fine] for z in zs_fine])
    X_fine, _, Z_fine = np.moveaxis(grid_fine, 2, 0)

    B_fine = magnet.getB(grid_fine)
    Bx_fine, _, Bz_fine = np.moveaxis(B_fine, 2, 0)
    B_demag_fine = magnet_demag.getB(grid_fine)
    Bx_demag_fine, _, Bz_demag_fine = np.moveaxis(B_demag_fine, 2, 0)

    ############
    # additional lineplots
    grid_lineplot = np.linspace((x_start, 0, z_value), (x_end, 0, z_value), 100)

    B_lineplot = magnet.getB(grid_lineplot)
    B_lineplot_demag = magnet_demag.getB(grid_lineplot)

    np.savez(
        file_path,
        X=X,
        Z=Z,
        Bx=Bx,
        Bz=Bz,
        norm_B=norm_B,
        Bx_demag=Bx_demag,
        Bz_demag=Bz_demag,
        norm_B_demag=norm_B_demag,
        X_fine=X_fine,
        Z_fine=Z_fine,
        Bx_fine=Bx_fine,
        Bz_fine=Bz_fine,
        Bx_demag_fine=Bx_demag_fine,
        Bz_demag_fine=Bz_demag_fine,
        grid_lineplot=grid_lineplot,
        B_lineplot=B_lineplot,
        B_lineplot_demag=B_lineplot_demag,
    )


cmap1 = plt.cm.jet

cmap = colors.ListedColormap(["violet", "lightgrey", "lightgreen"])
bounds = [-1, 0, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)

ax2.contourf(
    X_fine, Z_fine, np.sign(Bz_fine) + np.sign(Bz_demag_fine), cmap=cmap, alpha=0.5
)
ax2.contour(
    X_fine,
    Z_fine,
    np.sign(Bz_fine) + np.sign(Bz_demag_fine),
    levels=[-1, 1],
    linestyles="-",
    linewidths=2,
    colors="k",
)


ax2.quiver(
    X[Bz >= 0],
    Z[Bz >= 0],
    Bx[Bz >= 0] / norm_B[Bz >= 0],
    Bz[Bz >= 0] / norm_B[Bz >= 0],
    color="b",
    linewidth=1,
    scale=1e1,
)
ax2.quiver(
    X[Bz < 0],
    Z[Bz < 0],
    Bx[Bz < 0] / norm_B[Bz < 0],
    Bz[Bz < 0] / norm_B[Bz < 0],
    color="b",
    linewidth=1,
    scale=1e1,
)


# Display the B-field with streamplot using log10-scaled
# color function and linewidth
ax2.quiver(
    X[Bz_demag >= 0],
    Z[Bz_demag >= 0],
    Bx_demag[Bz_demag >= 0] / norm_B_demag[Bz_demag >= 0],
    Bz_demag[Bz_demag >= 0] / norm_B_demag[Bz_demag >= 0],
    color="r",
    linewidth=1,
    scale=1e1,
)
ax2.quiver(
    X[Bz_demag < 0],
    Z[Bz_demag < 0],
    Bx_demag[Bz_demag < 0] / norm_B_demag[Bz_demag < 0],
    Bz_demag[Bz_demag < 0] / norm_B_demag[Bz_demag < 0],
    color="r",
    linewidth=1,
    scale=1e1,
)

for i in range(len(X)):
    ax2.plot((x_lim[0], x_lim[1]), (Z[i], Z[i]), color="k", linewidth=0.7)
# ax.quiver(X, Z, 1., 0., color='k', linewidth=0.001, scale=1e1, headwidth=1, headlength=0)


# Outline magnet boundary
ax2.plot(
    [0.5, 0.5, x_lim[0]],
    [z_lim[0], 0.5, 0.5],
    "k--",
    lw=2,
)

# Figure styling
ax2.set(
    xlabel="X (a.u.)",
    ylabel="Z (a.u.)",
)

ax2.set_aspect("equal", adjustable="box")

ax2.spines["top"].set_color("darkviolet")
ax2.spines["bottom"].set_color("darkviolet")
ax2.spines["left"].set_color("darkviolet")
ax2.spines["right"].set_color("darkviolet")
ax2.spines["top"].set_linewidth(5)
ax2.spines["bottom"].set_linewidth(5)
ax2.spines["left"].set_linewidth(5)
ax2.spines["right"].set_linewidth(5)


# additional lineplots
ax2.plot(
    (x_start, x_end),
    (z_value, z_value),
    color="dodgerblue",
    linestyle="--",
    linewidth=3,
)


B_lineplot_spherical = cartesian_to_spherical(
    B_lineplot[:, 0], B_lineplot[:, 1], B_lineplot[:, 2]
)
B_lineplot_demag_spherical = cartesian_to_spherical(
    B_lineplot_demag[:, 0], B_lineplot_demag[:, 1], B_lineplot_demag[:, 2]
)
B_lineplot_spherical = np.array(B_lineplot_spherical).T
B_lineplot_demag_spherical = np.array(B_lineplot_demag_spherical).T

print(B_lineplot_spherical)
print(B_lineplot_demag_spherical)

factor_x = B_lineplot_demag[:, 0] / B_lineplot[:, 0]
factor_z = B_lineplot_demag[:, 2] / B_lineplot[:, 2]

factor_amp = B_lineplot_demag_spherical[:, 0] / B_lineplot_spherical[:, 0]
difference_polar_angle = B_lineplot_demag_spherical[:, 1] - B_lineplot_spherical[:, 1]


######
# for interpolation of roots
def interpolate(B_lineplot, grid_lineplot):
    pos_index = np.where(B_lineplot[:, 2] >= 0)[0][-1]
    neg_index = np.where(B_lineplot[:, 2] <= 0)[0][0]
    x_vals = np.array((B_lineplot[neg_index, 2], B_lineplot[pos_index, 2]))
    y_vals = np.array((grid_lineplot[neg_index, 0], grid_lineplot[pos_index, 0]))
    return np.interp(0, x_vals, y_vals)


zero1 = interpolate(B_lineplot, grid_lineplot)
zero2 = interpolate(B_lineplot_demag, grid_lineplot)

print(zero1, zero2)

ax3.scatter((zero1, zero2), (0, 0), marker="*", color="C1", s=100)
ax4.scatter((zero1, zero2), (1, 0), marker="*", color="C1", s=100)


lns1 = ax3.plot(grid_lineplot[:, 0], B_lineplot[:, 0], color="C0", label=r"$B_x$")
ax3.plot(grid_lineplot[:, 0], B_lineplot_demag[:, 0], color="C0", linestyle="--")
lns2 = ax3.plot(grid_lineplot[:, 0], B_lineplot[:, 2], color="C1", label=r"$B_z$")
ax3.plot(grid_lineplot[:, 0], B_lineplot_demag[:, 2], color="C1", linestyle="--")
lns3 = ax3.plot(
    grid_lineplot[:, 0], B_lineplot_spherical[:, 0], color="C2", label=r"$B_r$"
)
ax3.plot(
    grid_lineplot[:, 0], B_lineplot_demag_spherical[:, 0], color="C2", linestyle="--"
)
ax5 = ax3.twinx()
lns4 = ax5.plot(
    grid_lineplot[:, 0], B_lineplot_spherical[:, 1], color="C3", label=r"$B_\theta$"
)
ax5.plot(
    grid_lineplot[:, 0], B_lineplot_demag_spherical[:, 1], color="C3", linestyle="--"
)
ax3.plot(
    (grid_lineplot[0, 0], grid_lineplot[-1, 0]), (0.0, 0.0), color="k", linestyle=":"
)
lns7 = ax4.plot(grid_lineplot[:, 0], factor_x, color="C0", label=r"$f_x$")
lns8 = ax4.plot(grid_lineplot[:, 0], factor_z, color="C1", label=r"$f_z$")
lns9 = ax4.plot(grid_lineplot[:, 0], factor_amp, color="C2", label=r"$f_r$")
ax6 = ax4.twinx()
lns10 = ax6.plot(
    grid_lineplot[:, 0], difference_polar_angle, color="C3", label=r"$f_\theta$"
)
ax4.plot(
    (grid_lineplot[0, 0], grid_lineplot[-1, 0]), (1.0, 1.0), color="k", linestyle=":"
)
ax3.grid()
ax4.grid()
ax4.set_ylim((-1, 3))
# ax3.set_xticklabels([])
ax3.set_ylabel("field (a.u.)")
ax5.set_ylabel("angle (rad)")
ax4.set_ylabel("corr. factor")
ax6.set_ylabel("angle corr. (rad)")
ax4.set_xlabel("X (a.u.)")

lns_all = lns1 + lns2 + lns3 + lns4
labs = [l.get_label() for l in lns_all]
ax3.legend(lns_all, labs, loc="right")

lns5 = ax5.plot([], [], color="k", label=r"$\mathbf{B}^\text{hom}$")
lns6 = ax5.plot([], [], color="k", linestyle="--", label=r"$\mathbf{B}'$")
lns_all2 = lns5 + lns6
labs2 = [l.get_label() for l in lns_all2]
ax5.legend(lns_all2, labs2, loc="upper center")

lns_all3 = lns7 + lns8 + lns9 + lns10
labs3 = [l.get_label() for l in lns_all3]
ax4.legend(lns_all3, labs3)

# backgroundcolors
ax3.axvspan(x_lim[0], zero1, facecolor=cmap.colors[2], alpha=0.5, zorder=-10)
ax3.axvspan(zero1, zero2, facecolor=cmap.colors[1], alpha=0.5, zorder=-10)
ax3.axvspan(zero2, x_lim[1], facecolor=cmap.colors[0], alpha=0.5, zorder=-10)
ax4.axvspan(x_lim[0], zero1, facecolor=cmap.colors[2], alpha=0.5, zorder=-10)
ax4.axvspan(zero1, zero2, facecolor=cmap.colors[1], alpha=0.5, zorder=-10)
ax4.axvspan(zero2, x_lim[1], facecolor=cmap.colors[0], alpha=0.5, zorder=-10)


ax1.set_title("(a)")
ax2.set_title("(b)")
ax3.set_title("(c)")
ax4.set_title("(d)")
# plt.tight_layout()
plt.show()

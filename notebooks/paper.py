import matplotlib.pyplot as plt
import numpy as np
from magpylib import magnet
from magpylib_material_response import demag, meshing

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 16,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
)

labels = ["Component", "Euler", "Quaternion"]
colors_ = ["#784c99", "#6780be", "#e93ea5"]
markers = ["v", "o", "*"]


def all_outside(points, dims):
    a, b, c = dims
    for p in points:
        x, y, z = p
        if not (x <= a or y <= b or z <= c):
            print(p)
            return False

    return True


def plotB(xx, B, Bd):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey=True)

    ax[0].plot(xx, B[:, 0], color=colors_[1])
    ax[0].plot(xx, Bd[:, 0], color=colors_[2])
    ax[0].set_ylabel(r"$\mathbf{B}_x$")

    ax[1].plot(xx, B[:, 1], color=colors_[1], label="Analytical Field")
    ax[1].plot(xx, Bd[:, 1], color=colors_[2], label="True Field")
    ax[1].set_ylabel(r"$\mathbf{B}_y$")

    for a in ax:
        a.set_xlabel("X")
        a.set_xlim((-1, 1))
        a.grid(alpha=0.5)
        a.tick_params(
            axis="both",  # Apply to both x and y axis
            which="both",  # Apply to both major and minor ticks
            direction="in",  # Tick direction 'in' for inside the plot
            top=True,  # Show ticks on top
            right=True,  # Show ticks on right
            length=4,
        )

    plt.tight_layout()

    return fig, ax


def main():
    cuboid = magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 1, 0))
    xx = np.linspace(-1, 1, 101)

    observers = np.array([(x, 0.6, 0) for x in xx])
    all_outside(points=observers, dims=[1, 1, 1])

    B = cuboid.getB(observers)

    mesh = meshing.mesh_Cuboid(cuboid=cuboid, target_elems=100)
    demag.apply_demag(mesh, 1, True)
    Bd = mesh.getB(observers)

    fig, ax = plotB(xx, B, Bd)
    plt.savefig("B.pdf", format="pdf")


if __name__ == "__main__":
    main()

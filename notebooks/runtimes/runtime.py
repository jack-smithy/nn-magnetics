from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from magpylib import magnet
from magpylib_material_response import demag, meshing
from tqdm import tqdm as bar
import json

from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import SphericalCorrectionNetwork
from nn_magnetics.utils.physics import Bfield_homogeneous

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

DATA_PATH = "./data/3dof_chi_v3/small/validation"
WEIGHTS_PATH = "/Users/jacksmith/Documents/work/nn-magnetics/results/3dof_chi_spherical/2025-05-13 16:36:35.129285"
N_REPEATS = 1
BATCH_SIZES = [
    1,
    2,
    4,
    # 8,
    # 16,
    # 32,
    # 64,
    # 128,
    # 256,
    # 512,
    # 1024,
    # 2048,
    # 4096,
    # 16834,
    # 65536,
    # 262144,
]


def time_analytical(x, n_samples, n_repeats, dimensions):
    observers = x[:n_samples, 5:]
    polarizations = torch.tensor([0, 0, 1]).expand((observers.shape[0], -1))
    dimensions = torch.tensor(dimensions).expand((observers.shape[0], -1))

    total_time = 0
    for _ in range(n_repeats):
        start = perf_counter()
        _ = Bfield_homogeneous(
            observers=observers,
            dimensions=dimensions,
            polarizations=polarizations,
        )
        end = perf_counter()

        total_time += end - start

    return total_time / n_repeats


def time_demag(x, mesh, n_samples, n_repeats, susceptibility):
    observers = x[:n_samples, 5:].numpy()

    total_time = 0
    for _ in range(n_repeats):
        start = perf_counter()
        demag.apply_demag(
            mesh,
            susceptibility=susceptibility,
            inplace=True,
            min_log_time=10,
        )
        _ = mesh.getB(observers)
        end = perf_counter()

        total_time += end - start

    return total_time / n_repeats


def time_nn(x, model, n_samples, n_repeats):
    observers = x[:n_samples, :]

    total_time = 0
    for _ in range(n_repeats):
        start = perf_counter()
        _ = model(observers)
        end = perf_counter()

        total_time += end - start

    return total_time / n_repeats


def plot_times(batch_sizes, times_ana, times_demag, times_nn):
    assert len(times_ana) == len(times_demag) == len(times_nn)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        batch_sizes,
        times_demag,
        label="Full Solution",
        color=colors_[1],
        marker=markers[1],
    )
    ax.annotate("Magnetostatic MoM", xy=(3e1, 5e-1))

    ax.plot(
        batch_sizes,
        times_ana,
        label="Analytical solution",
        color=colors_[0],
        marker=markers[0],
    )
    ax.annotate("Analytical Solution", xy=(3e3, 4e-4))

    ax.plot(
        batch_sizes,
        times_nn,
        label="NN Solution",
        color=colors_[2],
        marker=markers[2],
    )
    ax.annotate("NN Solution", xy=(3e1, 1.2e-3))

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Batch Size")
    # ax.legend()

    ax.set_ylabel("Runtime (s)")
    ax.grid(alpha=0.5)
    ax.tick_params(
        axis="both",  # Apply to both x and y axis
        which="both",  # Apply to both major and minor ticks
        direction="in",  # Tick direction 'in' for inside the plot
        top=False,  # Show ticks on top
        right=False,  # Show ticks on right
        length=4,
    )
    plt.tight_layout()
    # plt.savefig("./runtimes.pdf", format="pdf")

    return fig, ax


def main():
    X, _ = AnisotropicData(DATA_PATH, dtype=torch.float32).get_magnets()

    x = X[0]
    a, b, susceptibility = x[0, 0], x[0, 1], x[0, 2:5]

    cuboid = magnet.Cuboid(dimension=(a, b, 1), polarization=(0, 0, 1))
    mesh = meshing.mesh_Cuboid(cuboid=cuboid, target_elems=100)

    model = SphericalCorrectionNetwork.load_from_path(
        Path(WEIGHTS_PATH) / "best_weights.pt",
        activation=torch.nn.functional.silu,
        save_path=None,
        save_weights=False,
        do_output_activation=False,
    )

    print("\n" + "=" * 10 + " Analytical Calculation " + "=" * 10)
    times_analytical = [
        time_analytical(
            x=x,
            n_samples=b,
            n_repeats=N_REPEATS,
            dimensions=[a, b, 1],
        )
        for b in bar(BATCH_SIZES)
    ]

    print("\n" + "=" * 14 + " MoM Calculation " + "=" * 13)
    times_demag = [
        time_demag(
            x=x,
            mesh=mesh,
            n_samples=b,
            n_repeats=N_REPEATS,
            susceptibility=susceptibility,
        )
        for b in bar(BATCH_SIZES)
    ]

    print("\n" + "=" * 14 + " NN Calculation " + "=" * 14)
    times_nn = [
        time_nn(
            x=x,
            model=model,
            n_samples=b,
            n_repeats=N_REPEATS,
        )
        for b in bar(BATCH_SIZES)
    ]

    data = {
        "n_repeats": N_REPEATS,
        "batch_sizes": BATCH_SIZES,
        "times_nn": times_nn,
        "times_demag": times_demag,
        "times_ana": times_analytical,
    }

    with open("notebooks/runtimes/data.json", "w+") as f:
        json.dump(data, f)

    fig, ax = plot_times(BATCH_SIZES, times_analytical, times_demag, times_nn)
    plt.savefig("notebooks/runtimes/runtimes_v1.pdf")
    plt.show()


if __name__ == "__main__":
    main()

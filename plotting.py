import json
import os
from typing import Dict, Literal, Tuple

from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from utils import relative_amplitude_error, angle_error


def plot_loss(stats: Dict, save_path: str | None = None, show_plot: bool = False):
    if save_path is None and not show_plot:
        raise ValueError(
            "At least one of `show_plot` and `save_path must be specified.`"
        )

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    ax[0].set_yscale("log")
    ax[0].plot(stats["train_loss"], label="Train")
    ax[0].plot(stats["test_loss"], label="Test")
    ax[0].legend()

    ax[1].plot(stats["angle_error"], label="Angle error")
    ax[1].legend()

    ax[2].plot(stats["amplitude_error"], label="Amplitude error")
    ax[2].legend()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(save_path)

        with open(f"{save_path}/training_stats.json", "w+") as f:
            json.dump(stats, f)

        plt.savefig(f"{save_path}/loss.png", format="png")

    if show_plot:
        plt.show()
    else:
        del fig, ax


def plot_histograms(stats: Dict, save_path: str | None, show_plot: bool):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

    mean_angle_baseline = np.mean(stats["angle_errors_baseline"])
    mean_amp_baseline = np.mean(stats["amp_errors_baseline"])
    mean_amp = np.mean(stats["amp_errors"])
    mean_angle = np.mean(stats["angle_errors"])

    ax[0, 0].hist(
        stats["angle_errors_baseline"],
        bins=20,
        label=f"Avg Error: {round(mean_angle_baseline, 2)} degrees",
    )
    ax[0, 0].set_ylabel("Count (Baseline)")
    ax[0, 0].legend()

    ax[0, 1].hist(
        stats["amp_errors_baseline"],
        bins=20,
        label=f"Avg Error: {round(mean_amp_baseline, 2)}%",
    )
    ax[0, 1].legend()

    ax[1, 0].hist(
        stats["angle_errors"],
        bins=20,
        label=f"Avg Error: {round(mean_angle, 2)} degrees",
    )
    ax[1, 0].set_xlabel("Angle Error")
    ax[1, 0].set_ylabel("Count (NN Correction)")
    ax[1, 0].legend()

    ax[1, 1].hist(
        stats["amp_errors"], bins=20, label=f"Avg Error: {round(mean_amp, 2)}%"
    )
    ax[1, 1].set_xlabel("Relative Amplitude Error")
    ax[1, 1].legend()

    if save_path is not None:
        with open(f"{save_path}/val_stats.json", "w+") as f:
            json.dump(stats, f)

        plt.savefig(f"{save_path}/hist.png", format="png")

    if show_plot:
        plt.show()


def plot_baseline_histograms(
    stats: Dict, save_path: str | None, show_plot: bool, **kwargs
):
    figsize = kwargs.pop("figsize", (10, 8))
    bins = kwargs.pop("bins", 20)

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=figsize)

    mean_angle_baseline = np.mean(stats["angle_errors_baseline"])
    mean_amp_baseline = np.mean(stats["amp_errors_baseline"])

    ax[1].hist(
        stats["angle_errors_baseline"],
        bins=bins,
        label=f"Avg Error: {round(mean_angle_baseline, 2)}°",
    )
    ax[0].set_ylabel("Count (Baseline)")
    ax[1].set_xlabel("Mean Angle Error")
    ax[1].legend()

    ax[0].hist(
        stats["amp_errors_baseline"],
        bins=bins,
        label=f"Avg Error: {round(mean_amp_baseline, 2)}%",
    )
    ax[0].legend()

    ax[0].set_xlabel("Mean Relative Amplitude Error")

    if save_path is not None:
        with open(f"{save_path}/val_stats.json", "w+") as f:
            json.dump(stats, f)

        plt.savefig(f"{save_path}/hist.png", format="png")

    if show_plot:
        plt.show()


def plot_magnet(
    coords: np.ndarray,
    values: np.ndarray,
    magnet_dims: Tuple[float, ...],
    chi: float,
    view: Literal["xy", "xz", "yz"] = "xy",
    resolution: int = 50,
):
    # Define the view mapping
    view_mapping = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

    if view not in view_mapping:
        raise ValueError("view must be one of 'xy', 'xz', or 'yz'")

    # Get indices for the chosen view
    i, j = view_mapping[view]

    # Create a regular grid to interpolate the data
    x_min, x_max = coords[:, i].min(), coords[:, i].max()
    y_min, y_max = coords[:, j].min(), coords[:, j].max()

    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate scattered data to regular grid
    zi = griddata(
        (coords[:, i], coords[:, j]),
        values,
        (xi, yi),
        method="cubic",
        fill_value=np.nan,
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the interpolated scalar field
    im = ax.contourf(xi, yi, zi, cmap="RdYlBu_r", shading="auto")

    # Add a colorbar
    plt.colorbar(im, ax=ax, label="|B|")
    # Draw magnet dimensions
    a, b, c = magnet_dims
    cx, cy, cz = (0, 0, 0)

    # Draw dimension lines based on the view
    if view == "xy":
        # Draw x-direction line
        ax.plot([cx, a / 2], [b / 2, b / 2], "r--", linewidth=2, label="Width (a)")
        # Draw y-direction line
        ax.plot([a / 2, a / 2], [cy, b / 2], "r--", linewidth=2, label="Height (b)")
    elif view == "xz":
        # Draw x-direction line
        ax.plot([cx, cx + a], [cz, cz], "r--", linewidth=2, label="Width (a)")
        # Draw z-direction line
        ax.plot([cx, cx], [cz, cz + c], "g--", linewidth=2, label="Depth (c)")
    else:  # 'yz'
        # Draw y-direction line
        ax.plot([cy, cy + b], [cz, cz], "b--", linewidth=2, label="Height (b)")
        # Draw z-direction line
        ax.plot([cy, cy], [cz, cz + c], "g--", linewidth=2, label="Depth (c)")

    # Set labels
    ax.set_xlabel(f"{view[0]} coordinate")
    ax.set_ylabel(f"{view[1]} coordinate")
    ax.set_title(f"Analytical solution - Chi={round(chi, 3)}")

    return fig, ax


def create_amplitude_error_plot(points, amp_errors, angle_errors, a, b):
    """
    Create a two-panel amplitude error plot from scattered point data.

    Parameters:
    points: array of shape (n, 3) containing [x, y, z] coordinates
    as_errors: array of shape (n,) containing AS error values
    nn_errors: array of shape (n,) containing NN error values
    """
    # Create regular grid for interpolation
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    grid_x = np.linspace(x_min, x_max, 26)
    grid_z = np.linspace(z_min, z_max, 26)
    X, Z = np.meshgrid(grid_x, grid_z)

    # Interpolate scattered data onto regular grid
    amp_grid = griddata(
        (points[:, 0], points[:, 2]),
        amp_errors,
        (X, Z),
        method="cubic",
        fill_value=np.nan,
    )

    angle_grid = griddata(
        (points[:, 0], points[:, 2]),
        angle_errors,
        (X, Z),
        method="cubic",
        fill_value=np.nan,
    )

    # Create figure with two subplots and space for colorbar
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[1, 1])

    # Set up colormap
    vmin1, vmax1 = min(amp_errors), max(amp_errors)

    levels1 = np.linspace(vmin1, vmax1, 21)

    vmin2, vmax2 = min(angle_errors), max(angle_errors)
    levels2 = np.linspace(vmin2, vmax2, 21)

    cmap = colormaps.get_cmap("RdYlBu_r")

    # Plot first heatmap (AS)
    cf1 = ax1.contourf(X, Z, amp_grid, levels=levels1, cmap=cmap, extend="both")
    ax1.set_title("Relative Amplitude Error (%)")

    # Plot second heatmap (NN)
    cf2 = ax2.contourf(X, Z, angle_grid, levels=levels2, cmap=cmap, extend="both")
    ax2.set_title("Angle Error (degrees)")

    # Add magnet annotation to both plots
    for ax in [ax1, ax2]:
        # Set axis labels and limits
        ax.set_xlabel("x [a.u.]")
        ax.set_ylabel("z [a.u.]")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

        ax.plot(
            [0, a / 2],
            [0.5, 0.5],
            "r--",
            linewidth=2,
        )
        # Draw y-direction line
        ax.plot(
            [a / 2, a / 2],
            [0, 0.5],
            "r--",
            linewidth=2,
        )

    # # Add colorbar
    cbar1 = plt.colorbar(
        cf1, cax=cax1, orientation="vertical", ticks=np.linspace(vmin1, vmax1, 10)
    )

    cbar2 = plt.colorbar(
        cf2, cax=cax2, orientation="vertical", ticks=np.linspace(vmin2, vmax2, 10)
    )

    # Adjust layout
    plt.tight_layout()

    return fig, (ax1, ax2)


if __name__ == "__main__":
    from dataset import ChiMode, get_data

    X_all, B_all = get_data("data/old", chi_mode=ChiMode.ISOTROPIC)
    X, B = np.squeeze(X_all), np.squeeze(B_all)

    B_demag = B[..., :3]
    B_ana = B[..., 3:]

    angle_errors_baseline = np.mean(angle_error(B_demag, B_ana), axis=1)
    amp_errors_baseline = np.mean(relative_amplitude_error(B_demag, B_ana), axis=1)

    stats = {
        "angle_errors_baseline": angle_errors_baseline,
        "amp_errors_baseline": amp_errors_baseline,
    }

    plot_baseline_histograms(stats, show_plot=True, save_path=None)

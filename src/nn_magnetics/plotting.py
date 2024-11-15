import json
from typing import Dict

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patches

from nn_magnetics.utils.cmaps import CMAP_ANGLE, CMAP_AMPLITUDE
from nn_magnetics.dataset import ChiMode, get_one_magnet
from nn_magnetics.validate import validate
from nn_magnetics.utils.metrics import calculate_metrics, calculate_metrics_baseline


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
        with open(f"{save_path}/training_stats.json", "w+") as f:
            json.dump(stats, f)

        plt.savefig(f"{save_path}/loss.png", format="png")

    if show_plot:
        plt.show()


def _plot_histograms(
    stats: Dict, save_path: str | None, show_plot: bool, tag: str = ""
):
    fig, ax = plt.subplots(
        ncols=2, nrows=2, figsize=(10, 10), sharex="col", sharey="col"
    )

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
        bins=10,
        label=f"Avg Error: {round(mean_angle, 2)} degrees",
    )
    ax[1, 0].set_xlabel("Mean Angle Error (degrees)")
    ax[1, 0].set_ylabel("Count (NN Correction)")
    ax[1, 0].legend()

    ax[1, 1].hist(
        stats["amp_errors"],
        bins=10,
        label=f"Avg Error: {round(mean_amp, 2)}%",
    )
    ax[1, 1].set_xlabel("Mean Relative Amplitude Error (%)")
    ax[1, 1].legend()

    plt.suptitle("Mean Errors Per Magnet")

    if save_path is not None:
        with open(f"{save_path}/val_stats.json", "w+") as f:
            json.dump(stats, f)

        plt.savefig(f"{save_path}/hist_{tag}.png", format="png")

    if show_plot:
        plt.show()


def plot_histograms(
    X_test: np.ndarray,
    B_test: np.ndarray,
    model: torch.nn.Module,
    save_path: str | None = None,
    show_plot: bool = False,
    tag: str = "",
):
    val_stats = validate(X_test, B_test, model)

    _plot_histograms(
        val_stats,
        save_path=save_path,
        show_plot=show_plot,
        tag=tag,
    )


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
        plt.savefig(f"{save_path}/hist.png", format="png")

    if show_plot:
        plt.show()


def plot_heatmaps_amplitude(
    grid,
    amplitude_errors_baseline,
    amplitude_errors_trained,
    a,
    b,
    tag,
    save_path=None,
    show_plot=False,
):
    eps_x = 0.01
    eps_y = 0.01

    x = grid.T[0] * a
    y = grid.T[1] * b
    z = grid.T[2]

    mask = y == y[0]
    x_slice = x[mask]
    z_slice = z[mask]

    amplitude_errors_trained_slice = amplitude_errors_trained[mask]
    amplitude_errors_baseline_slice = amplitude_errors_baseline[mask]

    x_bins = np.linspace(min(x_slice), max(x_slice), 25)
    z_bins = np.linspace(min(z_slice), max(z_slice), 25)

    vmin = min(
        min(amplitude_errors_trained_slice),
        min(amplitude_errors_baseline_slice),
    )

    vmax = max(
        max(amplitude_errors_trained_slice),
        max(amplitude_errors_baseline_slice),
    )

    norm_amplitude = colors.TwoSlopeNorm(
        vmin=vmin,
        vcenter=0,
        vmax=vmax,
    )

    heatmap_amplitude_trained, x_edges, z_edges = np.histogram2d(
        x_slice,
        z_slice,
        bins=[x_bins, z_bins],
        weights=amplitude_errors_trained_slice,
    )
    heatmap_counts_amplitude_trained, _, _ = np.histogram2d(
        x_slice, z_slice, bins=[x_bins, z_bins]
    )

    heatmap_amplitude_trained = np.divide(
        heatmap_amplitude_trained,
        heatmap_counts_amplitude_trained,
        where=heatmap_counts_amplitude_trained != 0,
    )

    heatmap_amplitude_baseline, x_edges, z_edges = np.histogram2d(
        x_slice,
        z_slice,
        bins=[x_bins, z_bins],
        weights=amplitude_errors_baseline_slice,
    )
    heatmap_counts_amplitude_baseline, _, _ = np.histogram2d(
        x_slice, z_slice, bins=[x_bins, z_bins]
    )

    heatmap_amplitude_baseline = np.divide(
        heatmap_amplitude_baseline,
        heatmap_counts_amplitude_baseline,
        where=heatmap_counts_amplitude_baseline != 0,
    )

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

    mesh = axs[0].pcolormesh(
        x_edges,
        z_edges,
        heatmap_amplitude_trained.T,
        shading="auto",
        cmap=CMAP_AMPLITUDE,
        norm=norm_amplitude,
    )

    axs[0].set_xlabel("X (mm)")
    axs[0].set_ylabel("Z (mm) - NN Solution")
    axs[0].add_patch(
        patches.Rectangle(
            (0, 0),
            width=a / 2 + eps_x,
            height=1 / 2 + eps_y,
            linewidth=2,
            edgecolor="k",
            facecolor="none",
        )
    )

    mesh = axs[1].pcolormesh(
        x_edges,
        z_edges,
        heatmap_amplitude_baseline.T,
        shading="auto",
        cmap=CMAP_AMPLITUDE,
        norm=norm_amplitude,
    )

    axs[1].set_xlabel("X (mm)")
    axs[1].set_ylabel("Z (mm) - Analytical Solution")
    axs[1].add_patch(
        patches.Rectangle(
            (0, 0),
            width=a / 2 + eps_x,
            height=1 / 2 + eps_y,
            linewidth=2,
            edgecolor="k",
            facecolor="none",
        )
    )

    cbar = fig.colorbar(mesh, ax=axs.ravel().tolist())
    cbar.set_label("Relative Amplitude Error (%)")

    if save_path is not None:
        plt.savefig(f"{save_path}/heatmap_amplitude_{tag}.png", format="png")

    if show_plot:
        plt.show()


def plot_heatmaps_angle(
    grid,
    angle_errors_baseline,
    angle_errors_trained,
    a,
    b,
    tag,
    save_path=None,
    show_plot=False,
):
    eps_x = 0.01
    eps_y = 0.01

    x = grid.T[0] * a
    y = grid.T[1] * b
    z = grid.T[2]

    mask = y == y[0]
    x_slice = x[mask]
    z_slice = z[mask]

    angle_errors_trained_slice = angle_errors_trained[mask]
    angle_errors_baseline_slice = angle_errors_baseline[mask]

    x_bins = np.linspace(min(x_slice), max(x_slice), 25)
    z_bins = np.linspace(min(z_slice), max(z_slice), 25)

    heatmap_angle_trained, x_edges, z_edges = np.histogram2d(
        x_slice,
        z_slice,
        bins=[x_bins, z_bins],
        weights=angle_errors_trained_slice,
    )
    heatmap_counts_angle_trained, _, _ = np.histogram2d(
        x_slice, z_slice, bins=[x_bins, z_bins]
    )

    heatmap_angle_trained = np.divide(
        heatmap_angle_trained,
        heatmap_counts_angle_trained,
        where=heatmap_counts_angle_trained != 0,
    )

    heatmap_angle_baseline, x_edges, z_edges = np.histogram2d(
        x_slice,
        z_slice,
        bins=[x_bins, z_bins],
        weights=angle_errors_baseline_slice,
    )
    heatmap_counts_amplitude_baseline, _, _ = np.histogram2d(
        x_slice, z_slice, bins=[x_bins, z_bins]
    )

    heatmap_amplitude_baseline = np.divide(
        heatmap_angle_baseline,
        heatmap_counts_amplitude_baseline,
        where=heatmap_counts_amplitude_baseline != 0,
    )

    # Plot the heatmap
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

    mesh = axs[0].pcolormesh(
        x_edges,
        z_edges,
        heatmap_angle_trained.T,
        shading="auto",
        cmap=CMAP_ANGLE,
    )
    # axs[0].quiver(x_slice, z_slice, Bx, Bz)
    axs[0].set_xlabel("X (mm)")
    axs[0].set_ylabel("Z (mm) - NN Solution")
    axs[0].add_patch(
        patches.Rectangle(
            (0, 0),
            width=a / 2 + eps_x,
            height=1 / 2 + eps_y,
            linewidth=2,
            edgecolor="k",
            facecolor="none",
        )
    )
    # plt.colorbar(mesh, label="Relative amplitude error (%)", ax=axs[0])

    mesh = axs[1].pcolormesh(
        x_edges,
        z_edges,
        heatmap_amplitude_baseline.T,
        shading="auto",
        cmap=CMAP_ANGLE,
    )
    # axs[1].quiver(x_slice, z_slice, Bx_pred, Bz_pred)
    axs[1].set_xlabel("X (mm)")
    axs[1].set_ylabel("Z (mm) - Analytical Solution")
    axs[1].add_patch(
        patches.Rectangle(
            (0, 0),
            width=a / 2 + eps_x,
            height=1 / 2 + eps_y,
            linewidth=2,
            edgecolor="k",
            facecolor="none",
        )
    )

    cbar = fig.colorbar(mesh, ax=axs.ravel().tolist())
    cbar.set_label("Angle Error (°)")

    if save_path is not None:
        plt.savefig(f"{save_path}/heatmap_angle_{tag}.png", format="png")

    if show_plot:
        plt.show()


def plot_heatmaps(
    model,
    save_path,
    tag,
    eval_path="data/anisotropic_chi/test_anisotropic/data_1.npz",
):
    X, B = get_one_magnet(
        chi_mode=ChiMode.ANISOTROPIC,
        data=np.load(eval_path),
    )

    grid = X[:, 4:]
    a = float(X[0, 0])
    b = float(X[0, 1])

    with torch.no_grad():
        B_pred = model(torch.tensor(X))

    angle_errors_baseline, amplitude_errors_baseline = calculate_metrics_baseline(
        B=B,
        return_abs=False,
    )
    angle_errors_trained, amplitude_errors_trained = calculate_metrics(
        B=torch.tensor(B),
        B_pred=B_pred,
        return_abs=False,
    )

    plot_heatmaps_amplitude(
        grid=grid,
        amplitude_errors_baseline=amplitude_errors_baseline,
        amplitude_errors_trained=amplitude_errors_trained,
        a=a,
        b=b,
        tag=tag,
        save_path=save_path,
    )

    plot_heatmaps_angle(
        grid=grid,
        angle_errors_baseline=angle_errors_baseline,
        angle_errors_trained=angle_errors_trained,
        a=a,
        b=b,
        tag=tag,
        save_path=save_path,
    )

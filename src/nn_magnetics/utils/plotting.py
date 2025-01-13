import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patches

from nn_magnetics.utils.cmaps import CMAP_ANGLE, CMAP_AMPLITUDE
from nn_magnetics.data.dataset import IsotropicData
from nn_magnetics.utils.metrics import (
    calculate_metrics_baseline,
    calculate_metrics_trained,
)


def plot_loss(
    train_loss,
    validation_loss,
    angle_error,
    amplitude_error,
    n_epochs,
):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True)

    plt.suptitle("Learning Curves")
    ax[0].set_xlim((0, n_epochs - 1))
    ax[0].plot(train_loss, label="Train")
    ax[0].plot(validation_loss, label="Test")
    ax[0].legend()

    ax[1].plot(angle_error, label="Angle error")
    ax[2].plot(amplitude_error, label="Amplitude error")
    ax[2].legend()

    plt.tight_layout()

    return fig, ax


def plot_baseline_histograms(
    path,
    figsize=(10, 8),
    bins=20,
):
    _, B = IsotropicData(path).get_magnets()

    angle_errors, amplitude_errors = [], []

    for Bi in B:
        angle_error, amp_error = calculate_metrics_baseline(Bi)
        angle_errors.append(torch.mean(angle_error))
        amplitude_errors.append(torch.mean(amp_error))

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=figsize)

    mean_angle_baseline = np.mean(angle_errors)
    mean_amp_baseline = np.mean(amplitude_errors)

    ax[0].set_ylabel("Count (Baseline)")
    ax[0].hist(
        amplitude_errors,
        bins=bins,
        label=f"Avg Error: {round(mean_amp_baseline, 2)}%",
    )
    ax[0].legend()
    ax[0].set_xlabel("Mean Relative Amplitude Error")

    ax[1].hist(
        angle_errors,
        bins=bins,
        label=f"Avg Error: {round(mean_angle_baseline, 2)}°",
    )
    ax[1].set_xlabel("Mean Angle Error")
    ax[1].legend()

    return fig, ax


def plot_histograms(
    X,
    B,
    model,
    figsize=(10, 10),
    bins=20,
):
    angle_errors_baseline, amplitude_errors_baseline = [], []

    for Bi in B:
        angle_error, amp_error = calculate_metrics_baseline(Bi)
        angle_errors_baseline.append(torch.mean(angle_error))
        amplitude_errors_baseline.append(torch.mean(amp_error))

    angle_errors, amplitude_errors = [], []

    for Xi, Bi in zip(X, B):
        angle_error, amp_error = calculate_metrics_trained(Xi, Bi, model)
        angle_errors.append(torch.mean(angle_error))
        amplitude_errors.append(torch.mean(amp_error))

    fig, ax = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=figsize,
        sharex="col",
        sharey="col",
    )

    mean_angle_baseline = round(np.mean(angle_errors_baseline), 4)
    mean_amp_baseline = round(np.mean(amplitude_errors_baseline), 4)
    mean_angle = round(np.mean(angle_errors), 4)
    mean_amp = round(np.mean(amplitude_errors), 4)

    print(mean_angle_baseline, mean_amp_baseline, mean_angle, mean_amp)

    ax[0, 0].set_ylabel("Count (Baseline)")
    ax[0, 0].hist(
        angle_errors_baseline,
        bins=bins,
        label=f"Avg Error: {mean_angle_baseline}°",
    )
    ax[0, 0].set_ylabel("Count (Baseline)")
    ax[0, 0].legend()

    ax[0, 1].hist(
        amplitude_errors_baseline,
        bins=bins,
        label=f"Avg Error: {mean_amp_baseline}%",
    )
    ax[0, 1].legend()

    ax[1, 0].hist(
        angle_errors,
        bins=bins,
        label=f"Avg Error: {mean_angle}°",
    )
    ax[1, 0].set_xlabel("Mean Angle Error (°)")
    ax[1, 0].set_ylabel("Count (NN Correction)")
    ax[1, 0].legend()

    ax[1, 1].hist(
        amplitude_errors,
        bins=bins,
        label=f"Avg Error: {mean_amp}%",
    )
    ax[1, 1].set_xlabel("Mean Relative Amplitude Error (%)")
    ax[1, 1].legend()

    return fig, ax


def plot_heatmaps_amplitude(
    grid: np.ndarray,
    amplitude_errors_baseline: np.ndarray,
    amplitude_errors_trained: np.ndarray,
    a: float,
    b: float,
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

    return fig, axs


def plot_heatmaps_angle(
    grid: np.ndarray,
    angle_errors_baseline: np.ndarray,
    angle_errors_trained: np.ndarray,
    a: float,
    b: float,
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

    return fig, axs


def plot_heatmaps(
    model: torch.nn.Module,
    X: torch.Tensor,
    B: torch.Tensor,
):
    grid = X[:, 3:]  # replace with 4 for anisotropic
    a = float(X[0, 0])
    b = float(X[0, 1])

    angle_errors_baseline, amplitude_errors_baseline = calculate_metrics_baseline(
        B=B,
        return_abs=False,
    )
    angle_errors_trained, amplitude_errors_trained = calculate_metrics_trained(
        X=X,
        B=B,
        model=model,
        return_abs=False,
    )

    fig, axs = plot_heatmaps_amplitude(
        grid=grid.numpy(),
        amplitude_errors_baseline=amplitude_errors_baseline.numpy(),
        amplitude_errors_trained=amplitude_errors_trained.numpy(),
        a=a,
        b=b,
    )

    plt.show()

    fig, axs = plot_heatmaps_angle(
        grid=grid.numpy(),
        angle_errors_baseline=angle_errors_baseline.numpy(),
        angle_errors_trained=angle_errors_trained.numpy(),
        a=a,
        b=b,
    )
    plt.show()

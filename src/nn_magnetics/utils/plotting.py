from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch
from matplotlib import colors, patches

from nn_magnetics.data.dataset import IsotropicData
from nn_magnetics.utils.cmaps import CMAP_AMPLITUDE, CMAP_ANGLE
from nn_magnetics.utils.metrics import (
    calculate_metrics_baseline,
    calculate_metrics_trained,
    calculate_metrics_trained_gnn,
)


def plot_div_loss(train_div, valid_div):
    n_epochs = len(train_div)
    plt.plot(train_div, label="Train")
    plt.plot(valid_div, label="Validation")

    plt.xlim((0, n_epochs - 1))
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Divergence")

    plt.show()


def plot_loss(
    train_loss: list,
    validation_loss: list,
    angle_error: list,
    amplitude_error: list,
    save_path: Path | None,
    baselines: tuple | None = None,  # (loss, angle, amp)
    log_scale: bool = False,
):
    ax: list[Axes]
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True)

    for a in ax:
        a.set_xlabel("Epochs")

        if log_scale:
            a.set_yscale("log")

    n_epochs = len(train_loss)

    ax[0].set_xlim((0, n_epochs - 1))
    ax[0].plot(train_loss, label="Train")
    ax[0].plot(validation_loss, label="Test")
    ax[0].legend()
    ax[0].set_ylabel("Loss")
    if baselines is not None:
        ax[0].hlines(baselines[0], 0, n_epochs, colors="black", linestyles="dashed")

    ax[1].plot(angle_error, label="Angle error")
    ax[1].set_ylabel("Angle Error (°)")
    # ax[1].set_ylim(bottom=0, top=max(angle_error) + 0.1)
    if baselines is not None:
        ax[1].hlines(baselines[1], 0, n_epochs, colors="black", linestyles="dashed")

    ax[2].plot(amplitude_error, label="Amplitude error")
    ax[2].set_ylabel("Relative Amplitude Error (%)")
    # ax[2].set_ylim(bottom=0, top=max(amplitude_error) + 0.1)
    if baselines is not None:
        ax[2].hlines(baselines[2], 0, n_epochs, colors="black", linestyles="dashed")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(f"{save_path}/learning_curves.png")
    else:
        plt.show()


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


def plot_histograms_gnn(loader, model, save_path, figsize=(8, 8), bins=20, tag=""):
    angle_errors_baseline, amplitude_errors_baseline = [], []

    for graph in loader:
        angle_error, amp_error = calculate_metrics_baseline(graph.y)
        angle_errors_baseline.append(torch.mean(angle_error))
        amplitude_errors_baseline.append(torch.mean(amp_error))

    angle_errors, amplitude_errors = [], []

    for graph in loader:
        angle_error, amp_error = calculate_metrics_trained_gnn(graph, model)
        angle_errors.append(torch.nan_to_num(torch.mean(angle_error), nan=180.0))
        amplitude_errors.append(torch.mean(amp_error))

    fig, ax = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=figsize,
        sharex="col",
        sharey="col",
    )

    mean_angle_baseline = round(float(np.mean(angle_errors_baseline)), 4)
    mean_amp_baseline = round(float(np.mean(amplitude_errors_baseline)), 4)
    mean_angle = round(float(np.mean(angle_errors)), 4)
    mean_amp = round(float(np.mean(amplitude_errors)), 4)

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

    if save_path is not None:
        fig.savefig(f"{save_path}/histograms{tag}.png")
    else:
        plt.show()


def plot_histograms(X, B, model, save_path, figsize=(8, 8), bins=20, tag=""):
    angle_errors_baseline, amplitude_errors_baseline = [], []

    for Bi in B:
        angle_error, amp_error = calculate_metrics_baseline(Bi)
        angle_errors_baseline.append(torch.mean(angle_error).numpy(force=True))
        amplitude_errors_baseline.append(torch.mean(amp_error).numpy(force=True))

    angle_errors, amplitude_errors = [], []

    for Xi, Bi in zip(X, B):
        angle_error, amp_error = calculate_metrics_trained(Xi, Bi, model)
        angle_errors.append(
            torch.nan_to_num(torch.mean(angle_error), nan=180.0).numpy(force=True)
        )
        amplitude_errors.append(torch.mean(amp_error).numpy(force=True))

    fig, ax = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=figsize,
        sharex="col",
        sharey="col",
    )

    mean_angle_baseline = round(float(np.mean(angle_errors_baseline)), 4)
    mean_amp_baseline = round(float(np.mean(amplitude_errors_baseline)), 4)
    mean_angle = round(float(np.mean(angle_errors)), 4)
    mean_amp = round(float(np.mean(amplitude_errors)), 4)

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

    if save_path is not None:
        fig.savefig(f"{save_path}/histograms{tag}.png")
    else:
        plt.show()


def plot_component_error(X, B, model, save_path):
    B_demag, B_reduced = B[..., :3], B[..., 3:]

    with torch.no_grad():
        predictions = model(X)
        B_corrected = model.correct_ansatz(B_reduced, predictions)

    field_measured = B_demag
    field_simulated = B_corrected

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
    ax1.plot(
        np.abs((field_measured[:, 0] - field_simulated[:, 0]) / field_measured[:, 0])
        * 100
    )
    ax1.set_title("X")
    ax1.set_ylabel("Relative Error (%)")
    ax1.set_xlabel("Point")

    ax2.plot(
        np.abs((field_measured[:, 1] - field_simulated[:, 1]) / field_measured[:, 0])
        * 100,
    )
    ax2.set_title("Y")
    ax2.set_xlabel("Point")

    ax3.plot(
        np.abs((field_measured[:, 2] - field_simulated[:, 2]) / field_measured[:, 2])
        * 100,
    )
    ax3.set_title("Z")
    ax3.set_xlabel("Point")

    fig.suptitle("Relative Error of NN Solution")

    if save_path is not None:
        plt.savefig(f"{save_path}/component-errors.png", format="png")
    else:
        plt.show()


def plot_component_error_histograms(X, B, model, save_path):
    B_demag, B_reduced = B[..., :3], B[..., 3:]

    with torch.no_grad():
        predictions = model(X)
        B_corrected = model.correct_ansatz(B_reduced, predictions)

    field_measured1 = B_demag.detach().numpy()
    field_simulated1 = B_corrected.detach().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

    mean = np.mean(
        np.abs((field_measured1 - field_simulated1) / field_measured1) * 100,
        axis=0,
    )

    ax1.hist(
        np.abs((field_measured1[:, 0] - field_simulated1[:, 0]) / field_measured1[:, 0])
        * 100,
        bins=20,
    )
    ax1.set_title(f"X: Mean={round(float(mean[0]), 4)}")
    ax1.set_ylabel("Count")
    ax2.set_xlabel("Relative Error (%)")

    ax2.hist(
        np.abs((field_measured1[:, 1] - field_simulated1[:, 1]) / field_measured1[:, 0])
        * 100,
        bins=20,
    )
    ax2.set_title(f"Y: Mean={round(float(mean[1]), 4)}")
    ax2.set_xlabel("Relative Error (%)")

    ax3.hist(
        np.abs((field_measured1[:, 2] - field_simulated1[:, 2]) / field_measured1[:, 2])
        * 100,
        bins=20,
    )
    ax3.set_title(f"Z: Mean={round(float(mean[2]), 4)}")
    ax3.set_xlabel("Relative Error (%)")

    fig.suptitle("Relative Error Frequency of NN Solution")

    if save_path is not None:
        plt.savefig(f"{save_path}/relative-error-histograms.png", format="png")
    else:
        plt.show()


def plot_heatmaps_amplitude(
    grid: np.ndarray,
    amplitude_errors_baseline: np.ndarray,
    amplitude_errors_trained: np.ndarray,
    a: float,
    b: float,
):
    eps_x = 0.01
    eps_y = 0.01

    x = grid.T[0]
    y = grid.T[1]
    z = grid.T[2]

    mask = y == y[0]
    x_slice = x[mask]
    z_slice = z[mask]

    amplitude_errors_trained_slice = np.clip(amplitude_errors_trained[mask], -6, 6)
    amplitude_errors_baseline_slice = np.clip(amplitude_errors_baseline[mask], -6, 6)

    x_bins = np.linspace(min(x_slice), max(x_slice), 25)
    z_bins = np.linspace(min(z_slice), max(z_slice), 25)

    # vmin = min(
    #     min(amplitude_errors_trained_slice),
    #     min(amplitude_errors_baseline_slice),
    # )

    # vmax = max(
    #     max(amplitude_errors_trained_slice),
    #     max(amplitude_errors_baseline_slice),
    # )

    vmin, vmax = -6, 6

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

    axs: list[Axes]
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

    mesh = axs[0].pcolormesh(
        x_edges,
        z_edges,
        heatmap_amplitude_trained.T,
        shading="auto",
        cmap=CMAP_AMPLITUDE,
        norm=norm_amplitude,
    )

    axs[0].set_xlabel("X (a.u.)")
    axs[0].set_ylabel("Z (a.u.) - NN Solution")
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

    axs[1].set_xlabel("X (a.u.)")
    axs[1].set_ylabel("Z (a.u.) - Analytical Solution")
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

    cbar = fig.colorbar(mesh, ax=axs.ravel().tolist())  # type: ignore (its a ndarray rather than list but list is better for type hi)
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

    x = grid.T[0]
    y = grid.T[1]
    z = grid.T[2]

    mask = y == y[0]
    x_slice = x[mask]
    z_slice = z[mask]

    angle_errors_trained_slice = np.clip(angle_errors_trained[mask], 0, 3)
    angle_errors_baseline_slice = np.clip(angle_errors_baseline[mask], 0, 3)

    vmin, vmax = 0, 3

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
        vmin=vmin,
        vmax=vmax,
    )
    # axs[0].quiver(x_slice, z_slice, Bx, Bz)
    axs[0].set_xlabel("X (a.u.)")
    axs[0].set_ylabel("Z (a.u.) - NN Solution")
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
        vmin=vmin,
        vmax=vmax,
    )
    # axs[1].quiver(x_slice, z_slice, Bx_pred, Bz_pred)
    axs[1].set_xlabel("X (a.u.)")
    axs[1].set_ylabel("Z (a.u.) - Analytical Solution")
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
    save_path: str | Path | None,
    tag: str = "",
):
    grid = X[:, 5:]  # replace with 4 for anisotropic
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

    fig1, _ = plot_heatmaps_amplitude(
        grid=grid.numpy(force=True),
        amplitude_errors_baseline=amplitude_errors_baseline.numpy(force=True),
        amplitude_errors_trained=amplitude_errors_trained.numpy(force=True),
        a=a,
        b=b,
    )

    if save_path is not None:
        fig1.savefig(f"{save_path}/amplitude_heatmap_{tag}.png")
    else:
        plt.show()

    fig2, _ = plot_heatmaps_angle(
        grid=grid.numpy(force=True),
        angle_errors_baseline=angle_errors_baseline.numpy(force=True),
        angle_errors_trained=angle_errors_trained.numpy(force=True),
        a=a,
        b=b,
    )

    if save_path is not None:
        fig2.savefig(f"{save_path}/angle_heatmap_{tag}.png")
    else:
        plt.show()


def one_magnet_errors(X, B, model, save_path):
    grid = X[:, 5:]  # replace with 4 for anisotropic
    a = float(X[0, 0])
    b = float(X[0, 1])

    amp_err, angle_err = calculate_metrics_trained(X, B, model, True)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

    angle_mean = round(float(np.mean(angle_err, where=~np.isnan(angle_err))), 3)
    angle_std = round(float(np.mean(angle_err, where=~np.isnan(angle_err))), 2)


def quiverplot(grid, B_hom, B_demag):
    x, z = grid[..., 0], grid[..., 2]

    Bx, Bz = B_hom[..., 0], B_hom[..., 2]
    Bdx, Bdz = B_demag[..., 0], B_demag[..., 2]

    fig, ax = plt.subplots()

    ax.quiver(x, z, Bx, Bz, color="blue", scale=8, label="$B$ Analytical", width=0.004)
    ax.quiver(
        x, z, Bdx, Bdz, color="red", scale=8, label="$B$ w/ Demag Effects", width=0.004
    )

    # ax.legend(loc=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x [a.u.]")
    ax.set_ylabel("z [a.u.]")

    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            0.5,
            0.5,
            fill=False,
            lw=1,
            alpha=0.8,
            color="k",
        )
    )

    ax.legend(loc=2)

    x1, x2, z1, z2 = 0.51, 0.58, 0.51, 0.55

    axins = ax.inset_axes(
        (0.7, 0.45, 0.5, 0.5),
        xlim=(x1, x2),
        ylim=(z1, z2),
        xticklabels=[],
        yticklabels=[],
    )
    ax.indicate_inset_zoom(axins, edgecolor="k", lw=1)

    axins.quiver(x, z, Bx, Bz, color="blue", scale=1, width=0.012)
    axins.quiver(x, z, Bdx, Bdz, color="red", scale=1, width=0.012)

    return fig, ax

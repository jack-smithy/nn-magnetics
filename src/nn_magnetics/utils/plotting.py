from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch
from matplotlib import colors, patches

from nn_magnetics.utils.metrics import (
    calculate_metrics_baseline,
    calculate_metrics_trained,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 16,
    }
)

labels = ["Component", "Euler", "Quaternion"]
colors_ = ["#784c99", "#6780be", "#e93ea5"]
markers = ["v", "o", "*"]


def plot_loss(train_loss, validation_loss, angle_error, amp_error, save_path, tag=None):
    ax: list[Axes]
    fig, ax = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 5),
    )

    n_epochs = len(train_loss[0])

    for i in range(3):
        ax[0].scatter(
            range(1, n_epochs + 1),
            validation_loss[i],
            label=labels[i],
            color=colors_[i],
            s=20,
            linewidths=2,
            marker=markers[i],
        )
    ax[0].set_xlim((0, n_epochs - 1))
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Validation Loss")

    for i in range(3):
        ax[1].scatter(
            range(1, n_epochs + 1),
            angle_error[i],
            label=labels[i],
            color=colors_[i],
            s=20,
            linewidths=2,
            marker=markers[i],
        )
    ax[1].set_xlim((0, n_epochs - 1))
    ax[1].set_ylabel("Angle Error (°)")

    for i in range(3):
        ax[2].scatter(
            range(1, n_epochs + 1),
            amp_error[i],
            label=labels[i],
            color=colors_[i],
            s=20,
            linewidths=2,
            marker=markers[i],
        )
    ax[2].set_xlim((0, n_epochs - 1))
    ax[2].set_ylabel("Relative Amplitude Error (%)")
    ax[2].legend()

    for a in ax:
        a.set_xlabel("Epoch")
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

    if save_path is not None:
        if tag is not None:
            if tag[0] != "_":
                tag = f"_{tag}"
        else:
            tag = ""

        fig.savefig(f"{save_path}/learning_curves{tag}.pdf", format="pdf")
    else:
        plt.show()


def plot_training(
    train_loss: list,
    validation_loss: list,
    angle_error: list,
    amplitude_error: list,
    save_path: Path | None,
    baselines: tuple | None = None,  # (loss, angle, amp)
    log_scale: bool = False,
    tag: str | None = None,
):
    ax: list[Axes]
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 6), sharex=True)

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
        if tag is not None:
            if tag[0] != "_":
                tag = f"_{tag}"

        fig.savefig(f"{save_path}/learning_curves{tag}.png")
    else:
        plt.show()


def plot_baseline_histograms(B, figsize=(10, 8), bins=20, reduction=np.mean):
    angle_errors, amplitude_errors = [], []

    for Bi in B:
        angle_error, amp_error = calculate_metrics_baseline(Bi)
        angle_errors.append(reduction(angle_error.numpy()))
        amplitude_errors.append(reduction(amp_error.numpy()))

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=True)

    mean_angle_baseline = np.mean(angle_errors)
    mean_amp_baseline = np.mean(amplitude_errors)

    ax[0].set_ylabel("Frequency")
    ax[0].hist(
        angle_errors,
        bins=bins,
        label=f"Avg Error: {round(mean_amp_baseline, 2)}%",
        color=colors_[1],
        edgecolor="black",
    )
    # ax[0].legend()
    ax[0].set_xlabel("Angle Error (°)")

    ax[1].hist(
        amplitude_errors,
        bins=bins,
        label=f"Avg Error: {round(mean_angle_baseline, 2)}°",
        color=colors_[1],
        edgecolor="black",
    )
    ax[1].set_xlabel("Amplitude Error (%)")
    plt.tight_layout()

    return fig, ax


def plot_histograms(X, B, model, save_path, figsize=(8, 8), tag=""):
    def get_bins(data):
        data_min = min(data)
        data_max = max(data)

        # Define your desired bar width
        bar_width = 0.02  # Set this to whatever width you want

        # Calculate the number of bins needed based on the width
        num_bins = int(np.ceil((data_max - data_min) / bar_width))

        # Create bins with fixed width
        bins = np.linspace(data_min, data_min + num_bins * bar_width, num_bins + 1)
        return bins

    angle_errors_baseline, amplitude_errors_baseline = [], []

    for Bi in B:
        angle_error, amp_error = calculate_metrics_baseline(Bi)
        angle_errors_baseline.append(torch.mean(angle_error))
        amplitude_errors_baseline.append(torch.mean(amp_error))

    angle_errors, amplitude_errors = [], []

    for Xi, Bi in zip(X, B):
        angle_error, amp_error = calculate_metrics_trained(Xi, Bi, model)
        angle_errors.append(torch.nan_to_num(torch.mean(angle_error), nan=180.0))
        amplitude_errors.append(torch.mean(amp_error))

    ax: list[Axes]
    fig, ax = plt.subplots(
        ncols=2,
        nrows=1,
        figsize=figsize,
        sharey=True,
    )

    mean_angle_baseline = round(float(np.mean(angle_errors_baseline)), 4)
    mean_amp_baseline = round(float(np.mean(amplitude_errors_baseline)), 4)
    mean_angle = round(float(np.mean(angle_errors)), 4)
    mean_amp = round(float(np.mean(amplitude_errors)), 4)

    # ax[0][0].hist(
    #     angle_errors_baseline,
    #     bins=get_bins(angle_errors_baseline),
    #     label=f"Avg Error: {mean_angle_baseline}°",
    # )
    # ax[0][0].set_ylabel("Count (Analytical Solution)")
    # # ax[0][0].legend()

    # ax[0][1].hist(
    #     amplitude_errors_baseline,
    #     bins=get_bins(amplitude_errors_baseline),
    #     label=f"Avg Error: {mean_amp_baseline}%",
    # )
    # # ax[0][1].legend()

    count, bins, _ = ax[0].hist(
        angle_errors,
        # bins=get_bins(angle_errors),
        bins=30,
        label=f"Avg Error: {mean_angle}°",
        edgecolor="black",
        color=colors_[1],
    )
    # kde = gaussian_kde(angle_errors)
    # x_vals = np.linspace(min(angle_errors), max(angle_errors), 1000)
    # ax[0].plot(x_vals, kde(x_vals))
    ax[0].set_xlabel("Angle Error (°)")
    ax[0].set_ylabel("Frequency")

    count, bins, _ = ax[1].hist(
        amplitude_errors,
        # bins=get_bins(amplitude_errors),
        bins=30,
        label=f"Avg Error: {mean_amp}%",
        edgecolor="black",
        color=colors_[1],
    )
    # kde = gaussian_kde(amplitude_errors)
    # x_vals = np.linspace(min(amplitude_errors), max(amplitude_errors), 1000)
    # ax[1].plot(x_vals, kde(x_vals))
    ax[1].set_xlabel("Amplitude Error (%)")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(f"{save_path}/histograms{tag}.pdf", format="pdf")
    else:
        plt.show()


def plot_histograms_with_baseline(X, B, model, save_path, figsize=(8, 8), tag=""):
    def get_bins(data):
        data_min = min(data)
        data_max = max(data)

        # Define your desired bar width
        bar_width = 0.02  # Set this to whatever width you want

        # Calculate the number of bins needed based on the width
        num_bins = int(np.ceil((data_max - data_min) / bar_width))

        # Create bins with fixed width
        bins = np.linspace(data_min, data_min + num_bins * bar_width, num_bins + 1)
        return bins

    angle_errors_baseline, amplitude_errors_baseline = [], []

    for Bi in B:
        angle_error, amp_error = calculate_metrics_baseline(Bi)
        angle_errors_baseline.append(torch.max(angle_error))
        amplitude_errors_baseline.append(torch.max(amp_error))

    angle_errors, amplitude_errors = [], []

    for Xi, Bi in zip(X, B):
        angle_error, amp_error = calculate_metrics_trained(Xi, Bi, model)
        angle_errors.append(torch.nan_to_num(torch.max(angle_error), nan=180.0))
        amplitude_errors.append(torch.max(amp_error))

    mean_angle_baseline = round(float(np.mean(angle_errors_baseline)), 4)
    mean_amp_baseline = round(float(np.mean(amplitude_errors_baseline)), 4)
    mean_angle = round(float(np.mean(angle_errors)), 4)
    mean_amp = round(float(np.mean(amplitude_errors)), 4)

    ax: list[list[Axes]]
    fig, ax = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=figsize,
        sharey="col",
        sharex="col",
    )

    # |-----|-----|
    # |(0,0)|(0,1)|
    # |-----|-----|
    # |(1,0)|(1,1)|
    # |-----|-----|

    # baseline angle
    ax[0][0].hist(
        angle_errors_baseline,
        # bins=get_bins(angle_errors_baseline),
        bins=30,
        label=f"Avg Error: {mean_angle_baseline}°",
        # edgecolor="black",
        color=colors_[1],
    )

    # kde = gaussian_kde(angle_errors_baseline)
    # x_vals = np.linspace(
    #     min(angle_errors_baseline),
    #     max(angle_errors_baseline),
    #     1000,
    # )
    # ax[0][0].plot(
    #     x_vals,
    #     kde(x_vals),
    #     label=f"Avg Error: {mean_angle_baseline}°",
    #     color=colors_[1],
    # )

    ax[0][0].set_ylabel("Count (Analytical Solution)")
    ax[0][0].legend()

    # baseline amplitude

    ax[0][1].hist(
        amplitude_errors_baseline,
        # bins=get_bins(amplitude_errors_baseline),
        bins=30,
        label=f"Avg Error: {mean_amp_baseline}%",
        # edgecolor="black",
        color=colors_[1],
    )

    # kde = gaussian_kde(amplitude_errors_baseline)
    # x_vals = np.linspace(
    #     min(amplitude_errors_baseline),
    #     max(amplitude_errors_baseline),
    #     1000,
    # )
    # ax[0][1].plot(
    #     x_vals,
    #     kde(x_vals),
    #     label=f"Avg Error: {mean_amp_baseline}%",
    #     color=colors_[1],
    # )
    ax[0][1].legend()

    ax[1][0].hist(
        angle_errors,
        # bins=get_bins(angle_errors),
        bins=30,
        label=f"Avg Error: {mean_angle}°",
        # edgecolor="black",
        color=colors_[1],
    )
    # kde = gaussian_kde(angle_errors)
    # x_vals = np.linspace(min(angle_errors), max(angle_errors), 1000)
    # ax[1][0].plot(
    #     x_vals,
    #     kde(x_vals),
    #     label=f"Avg Error: {mean_angle}°",
    #     color=colors_[1],
    # )
    ax[1][0].legend()
    ax[1][0].set_xlabel("Angle Error (°)")
    ax[1][0].set_ylabel("Count (NN)")

    ax[1][1].hist(
        amplitude_errors,
        # bins=get_bins(amplitude_errors),
        bins=30,
        label=f"Avg Error: {mean_amp}%",
        # edgecolor="black",
        color=colors_[1],
    )
    # kde = gaussian_kde(amplitude_errors)
    # x_vals = np.linspace(min(amplitude_errors), max(amplitude_errors), 1000)
    # ax[1][1].plot(
    #     x_vals,
    #     kde(x_vals),
    #     label=f"Avg Error: {mean_amp}°",
    #     color=colors_[1],
    # )
    ax[1][1].set_xlabel("Amplitude Error (%)")
    ax[1][1].legend()

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(f"{save_path}/histograms{tag}.pdf", format="pdf")
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

    amplitude_errors_trained_slice = amplitude_errors_trained[mask]
    amplitude_errors_baseline_slice = amplitude_errors_baseline[mask]

    x_bins = np.linspace(min(x_slice), max(x_slice), 25)
    z_bins = np.linspace(min(z_slice), max(z_slice), 25)

    vmin, vmax = -10, 10
    linthresh, linscale = 0.1, 1.0

    norm = colors.SymLogNorm(
        vmin=vmin,
        vmax=vmax,
        linthresh=linthresh,
        linscale=linscale,
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

    axs: Axes
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

    mesh = axs.pcolormesh(
        x_edges,
        z_edges,
        heatmap_amplitude_trained.T,
        shading="auto",
        cmap=plt.cm.get_cmap("RdBu_r"),
        norm=norm,
    )

    axs.set_xlabel("X (a.u.)")
    axs.set_ylabel("Z (a.u.)")
    axs.set_xlim((0, a * 2.5))
    axs.set_ylim((0, 2.5))

    axs.add_patch(
        patches.Rectangle(
            (0, 0),
            width=a / 2 + eps_x,
            height=1 / 2 + eps_y,
            linewidth=2,
            edgecolor="k",
            facecolor="white",
        )
    )

    cbar = fig.colorbar(mesh, ax=axs, location="bottom")  # type: ignore
    tick_locations = [-10, -1, -0.1, 0, 0.1, 1, 10]

    cbar.set_ticks(tick_locations)
    cbar.set_ticklabels([f"{x:.1f}" for x in tick_locations])

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

    vmin, vmax = 0, 10
    linthresh, linscale = 0.1, 1.0

    angle_errors_trained_slice = angle_errors_trained[mask]
    angle_errors_baseline_slice = angle_errors_baseline[mask]

    norm = colors.SymLogNorm(
        vmin=vmin,
        vmax=vmax,
        linthresh=linthresh,
        linscale=linscale,
    )

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
    heatmap_counts_angle_baseline, _, _ = np.histogram2d(
        x_slice, z_slice, bins=[x_bins, z_bins]
    )

    heatmap_angle_baseline = np.divide(
        heatmap_angle_baseline,
        heatmap_counts_angle_baseline,
        where=heatmap_counts_angle_baseline != 0,
    )

    axs: Axes
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

    mesh = axs.pcolormesh(
        x_edges,
        z_edges,
        heatmap_angle_trained.T,
        shading="auto",
        cmap=plt.get_cmap("Reds"),
        norm=norm,
    )

    axs.set_xlabel("X (a.u.)")
    axs.set_ylabel("Z (a.u.)")
    axs.set_xlim((0, a * 2.5))
    axs.set_ylim((0, 2.5))

    axs.add_patch(
        patches.Rectangle(
            (0, 0),
            width=a / 2 + eps_x,
            height=1 / 2 + eps_y,
            linewidth=2,
            edgecolor="k",
            facecolor="white",
        )
    )

    cbar = fig.colorbar(mesh, ax=axs, location="bottom")  # type: ignore
    tick_locations = [0.01, 0.1, 1, 10]

    cbar.set_ticks(tick_locations)
    cbar.set_ticklabels([f"{x:.1f}" for x in tick_locations])
    cbar.set_label("Angle Error (°)")

    return fig, axs


def plot_heatmaps(
    model: torch.nn.Module,
    X: torch.Tensor,
    B: torch.Tensor,
    save_path: str | Path | None,
    tag: str = "",
):
    grid = X[:, 5:]
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
        grid=grid.numpy(),
        amplitude_errors_baseline=amplitude_errors_baseline.numpy(),
        amplitude_errors_trained=amplitude_errors_trained.numpy(),
        a=a,
        b=b,
    )

    if save_path is not None:
        fig1.savefig(f"{save_path}/amplitude_heatmap{tag}.pdf", format="pdf")
    else:
        plt.show()

    fig2, _ = plot_heatmaps_angle(
        grid=grid.numpy(),
        angle_errors_baseline=angle_errors_baseline.numpy(),
        angle_errors_trained=angle_errors_trained.numpy(),
        a=a,
        b=b,
    )

    if save_path is not None:
        fig2.savefig(f"{save_path}/angle_heatmap{tag}.pdf", format="pdf")
    else:
        plt.show()


def plot_times(batch_sizes, times_ana, times_demag, times_nn):
    assert len(times_ana) == len(times_demag) == len(times_nn)

    plt.plot(batch_sizes, times_ana, label="Analytical solution")
    plt.plot(batch_sizes, times_demag, label="Full Solution")
    plt.plot(batch_sizes, times_nn, label="NN Solution")
    plt.legend()
    plt.show()

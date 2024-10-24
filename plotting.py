import json
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


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


def plot_histograms(stats: Dict, save_path: str):
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

    with open(f"{save_path}/val_stats.json", "w+") as f:
        json.dump(stats, f)

    plt.savefig(f"{save_path}/hist.png", format="png")

    plt.show()

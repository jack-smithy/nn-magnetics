from typing import Dict

import matplotlib.pyplot as plt


def plot(stats: Dict, save_path: str | None = None):
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
        plt.savefig(save_path, format="png")

    plt.show()

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal
import wandb
from wakepy import keep

from nn_magnetics import (
    train_isotropic,
    evaluate_isotropic,
    train_anisotropic,
    evaluate_anisotropic,
)


@dataclass
class ModelConfig:
    """Configuration for the ML model and training process."""

    learning_rate: float
    architecture: Literal["MLP"]
    hidden_dim_factor: int
    activation: Literal["SiLU"]
    loss_fn: Literal["field", "correction", "amplitude"]
    data_dir: str
    epochs: int
    batch_size: int
    gamma: float
    save_path: str


def setup_wandb(project_name: str, config: ModelConfig) -> None:
    """Initialize Weights & Biases tracking."""
    wandb.init(project=project_name, config=config.__dict__)
    if wandb.run is not None:
        run_specific_path = f"results/{project_name}/{wandb.run.name}"
        wandb.config.update({"save_path": wandb.run.name}, allow_val_change=True)
        config.save_path = run_specific_path


def run_experiment(config: ModelConfig) -> None:
    """Run the training and evaluation pipeline."""
    train_isotropic(config.__dict__)
    evaluate_isotropic(config.__dict__)


def main() -> None:
    project_name = "isotropic_chi_amplitude_correction"
    timestamp = str(datetime.datetime.now())
    config = ModelConfig(
        learning_rate=0.001,
        architecture="MLP",
        hidden_dim_factor=6,
        activation="SiLU",
        loss_fn="amplitude",
        data_dir="data/isotropic_chi",
        epochs=30,
        batch_size=128,
        gamma=0.95,
        save_path=f"results/{project_name}/{timestamp}",
    )

    setup_wandb(project_name, config)
    run_experiment(config)


if __name__ == "__main__":
    with keep.running():
        main()

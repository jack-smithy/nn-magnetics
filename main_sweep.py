import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import torch
import wandb
from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import QuaternionNet

DEVICE = "cpu"

sweep_config = {
    "method": "bayes",
    "metric": {"name": "validation_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"min": 64, "max": 2048},
        "gamma": {"min": 0.9, "max": 1.0},
        "loss": {"values": ["mse", "l1"]},
        "activation": {"values": ["silu", "tanh"]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="3dof-chi-quaternions")
activations = {"silu": F.silu, "tanh": F.tanh}
losses = {"l1": F.l1_loss, "mse": F.mse_loss}


def train():
    wandb.init()

    assert wandb.run is not None

    train_data = AnisotropicData("data/3dof_chi/train_fast", device=DEVICE)
    valid_data = AnisotropicData("data/3dof_chi/validation_fast", device=DEVICE)

    train_loader = DataLoader(
        train_data,
        batch_size=wandb.config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=wandb.config.batch_size,
        shuffle=False,
    )

    model = QuaternionNet(
        in_features=8,
        hidden_dim_factor=6,
        activation=activations[wandb.config.activation],
        save_weights=False,
        do_output_activation=False,
    ).to(torch.float64)

    loss = losses[wandb.config.loss]

    optimizer = Adam(
        params=model.parameters(),
        lr=wandb.config.learning_rate,
    )

    model.lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=wandb.config.gamma)

    _ = model.fit(
        train_loader,
        valid_loader,
        loss,
        optimizer,
        15,
    )

    wandb.finish()


if __name__ == "__main__":
    # wandb.agent(
    #     "zgwihfd6",
    #     function=train,
    #     count=100,
    #     entity="jack-smithy-university-of-vienna",
    #     project="3dof-chi-quaternions",
    # )

    wandb.agent(sweep_id=sweep_id, function=train)

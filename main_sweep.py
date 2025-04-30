import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import torch
import wandb
from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import AngleAmpCorrectionNetwork

DEVICE = "cpu"

sweep_config = {
    "method": "bayes",
    "metric": {"name": "validation_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"value": 256, "distribution": "constant"},
        "gamma": {"min": 0.9, "max": 1.0},
        "loss": {"value": "l1", "distribution": "constant"},
        "activation": {"value": "silu", "distribution": "constant"},
        "epochs": {"value": 15, "distribution": "constant"},
        "lr_scheduler": {"value": "cosine"},
    },
}

# sweep_id = wandb.sweep(sweep_config, project="3dof_chi_v2")
activations = {"silu": F.silu, "tanh": F.tanh}
losses = {"l1": F.l1_loss, "mse": F.mse_loss}


def train():
    wandb.init()

    assert wandb.run is not None

    train_data = AnisotropicData("data/3dof_chi_v2/train", device=DEVICE)
    valid_data = AnisotropicData("data/3dof_chi_v2/validation", device=DEVICE)

    train_loader = DataLoader(
        train_data,
        batch_size=int(wandb.config.batch_size),
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=int(wandb.config.batch_size),
        shuffle=False,
    )

    model = AngleAmpCorrectionNetwork(
        activation=activations[wandb.config.activation],
        save_weights=False,
        save_path=None,
    ).to(torch.float64)

    optimizer = Adam(
        params=model.parameters(),
        lr=wandb.config.learning_rate,
    )

    model.lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=wandb.config.gamma)

    _ = model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=losses[wandb.config.loss],
        optimizer=Adam(params=model.parameters(), lr=wandb.config.learning_rate),
        epochs=wandb.config.epochs,
    )

    wandb.finish()


if __name__ == "__main__":
    wandb.agent(
        "n3earp9a",
        function=train,
        count=100,
        entity="jack-smithy-university-of-vienna",
        project="3dof_chi_v2",
    )

    # wandb.agent(sweep_id=sweep_id, function=train)

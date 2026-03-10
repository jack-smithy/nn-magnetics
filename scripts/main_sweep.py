import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import torch
import wandb
from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import (
    AngleAmpCorrectionNetwork,
    QuaternionNet,
    FieldCorrectionNetwork,
    SphericalCorrectionNetwork,
)

DEVICE = "cpu"
DTYPE = torch.float32

sweep_config = {
    "method": "bayes",
    "metric": {"name": "validation/loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [0.00001, 0.0001, 0.001, 0.01]},
        "batch_size": {"values": [512, 1024, 2048, 4096, 8192]},
        "gamma": {"min": 0.9, "max": 1.0},
        "p": {"min": 0.0, "max": 0.2},
        "weight_decay": {"min": 0.0, "max": 0.05},
        "epochs": {"value": 20},
    },
}

activations = {
    "silu": F.silu,
    "tanh": F.tanh,
    "gelu": F.gelu,
    "sigmoid": F.sigmoid,
}
network = {
    "euler": AngleAmpCorrectionNetwork,
    "field": FieldCorrectionNetwork,
    "quaternion": QuaternionNet,
    "spherical": SphericalCorrectionNetwork,
}

sweep_id = wandb.sweep(sweep_config, project="3dof_chi_spherical")


def train():
    wandb.init()

    assert wandb.run is not None

    train_data = AnisotropicData(
        "data/3dof_chi_v2/train_fast", device=DEVICE, dtype=DTYPE
    )
    valid_data = AnisotropicData(
        "data/3dof_chi_v2/validation_fast", device=DEVICE, dtype=DTYPE
    )

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

    model = SphericalCorrectionNetwork(
        activation=F.silu,
        save_weights=False,
        save_path=None,
        p=wandb.config.p,
    ).to(DTYPE)

    optimizer = Adam(
        params=model.parameters(),
        lr=wandb.config.learning_rate,
    )

    model.lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=wandb.config.gamma)

    _ = model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=F.l1_loss,
        optimizer=optimizer,
        epochs=wandb.config.epochs,
    )

    wandb.finish()


if __name__ == "__main__":
    # wandb.agent(
    #     "cepvj27b",
    #     function=train,
    #     count=100,
    #     entity="jack-smithy-university-of-vienna",
    #     project="3dof_chi_v2_large",
    # )

    wandb.agent(sweep_id=sweep_id, function=train)

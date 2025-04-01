import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import torch
import wandb
from nn_magnetics.data import AnisotropicData
from nn_magnetics.models import FieldCorrectionNetwork
from nn_magnetics.models.utils import DivergenceLoss

DEVICE = "cpu"

sweep_config = {
    "method": "bayes",
    "metric": {"name": "validation_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"min": 64, "max": 512},
        "gamma": {"min": 0.9, "max": 1.0},
        "lambda_": {"min": 0.0, "max": 4.0},
    },
}

sweep_id = wandb.sweep(sweep_config, project="3dof-chi-pinn")


def main():
    wandb.init()

    assert wandb.run is not None

    train_data = AnisotropicData("data/3dof_chi_v2/train_fast", device=DEVICE)
    valid_data = AnisotropicData("data/3dof_chi_v2/validation_fast", device=DEVICE)

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

    model = FieldCorrectionNetwork(
        in_features=8,
        hidden_dim_factor=6,
        activation=F.silu,
        save_weights=False,
    ).to(torch.float64)

    loss = DivergenceLoss(lambda_=wandb.config.lambda_)

    optimizer = Adam(
        params=model.parameters(),
        lr=wandb.config.learning_rate,
        fused=True,
    )

    model.lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=wandb.config.gamma)

    _ = model.fit(
        train_loader,
        valid_loader,
        loss,
        optimizer,
        10,
    )

    wandb.finish()


if __name__ == "__main__":
    # wandb.agent(
    #     "kh16q5b4",
    #     function=main,
    #     count=100,
    #     entity="jack-smithy-university-of-vienna",
    #     project="3dof-chi-pinn",
    # )

    wandb.agent(sweep_id=sweep_id, function=main)

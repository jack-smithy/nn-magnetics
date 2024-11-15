from typing import Type, Literal

import torch
import torch.nn.functional as F
from torch import nn


class Network(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_dim_factor,
        out_features,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(
            in_features=in_features,
            out_features=4 * hidden_dim_factor,
        )
        self.linear2 = nn.Linear(
            in_features=4 * hidden_dim_factor,
            out_features=8 * hidden_dim_factor,
        )
        self.linear3 = nn.Linear(
            in_features=8 * hidden_dim_factor,
            out_features=4 * hidden_dim_factor,
        )
        self.linear4 = nn.Linear(
            in_features=4 * hidden_dim_factor,
            out_features=2 * hidden_dim_factor,
        )
        self.linear5 = nn.Linear(
            in_features=2 * hidden_dim_factor,
            out_features=hidden_dim_factor,
        )
        self.output = nn.Linear(
            in_features=1 * hidden_dim_factor,
            out_features=out_features,
        )
        self.activation = kwargs.get("activation", F.silu)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = self.activation(self.linear5(x))
        return self.activation(self.output(x))


class CorrectionLoss(nn.Module):
    def __init__(self, loss: Type[nn.Module] = nn.L1Loss) -> None:
        super().__init__()
        self.loss = loss()

    def forward(self, B, B_pred):
        B_demag, B_reduced = B[..., :3], B[..., 3:]
        return self.loss(B_demag / B_reduced, B_pred)


class FieldLoss(nn.Module):
    def __init__(self, loss: Type[nn.Module] = nn.L1Loss) -> None:
        super().__init__()
        self.loss = loss()

    def forward(self, B, B_pred):
        B_demag, B_reduced = B[..., :3], B[..., 3:]
        return self.loss(B_demag, B_pred * B_reduced)


def get_loss(name: Literal["field", "correction"]) -> Type[nn.Module]:
    if name == "field":
        return FieldLoss
    elif name == "correction":
        return CorrectionLoss
    else:
        raise ValueError(f"Invalid loss function: {name}")


def get_num_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return (
        sum(p.numel() for p in model.parameters())
        if trainable_only
        else sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

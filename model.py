import torch.nn.functional as F
from torch import nn
import torch
from typing import Type

torch.set_default_dtype(torch.float64)


class _Network(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear4 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear5 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear6 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear7 = nn.Linear(in_features=hidden_dim, out_features=out_features)

    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = F.silu(self.linear2(x))
        x = F.silu(self.linear3(x))
        x = F.silu(self.linear4(x))
        x = F.silu(self.linear5(x))
        x = F.silu(self.linear6(x))
        return self.linear7(x)


class Network(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_dim_factor,
        out_features,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

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
        self.activation = kwargs.get("activation", F.relu)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = self.activation(self.linear5(x))
        return self.output(x)


model_original = nn.Sequential(
    nn.Linear(7, 24),
    nn.ReLU(),
    nn.Linear(24, 48),
    nn.ReLU(),
    nn.Linear(48, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 3),
)


class PhysicsLoss(nn.Module):
    def __init__(self, loss: Type[nn.Module] = nn.L1Loss) -> None:
        super().__init__()
        self.loss = loss()

    def forward(self, B, B_pred):
        B_demag = B[..., :3]
        B_ana = B[..., 3:]

        return self.loss(B_demag, torch.mul(B_ana, B_pred))

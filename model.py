import torch.nn.functional as F
from torch import nn
import torch

torch.set_default_dtype(torch.float64)


class Network(nn.Module):
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


model_original = nn.Sequential(
    nn.Linear(6, 24),
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

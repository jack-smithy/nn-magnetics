import torch.nn.functional as F
from torch import nn


class Network(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear3 = nn.Linear(in_features=hidden_dim, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

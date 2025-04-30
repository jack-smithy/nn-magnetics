import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset


# Load the state dictionary without loading it into a model
state_dict = torch.load(
    "/Users/jacksmith/Documents/work/nn-magnetics/results/3dof_chi_v2/2025-04-30 10:56:20.115777/best_weights.pt"
)

# Print all keys and their shapes
print("State dictionary contents:")
for key, value in state_dict.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"{key}: {type(value)}")

# Random data
x = torch.randn(100, 10)
y = torch.randn(100, 1)


class MockData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


data = MockData(x, y)

loader = DataLoader(dataset=data, batch_size=10)

# Simple model
model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

# Training loop
for epoch in range(100):
    losses = []
    for xi, yi in loader:
        y_pred = model(xi)
        loss = criterion(y_pred, yi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)

    scheduler.step()
    epoch_loss = torch.stack(losses)

    if epoch % 10 == 0:
        print(scheduler.get_lr())
        print(f"Epoch {epoch}, Loss: {epoch_loss.mean().item():.4f}")

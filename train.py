import statistics

import matplotlib.pyplot as plt
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from dataset import DemagData, make_train_test_split
from model import Network
from utils import get_device

DEVICE = get_device()

stats = {
    "train_loss": [],
    "test_loss": [],
}

X_train, X_test, y_train, y_test = make_train_test_split("./data")

train_dataset = DemagData(X=X_train, y=y_train, device=DEVICE)
test_dataset = DemagData(X=X_test, y=y_test, device=DEVICE)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32)

model = Network(
    in_features=X_train.shape[1],
    out_features=y_train.shape[1],
    hidden_dim=64,
).to(DEVICE)

criterion = nn.MSELoss()
opt = Adam(params=model.parameters(), lr=1e-5)

EPOCHS = 3

for ep in range(EPOCHS):
    epoch_train_loss = []
    epoch_test_loss = []

    model.train()
    for X, y in train_dataloader:
        y_pred = model(X)
        loss = criterion(y, y_pred)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_train_loss.append(float(loss.cpu()))

    model.eval()
    for X, y in test_dataloader:
        y_pred = model(X)
        loss = criterion(y, y_pred)
        epoch_test_loss.append(float(loss.cpu()))

    stats["test_loss"].append(statistics.mean(epoch_test_loss))
    stats["train_loss"].append(statistics.mean(epoch_train_loss))

plt.plot(stats["train_loss"], label="Train")
plt.plot(stats["test_loss"], label="Test")
plt.legend()
plt.show()

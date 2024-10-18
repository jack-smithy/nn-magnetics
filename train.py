import datetime
import json
from typing import Tuple
import os
import numpy as np
import torch
import tqdm
from torch import nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from dataset import DemagData, make_train_test_split
from metrics import (
    angle_error_torch,
    relative_amplitude_error_torch,
)
from model import model_original
from plotting import plot
from utils import get_device

DEVICE = get_device(use_accelerators=False)
EPOCHS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
SAVE_PATH = f"results/{str(datetime.datetime.now())}"

STATS = {
    "train_loss": [],
    "test_loss": [],
    "angle_error": [],
    "amplitude_error": [],
    "params": {
        "lr": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.L1Loss,
    optimizer: Optimizer,
) -> float:
    model.train()
    batch_losses = []

    loss = torch.inf
    for X, y in dataloader:
        y_pred = model(X)
        loss = criterion(y, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.numpy(force=True))

    return np.mean(batch_losses, axis=-1)


def test_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.L1Loss,
) -> Tuple[float, float, float]:
    model.eval()
    batch_losses = []
    batch_angle_errors = []
    batch_amp_errors = []
    for X, y in dataloader:
        with torch.no_grad():
            y_pred = model(X)

            batch_losses.append(criterion(y, y_pred).numpy(force=True))
            batch_angle_errors.append(angle_error_torch(y, y_pred))
            batch_amp_errors.append(relative_amplitude_error_torch(y, y_pred))

    return (
        np.mean(batch_losses, axis=-1),
        np.mean(batch_angle_errors, axis=-1),
        np.mean(batch_amp_errors, axis=-1),
    )


X_train, X_test, y_train, y_test = make_train_test_split("./data")

train_dataset = DemagData(X=X_train, y=y_train, device=DEVICE)
test_dataset = DemagData(X=X_test, y=y_test, device=DEVICE)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.L1Loss()

opt = Adam(params=model_original.parameters(), lr=LEARNING_RATE)


for ep in tqdm.tqdm(range(EPOCHS), unit="epoch"):
    epoch_train_losses = train_one_epoch(
        model=model_original,
        dataloader=train_dataloader,
        criterion=criterion,
        optimizer=opt,
    )

    epoch_test_losses, epoch_angle_errors, epoch_amp_errors = test_one_epoch(
        model=model_original,
        dataloader=test_dataloader,
        criterion=criterion,
    )

    STATS["test_loss"].append(epoch_test_losses)
    STATS["train_loss"].append(epoch_train_losses)
    STATS["angle_error"].append(epoch_angle_errors)
    STATS["amplitude_error"].append(epoch_amp_errors)

os.makedirs(SAVE_PATH)

with open(f"{SAVE_PATH}/stats.json", "w+") as f:
    json.dump(STATS, f)

plot(STATS, f"{SAVE_PATH}/loss.png")

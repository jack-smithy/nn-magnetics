import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from dataset import ChiMode, DemagData, get_data
from model import Network, PhysicsLoss
from plotting import plot_histograms, plot_loss
from train import test_one_epoch, train_one_epoch, validate
from utils import get_device

torch.manual_seed(0)
np.random.seed(0)

DEVICE = get_device(use_accelerators=False)
SAVE_PATH = f"results/{str(datetime.datetime.now())}"
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 20


def main():
    training_stats = {
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

    X, B = get_data(Path("data/isotropic_chi"), ChiMode.ISOTROPIC)

    data = train_test_split(X, B, test_size=0.5, shuffle=True)
    original_shape_data = data.copy()

    X_train, X_test, B_train, B_test = [d.reshape(-1, 6) for d in data]

    train_dataloader = DataLoader(
        dataset=DemagData(X=X_train, y=B_train, device=DEVICE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset=DemagData(X=X_test, y=B_test, device=DEVICE),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    criterion = PhysicsLoss()

    model = Network(
        in_features=X_train.shape[1],
        hidden_dim_factor=6,
        out_features=3,
    ).to(DEVICE)

    opt = Adam(params=model.parameters(), lr=LEARNING_RATE)

    best_loss = np.inf

    for _ in tqdm.tqdm(range(EPOCHS), unit="epoch", disable=False):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=opt,
        )

        test_loss = test_one_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
        )

        training_stats["test_loss"].append(test_loss)
        training_stats["train_loss"].append(train_loss)
        training_stats["angle_error"].append(0)
        training_stats["amplitude_error"].append(0)

        if test_loss < best_loss:
            best_loss = test_loss

    plot_loss(training_stats, SAVE_PATH, show_plot=True)

    val_stats = validate(original_shape_data, model, criterion)
    plot_histograms(val_stats, save_path=SAVE_PATH)


if __name__ == "__main__":
    main()

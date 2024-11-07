import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
import copy
import wandb
from dataset import ChiMode, DemagData, get_data_parallel
from model import Network, PhysicsLoss
from plotting import plot_histograms, plot_loss
from train import test_one_epoch, train_one_epoch, validate
from utils import get_device

torch.manual_seed(0)
np.random.seed(0)


def main(epochs, batch_size, learning_rate, save_path, log=False):
    training_stats = {
        "train_loss": [],
        "test_loss": [],
        "angle_error": [],
        "amplitude_error": [],
        "params": {
            "lr": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        },
    }

    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="test-isotropic",
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": "MLP",
                "dataset": "data/",
                "epochs": epochs,
                "batch_size": batch_size,
            },
        )

    X, B = get_data_parallel(Path("data/isotropic_chi_v2"), ChiMode.ISOTROPIC)

    data = train_test_split(X, B, test_size=0.5, shuffle=True)
    original_shape_data = data.copy()

    X_train, X_test, B_train, B_test = data

    X_train = X_train.reshape(-1, 6)
    X_test = X_test.reshape(-1, 6)
    B_train = B_train.reshape(-1, 6)
    B_test = B_test.reshape(-1, 6)

    train_dataloader = DataLoader(
        dataset=DemagData(X=X_train, y=B_train, device=DEVICE),
        batch_size=batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset=DemagData(X=X_test, y=B_test, device=DEVICE),
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = PhysicsLoss()

    model = Network(
        in_features=X_train.shape[1],
        hidden_dim_factor=6,
        out_features=3,
    )

    opt = Adam(params=model.parameters(), lr=learning_rate)

    best_weights = copy.deepcopy(model).state_dict()

    best_loss = np.inf
    for _ in tqdm.tqdm(range(epochs), unit="epoch", disable=False):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=opt,
        )

        test_loss, test_angle_err, test_amp_err = test_one_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
        )

        training_stats["test_loss"].append(test_loss)
        training_stats["train_loss"].append(train_loss)
        training_stats["angle_error"].append(test_angle_err)
        training_stats["amplitude_error"].append(test_amp_err)

        if log:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "angle_error": test_angle_err,
                    "amp_err": test_amp_err,
                }
            )

        if test_loss < best_loss:
            best_loss = test_loss
            best_weights = copy.deepcopy(model).state_dict()

    val_stats = validate(original_shape_data, model, criterion)

    plot_loss(training_stats, save_path, show_plot=False)
    plot_histograms(val_stats, save_path=save_path, show_plot=False)
    torch.save(best_weights, f"{save_path}/weights.pt")


if __name__ == "__main__":
    DEVICE = get_device(use_accelerators=False)
    EPOCHS = 50
    BATCH_SIZE = 96
    LEARNING_RATE = 1e-4
    SAVE_PATH = f"results/{str(datetime.datetime.now())}"

    main(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_path=SAVE_PATH,
        log=True,
    )

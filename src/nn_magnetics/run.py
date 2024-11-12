import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

import wandb
from nn_magnetics.dataset import ChiMode, DemagData, get_data_parallel
from nn_magnetics.model import FieldLoss, Network, CorrectionLoss
from nn_magnetics.plotting import (
    plot_heatmaps,
    plot_histograms,
    plot_loss,
)
from nn_magnetics.train import (
    test_one_epoch,
    train_one_epoch,
    validate,
)

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cpu"


def run(epochs, batch_size, learning_rate, data_dir, save_path, log=False):
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

    os.makedirs(save_path)

    X_train, B_train = get_data_parallel(f"{data_dir}/train", ChiMode.ISOTROPIC)
    X_test, B_test = get_data_parallel(f"{data_dir}/test", ChiMode.ISOTROPIC)

    train_dataloader = DataLoader(
        dataset=DemagData(
            X=X_train.reshape(-1, 6),
            y=B_train.reshape(-1, 6),
            device=DEVICE,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        dataset=DemagData(
            X=X_test.reshape(-1, 6),
            y=B_test.reshape(-1, 6),
            device=DEVICE,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = FieldLoss()

    model = Network(
        in_features=6,
        hidden_dim_factor=6,
        out_features=3,
        activation=F.silu,
    )

    if wandb.run is not None:
        wandb.watch(model, log="all")

    opt = Adam(params=model.parameters(), lr=learning_rate)

    for ep in tqdm.tqdm(range(epochs), unit="epoch", disable=False):
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

        if wandb.run is not None:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "angle_error": test_angle_err,
                    "amp_err": test_amp_err,
                }
            )

    val_stats = validate(
        X_test,
        B_test,
        model,
        criterion,
    )

    plot_loss(training_stats, save_path, show_plot=False)
    plot_histograms(val_stats, save_path=save_path, show_plot=False)
    plot_heatmaps(model=model, save_path=save_path, epoch="done")

    torch.save(model.state_dict(), f"{save_path}/weights.pt")

import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

import wandb
from nn_magnetics.dataset import ChiMode, DemagData, get_data_parallel, get_one_magnet
from nn_magnetics.model import Network, FieldLoss
from nn_magnetics.plotting import (
    plot_histograms,
    plot_loss,
    plot_heatmaps_amplitude,
    plot_heatmaps_angle,
)
from nn_magnetics.train import (
    test_one_epoch,
    train_one_epoch,
    validate,
    calculate_metrics,
    calculate_metrics_baseline,
)

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cpu"


def plot_heatmaps(model, save_path, epoch):
    X, B = get_one_magnet(
        chi_mode=ChiMode.ISOTROPIC,
        data=np.load("data/isotropic_chi/eval/data_1.npz"),
    )

    grid = X[:, 3:]
    a = float(X[0, 0])
    b = float(X[0, 1])
    chi = float(X[0, 2])

    with torch.no_grad():
        B_pred = model(torch.tensor(X))

    angle_errors_baseline, amplitude_errors_baseline = calculate_metrics_baseline(
        B=B,
        return_abs=False,
    )
    angle_errors_trained, amplitude_errors_trained = calculate_metrics(
        B=torch.tensor(B),
        B_pred=B_pred,
        return_abs=False,
    )

    plot_heatmaps_amplitude(
        grid=grid,
        amplitude_errors_baseline=amplitude_errors_baseline,
        amplitude_errors_trained=amplitude_errors_trained,
        a=a,
        b=b,
        chi=chi,
        epoch=epoch,
        save_path=save_path,
    )

    plot_heatmaps_angle(
        grid=grid,
        angle_errors_baseline=angle_errors_baseline,
        angle_errors_trained=angle_errors_trained,
        a=a,
        b=b,
        chi=chi,
        epoch=epoch,
        save_path=save_path,
    )


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

    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="test-isotropic-correction-loss",
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": "MLP-SiLU-double-width",
                "dataset": "data/",
                "epochs": epochs,
                "batch_size": batch_size,
            },
        )

    os.makedirs(save_path)

    X_train, B_train = get_data_parallel(f"{data_dir}/train_fast", ChiMode.ISOTROPIC)
    X_test, B_test = get_data_parallel(f"{data_dir}/test_fast", ChiMode.ISOTROPIC)

    train_dataloader = DataLoader(
        dataset=DemagData(
            X=X_train.reshape(-1, 6),
            y=B_train.reshape(-1, 6),
            device=DEVICE,
        ),
        batch_size=batch_size,
        shuffle=False,
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
        hidden_dim_factor=12,
        out_features=3,
        activation=F.silu,
    )

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

        if log:
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

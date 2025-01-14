import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from wakepy import keep

import wandb
from nn_magnetics.data.dataset import IsotropicData, AnisotropicData
from nn_magnetics.model import AngleAmpCorrectionNetwork, FieldCorrectionNetwork
from nn_magnetics.utils.plotting import (
    plot_heatmaps,
    plot_histograms,
    plot_loss,
)


def main():
    wandb.init(project="euler-correction")
    assert wandb.run is not None

    train_data = AnisotropicData("data/anisotropic_chi/train_fast", device="cpu")
    valid_data = AnisotropicData("data/anisotropic_chi/test_fast", device="cpu")
    print("Loaded data")

    train_loader = DataLoader(train_data, batch_size=64)
    valid_loader = DataLoader(valid_data, batch_size=64)

    model = AngleAmpCorrectionNetwork(
        in_features=7,
        hidden_dim_factor=6,
        out_features=4,
    ).to("cpu")

    loss = nn.L1Loss()
    optimizer = Adam(params=model.parameters(), lr=0.0001)
    epochs = 3

    print("Started training")
    train_losses, valid_losses, angle_errs, amp_errs = model.train_model(
        train_loader,
        valid_loader,
        loss,
        optimizer,
        epochs,
    )

    print("Finished training")
    fig, ax = plot_loss(train_losses, valid_losses, angle_errs, amp_errs, epochs)
    plt.show()

    print("Plotting eval histograms")
    X, B = valid_data.get_magnets()
    fig, ax = plot_histograms(X, B, model)
    plt.show()

    print("Plotting heatmaps")
    plot_heatmaps(model, X[0], B[0])


if __name__ == "__main__":
    with keep.running():
        main()

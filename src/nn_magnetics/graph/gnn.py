import pickle

import numpy as np
import torch
from torch.nn import L1Loss, Linear, MSELoss, ReLU, SiLU, Tanh
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GATConv, GCNConv, GraphNorm, Sequential

from nn_magnetics.data import simulate_demag
from nn_magnetics.utils.metrics import angle_error, relative_amplitude_error
from nn_magnetics.utils.plotting import plot_loss


class GCN(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(0)
        super().__init__()

        self.net = Sequential(
            "x, edge_index -> x",
            [
                (GCNConv(8, 32), "x, edge_index -> x"),
                SiLU(inplace=True),
                GraphNorm(32),
                (GCNConv(32, 64), "x, edge_index -> x"),
                SiLU(inplace=True),
                GraphNorm(64),
                Linear(64, 64),
                SiLU(inplace=True),
                Linear(64, 32),
                SiLU(inplace=True),
                Linear(32, 16),
                SiLU(inplace=True),
                Linear(16, 4),
            ],
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        return self.net(x, edge_index)


model = Sequential(
    "x, edge_index",
    [
        (GCNConv(8, 32), "x, edge_index -> x"),
        Tanh(),
        BatchNorm(32),
        (GCNConv(32, 64), "x, edge_index -> x"),
        Tanh(),
        (GCNConv(64, 128), "x, edge_index -> x"),
        Tanh(),
        BatchNorm(128),
        Linear(128, 128),
        Tanh(),
        Linear(128, 64),
        Tanh(),
        Linear(64, 16),
        Tanh(),
        Linear(16, 3),
    ],
)


def make_data(n):
    graphs = []
    for _ in range(n):
        a = np.random.rand()
        b = np.random.rand()
        chi = tuple(np.random.rand(3))

        data = simulate_demag(a, b, chi, 3, display=False, targel_elems=10)
        X, B, edge_index = make_feature_vector(data)

        graph = Data(x=X, y=B, edge_index=edge_index, dtype=torch.float)
        graphs.append(graph)

    loader = DataLoader(graphs)
    return loader


def print_dict(epoch, stats):
    angle = stats["angle_error"]
    amp = stats["amp_error"]
    train_loss = stats["train_loss"]
    test_loss = stats["test_loss"]

    print(
        f"\tEpoch {epoch}\nTrain loss: {train_loss}\nTest loss: {test_loss}\nAngle Error: {angle}\nAmp Error: {amp}\n\n"
    )


def main():
    # loader = make_data(200)
    # loader_test = make_data(200)

    # with open("loader.pkl", "wb") as f:
    #     pickle.dump(loader, f)

    # with open("loader_test.pkl", "wb") as f:
    #     pickle.dump(loader_test, f)

    with open("loader.pkl", "rb") as f:
        loader = pickle.load(f)

    with open("loader.pkl", "rb") as f:
        loader_test = pickle.load(f)

    loader = DataLoader(loader.dataset[:3])
    loader_test = DataLoader(loader_test.dataset[:3])

    # model = GCN()
    criterion = MSELoss()

    opt = Adam(params=model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer=opt, gamma=0.99)

    angle_error_baseline = []
    amp_error_baseline = []
    loss_baseline = []
    for batch in loader_test:
        y_true = batch.y[..., :3]
        ansatz = batch.y[..., 3:]

        true_corrections = y_true / ansatz

        loss = criterion(true_corrections, torch.ones_like(true_corrections))

        angle_errors_batch_baseline = angle_error(ansatz, y_true)
        amp_errors_batch_baseline = relative_amplitude_error(
            ansatz, y_true, return_abs=True
        )

        loss_baseline.append(loss.item())
        angle_error_baseline.append(torch.mean(angle_errors_batch_baseline).item())
        amp_error_baseline.append(torch.mean(amp_errors_batch_baseline).item())

    mean_angle_error = np.mean(angle_error_baseline)
    mean_amp_error = np.mean(amp_error_baseline)
    mean_loss = np.mean(loss_baseline)

    history = []
    epochs = 200
    for ep in range(1, epochs + 1):
        batch_losses_train = []
        batch_losses_test = []
        angle_errors = []
        amp_errors = []

        for batch in loader:
            correction = model(batch.x, batch.edge_index)

            y_true = batch.y[..., :3]
            ansatz = batch.y[..., 3:]

            # angles = correction[..., :3]
            # amplitudes = correction[..., 3]

            # R = batch_rotation_matrices(angles)

            # y_pred = amplitudes[:, None] * torch.einsum("nij,nj->ni", R, ansatz)

            true_corrections = y_true / ansatz

            loss = criterion(true_corrections, correction)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            batch_losses_train.append(loss.item())

        for batch in loader_test:
            correction = model(batch.x, batch.edge_index)

            y_true = batch.y[..., :3]
            ansatz = batch.y[..., 3:]

            # angles = correction[..., :3]
            # amplitudes = correction[..., 3]

            # R = batch_rotation_matrices(angles)

            # y_pred = amplitudes[:, None] * torch.einsum("nij,nj->ni", R, ansatz)

            true_corrections = y_true / ansatz

            loss = criterion(true_corrections, correction)

            y_pred = ansatz * correction

            angle_errors_batch = angle_error(y_pred, y_true)
            amp_errors_batch = relative_amplitude_error(y_pred, y_true, return_abs=True)

            angle_errors_batch_baseline = angle_error(ansatz, y_true)
            amp_errors_batch_baseline = relative_amplitude_error(
                ansatz, y_true, return_abs=True
            )

            angle_errors.append(torch.mean(angle_errors_batch).item())
            amp_errors.append(torch.mean(amp_errors_batch).item())
            batch_losses_test.append(loss.item())

        hist = {
            "train_loss": np.mean(batch_losses_train),
            "test_loss": np.mean(batch_losses_test),
            "angle_error": np.mean(angle_errors),
            "amp_error": np.mean(amp_errors),
        }

        history.append(hist)

        print_dict(ep, hist)

    print(true_corrections[:5])
    print(correction[:5])

    plot_loss(
        [h["train_loss"] for h in history],
        [h["test_loss"] for h in history],
        [h["angle_error"] for h in history],
        [h["amp_error"] for h in history],
        n_epochs=epochs,
        save_path=None,
        baselines=(mean_loss, mean_angle_error, mean_amp_error),
    )


if __name__ == "__main__":
    main()

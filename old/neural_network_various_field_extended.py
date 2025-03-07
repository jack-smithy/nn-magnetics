import copy
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
import tqdm
from sklearn.model_selection import train_test_split
from metrics import relative_amplitude_error, angle_error


def calculation(data_range, mode):
    """
    type = 'magnetization', 'magnetization_extended', 'field', 'field_extended'
    """
    data = np.load("data/%d.npz" % data_range)

    if mode == "magnetization":
        cell_pos = data["cell_pos"]
        length = len(cell_pos)

        input_data_new = np.vstack(
            (
                np.ones(length) * data["a"],
                np.ones(length) * data["b"],
                np.ones(length) * data["chi"],
                cell_pos[:, 0] / data["a"],
                cell_pos[:, 1] / data["b"],
                cell_pos[:, 2],
            )
        ).T
        output_data_new = data["cell_field"] - 4 * np.pi / 10 * data["cell_H"]
    elif mode == "magnetization_extended":
        cell_pos = data["cell_pos"]
        length = len(cell_pos)

        input_data_new = np.vstack(
            (
                np.ones(length) * data["a"],
                np.ones(length) * data["b"],
                np.ones(length) * data["chi"],
                cell_pos[:, 0] / data["a"],
                cell_pos[:, 1] / data["b"],
                cell_pos[:, 2],
            )
        ).T
        output_data_new = data["magnetization_reduced"]
    elif mode == "field":
        grid = data["grid"]
        length = len(grid)

        input_data_new = np.vstack(
            (
                np.ones(length) * data["a"],
                np.ones(length) * data["b"],
                np.ones(length) * data["chi"],
                grid[:, 0] / data["a"],
                grid[:, 1] / data["b"],
                grid[:, 2],
            )
        ).T
        output_data_new = data["grid_field"]
    elif mode == "field_extended":
        grid = data["grid"]
        length = len(grid)

        input_data_new = np.vstack(
            (
                np.ones(length) * data["a"],
                np.ones(length) * data["b"],
                np.ones(length) * data["chi"],
                grid[:, 0] / data["a"],
                grid[:, 1] / data["b"],
                grid[:, 2],
            )
        ).T
        output_data_new = data["grid_field_reduced"]
    else:
        raise ValueError("Incorrect mode")

    return (input_data_new, output_data_new)


data_range = np.arange(0, 10)
mode = "field_extended"

input_data, output_data = calculation(data_range[0], mode)


for i in range(1, len(data_range)):
    input_data_new, output_data_new = calculation(data_range[i], mode)

    input_data = np.concatenate((input_data, input_data_new), axis=0)
    output_data = np.concatenate((output_data, output_data_new), axis=0)

    print(i)


# Define the model
model = nn.Sequential(
    nn.Linear(6, 24),
    nn.ReLU(),
    nn.Linear(24, 48),
    nn.ReLU(),
    nn.Linear(48, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 3),
)


# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = Adam(model.parameters(), lr=0.00001)


# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(
    input_data, output_data, train_size=0.7, shuffle=True
)
X_train = torch.tensor(X_train, dtype=torch.float64)
y_train = torch.tensor(y_train, dtype=torch.float64)
X_test = torch.tensor(X_test, dtype=torch.float64)
y_test = torch.tensor(y_test, dtype=torch.float64)

# training parameters
n_epochs = 100  # number of epochs to run
batch_size = 32  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf  # init to infinity
best_weights = None
history = []

# training loop
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
if best_weights is not None:
    model.load_state_dict(best_weights)

print("MSE: %f" % best_mse)
print("RMSE: %f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()

torch.save(model.state_dict(), "model_%s.pth" % mode)

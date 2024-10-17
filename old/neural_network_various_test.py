import copy
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split

from Python_useful import amplitude_error, angle_error


def calculation(data_range, mode):
    # mode = 'magnetization', 'magnetization_extended', 'field', 'field_extended'

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
        raise ValueError("Error: Wrong keyword for mode")

    return (input_data_new, output_data_new)


data_range = np.arange(1000, 2000)
mode = "field"

input_data, output_data = calculation(data_range[0], mode)


for i in range(1, len(data_range)):
    input_data_new, output_data_new = calculation(data_range[i], mode)

    input_data = np.concatenate((input_data, input_data_new), axis=0)
    output_data = np.concatenate((output_data, output_data_new), axis=0)

    print(i)


# Define the model
# model = nn.Sequential(
#     nn.Linear(6, 24),
#     nn.ReLU(),
#     nn.Linear(24, 12),
#     nn.ReLU(),
#     nn.Linear(12, 6),
#     nn.ReLU(),
#     nn.Linear(6, 3)
# )
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


model.load_state_dict(torch.load("model_%s.pth" % mode))

predicted = np.zeros((len(input_data), 3))


model.eval()
with torch.no_grad():
    # Test out inference with 5 samples from the original test set
    for i in range(len(input_data)):
        X_sample = input_data[i, :]
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        start_time = time.perf_counter()
        y_pred = model(X_sample)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # print("Elapsed time: ", elapsed_time)
        # print(f"{input_data[i,:]} -> {y_pred[0].numpy()} (expected {output_data[i]}, difference {y_pred[0].numpy()-output_data[i]})")

        predicted[i, 0] = y_pred[0].numpy()
        predicted[i, 1] = y_pred[1].numpy()
        predicted[i, 2] = y_pred[2].numpy()

print("data")
print(output_data)
print(predicted)


np.savez(
    "%s_additional_data_newcostfunction.npz" % mode,
    input_data=input_data,
    output_data=output_data,
    predicted=predicted,
)


###################

mode = "field"

data = np.load("%s_additional_data_newcostfunction.npz" % mode)
input_data = data["input_data"]
output_data = data["output_data"]
predicted = data["predicted"]

# error = np.abs(np.linalg.norm(output_data, axis=-1) - np.linalg.norm(predicted, axis=-1))
error = np.linalg.norm(output_data - predicted, axis=-1)
amplitude = amplitude_error.amplitude_error(output_data, predicted)
angle = angle_error.angle_error(output_data, predicted)

# print('x')
# plt.plot(output_data[:,0])
# plt.plot(predicted[:,0])
# plt.show()

# print('y')
# plt.plot(output_data[:,1])
# plt.plot(predicted[:,1])
# plt.show()

# print('z')
# plt.plot(output_data[:,2])
# plt.plot(predicted[:,2])
# plt.show()

# print('magnitudes')
# plt.plot(np.linalg.norm(output_data, axis=-1), label='exact')
# plt.plot(np.linalg.norm(predicted, axis=-1), label='NN output')
# plt.xlabel('training data set')
# plt.ylabel('amplitude [a.u.]')
# plt.legend()
# plt.show()

# print('magnitude error')
# plt.plot(error)
# plt.xlabel('training data set')
# plt.ylabel('vector error [a.u.]')
# plt.show()

# print('amplitude error')
# plt.plot(amplitude)
# plt.xlabel('training data set')
# plt.ylabel('relative amplitude error [%]')
# plt.show()

# print('angle error')
# plt.plot(angle)
# plt.xlabel('training data set')
# plt.ylabel('angle error [°]')
# plt.show()

plt.boxplot(amplitude, vert=False, whis=(1, 99))
plt.grid()
plt.xlabel("relative amplitude error [%]")
plt.show()

plt.boxplot(angle, vert=False, whis=(1, 99))
plt.grid()
plt.xlabel("angle error [°]")
plt.show()

plt.boxplot(error, vert=False, whis=(1, 99))
plt.grid()
plt.xlabel("vector error [a.u.]")
plt.show()


n_bins = 10
data = amplitude

fig, ax = plt.subplots(nrows=7, ncols=1, sharex=True)

_, bins, _ = ax[0].hist(data, bins=n_bins, density=True)

bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = bins[1:] - bins[:-1]


logical_values = np.zeros((len(data), n_bins), dtype=bool)
for i in range(0, n_bins - 1):
    logical_values[:, i] = (data >= bins[i]) * (data < bins[i + 1])
logical_values[:, -1] = (data >= bins[-2]) * (data <= bins[-1])


a_values = []
b_values = []
axb_values = []
b_a_values = []
chi_values = []
distance = []
for i in range(0, n_bins):
    a_values.append(input_data[logical_values[:, i], 0])
    b_values.append(input_data[logical_values[:, i], 1])
    axb_values.append(
        input_data[logical_values[:, i], 0] * input_data[logical_values[:, i], 1]
    )
    b_a_values.append(
        input_data[logical_values[:, i], 1] / input_data[logical_values[:, i], 0]
    )
    chi_values.append(input_data[logical_values[:, i], 2])
    distance.append(
        np.sqrt(
            input_data[logical_values[:, i], 3] ** 2
            + input_data[logical_values[:, i], 4] ** 2
            + input_data[logical_values[:, i], 5] ** 2
        )
    )


ax[0].grid()
ax[0].set_ylabel("density")

ax[1].boxplot(a_values, whis=[1, 99], positions=bin_centers, widths=bin_widths * 0.8)
ax[1].grid()
ax[1].set_ylabel("a")

ax[2].boxplot(b_values, whis=[1, 99], positions=bin_centers, widths=bin_widths * 0.8)
ax[2].grid()
ax[2].set_ylabel("b")

ax[3].boxplot(axb_values, whis=[1, 99], positions=bin_centers, widths=bin_widths * 0.8)
ax[3].grid()
ax[3].set_ylabel("a*b")

ax[4].boxplot(b_a_values, whis=[1, 99], positions=bin_centers, widths=bin_widths * 0.8)
ax[4].grid()
ax[4].set_ylabel("b/a")

ax[5].boxplot(chi_values, whis=[1, 99], positions=bin_centers, widths=bin_widths * 0.8)
ax[5].grid()
ax[5].set_ylabel("chi")

ax[6].boxplot(distance, whis=[1, 99], positions=bin_centers, widths=bin_widths * 0.8)
ax[6].grid()
ax[6].set_ylabel("distance [rel. coord.]")

ax[-1].set_xticks(bin_centers)
ax[-1].set_xticklabels(np.round(bin_centers, decimals=4))
ax[-1].set_xlabel("relative amplitude error [%]")
plt.show()

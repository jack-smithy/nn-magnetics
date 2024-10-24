from dataset import make_train_test_split, ChiMode, get_data
from train import calculate_metrics_baseline
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

X, B = get_data(path=Path("data/anisotropic_chi"), chi_mode=ChiMode.ANISOTROPIC)

angle_errs, amp_errs = calculate_metrics_baseline(B)

angle_dist = np.mean(angle_errs, axis=1)
amp_dist = np.mean(amp_errs, axis=-1)

mean_angle_error = np.mean(angle_dist)
mean_amp_error = np.mean(amp_dist)

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

ax[0].hist(angle_dist, bins=20, label=f"Mean error: {round(mean_angle_error, 3)}")
ax[0].set_ylabel("Count (baseline)")
ax[0].set_xlabel("Angle error (degrees)")
ax[0].legend()

ax[1].hist(amp_dist, bins=20, label=f"Mean error: {round(mean_amp_error, 3)}")
ax[1].set_xlabel("Relative amplitude error (%)")
ax[1].legend()

plt.tight_layout()
plt.show()

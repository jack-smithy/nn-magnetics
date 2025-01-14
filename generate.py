import numpy as np
from nn_magnetics.data.create_data import simulate_demag


def simulate_task(index):
    chi_x = np.random.uniform(0.0, 1.0)
    chi_y = np.random.uniform(0.0, 1.0)
    chi_z = np.random.uniform(0.0, 1.0)
    chi = (chi_x, chi_y, chi_z)

    a = np.random.uniform(low=0.3, high=3.0)
    b = np.random.uniform(low=0.3, high=3.0)
    # chi = np.random.uniform(0, 1)

    if a > b:
        a, b = b, a

    print(f"Starting simuation: {index}")
    data = simulate_demag(a, b, chi)
    path = f"data/3dof_chi/train/data_{index}.npz"
    np.savez(path, **data)


if __name__ == "__main__":
    for idx in range(1, 11):
        simulate_task(idx)

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

    print(f"Starting simuation: {index+1}")
    data = simulate_demag(a, b, chi)
    path = f"data/3dof_chi_v3/large/test/data_{index+1}.npz"
    np.savez(path, **data)


def main():
    for idx in range(400):
        simulate_task(idx)


if __name__ == "__main__":
    main()

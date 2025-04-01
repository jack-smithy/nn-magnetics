import numpy as np
from nn_magnetics.data.create_data import simulate_demag


def main():
    for idx in range(200):
        chi_x = np.random.uniform(0.0, 1.0)
        chi_y = np.random.uniform(0.0, 1.0)
        chi_z = np.random.uniform(0.0, 1.0)

        a = np.random.uniform(low=0.3, high=3.0)
        b = np.random.uniform(low=0.3, high=3.0)

        if a > b:
            a, b = b, a

        print(f"Starting simuation: {idx+1}")
        data = simulate_demag(a, b, chi_x, chi_y, chi_z)
        path = f"data/3dof_chi_v2/validation/data_{idx+1}.npz"
        np.savez(path, **data)


if __name__ == "__main__":
    main()

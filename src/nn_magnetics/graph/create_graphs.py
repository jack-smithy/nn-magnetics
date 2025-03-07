from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

tol = 1e-5


def _get_graph_from_path(path):
    print(f"Starting {path}")
    data = np.load(path)
    grid = torch.from_numpy(data["grid"])
    length = len(grid)

    n_nodes = grid.size(0)
    edges = []

    for d in range(3):
        other_dims = [i for i in range(3) if i != d]
        groups = {}
        for i in range(n_nodes):
            key = tuple((grid[i, other_dims] / tol).round().int().tolist())
            groups.setdefault(key, []).append(i)
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            sorted_idxs = sorted(idxs, key=lambda i: grid[i, d].item())
            for j in range(len(sorted_idxs) - 1):
                i1 = sorted_idxs[j]
                i2 = sorted_idxs[j + 1]

                if abs(grid[i2, d] - grid[i1, d]) > tol:
                    edges.append([i1, i2])
                    edges.append([i2, i1])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    input_data_new = np.vstack(
        (
            np.ones(length) * data["a"],
            np.ones(length) * data["b"],
            np.ones(length) * data["chi_x"],
            np.ones(length) * data["chi_y"],
            np.ones(length) * data["chi_z"],
            grid[:, 0] / data["a"],
            grid[:, 1] / data["b"],
            grid[:, 2],
        )
    ).T

    # get the corresponding labels
    output_data_new = np.concatenate(
        (data["grid_field"], data["grid_field_reduced"]),
        axis=1,
    )

    X = torch.from_numpy(input_data_new)
    B = torch.from_numpy(output_data_new)

    print(f"Done {path}")

    return X, B, edge_index


def get_graphs_from_dir(path: Path):
    graphs = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_get_graph_from_path, file) for file in path.iterdir()
        ]

        for future in as_completed(futures):
            X, B, edge_index = future.result()
            graph = Data(x=X, y=B, edge_index=edge_index)
            graphs.append(graph)

    return graphs


if __name__ == "__main__":
    graphs = get_graphs_from_dir(Path("data/3dof_chi/train_fast"))
    print(len(graphs))


#################################
### old #########################
#################################


def compute_edge_index_fast(pos: np.ndarray, tol=1e-5):
    n_nodes = pos.shape[0]
    edges = []

    for d in range(3):
        other_dims = [i for i in range(3) if i != d]
        groups = {}
        for i in range(n_nodes):
            key = tuple((pos[i, other_dims] / tol).round().tolist())
            groups.setdefault(key, []).append(i)
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            sorted_idxs = sorted(idxs, key=lambda i: pos[i, d].item())
            for j in range(len(sorted_idxs) - 1):
                i1 = sorted_idxs[j]
                i2 = sorted_idxs[j + 1]

                if abs(pos[i2, d] - pos[i1, d]) > tol:
                    edges.append([i1, i2])
                    edges.append([i2, i1])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return edge_index

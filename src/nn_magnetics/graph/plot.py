import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data


def plot_graph_3d(x, edge_index):
    """
    Creates a 3D plot of a graph represented by a PyG Data object.

    Args:
        data (Data): PyG Data object with x (node coordinates) and edge_index attributes.
                      Assumes x is of shape [num_nodes, 3] and represents (x, y, z) coordinates.
                      edge_index is of shape [2, num_edges] and represents connections between nodes.
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot nodes
    ax.scatter(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        c="blue",
        marker="o",
        s=100,  # type: ignore
        label="Nodes",
    )

    # Plot edges
    for i in range(edge_index.shape[1]):
        start_node = edge_index[0, i]
        end_node = edge_index[1, i]

        start_point = x[start_node]
        end_point = x[end_node]

        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            [start_point[2], end_point[2]],
            color="red",
            alpha=0.5,
            linewidth=1,
            label="Edges" if i == 0 else "",
        )  # Label only the first edge for legend

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore
    ax.set_title("3D Graph Visualization")
    ax.legend()
    plt.show()

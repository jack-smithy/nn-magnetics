import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Define scaling factors
a, b, c = 1.0, 1.0, 1.0  # Replace with actual values

# Define parameters
num_points = 4  # 3 points along each axis
scaling_exponent = 2  # Adjust for more or less density near origin


# Generate scaled coordinates
def nonuniform_spacing(N, L, p):
    lin = np.linspace(0, 1, N)  # Normalized linear space
    scaled = (lin**p) * L / (1**p)  # Apply nonlinear transformation
    return scaled


# x = nonuniform_spacing(num_points, 2.5 * a, scaling_exponent)
# y = nonuniform_spacing(num_points, 2.5 * b, scaling_exponent)
# z = nonuniform_spacing(num_points, 2.5 * c, scaling_exponent)

# # Create 3D grid
# X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# # Stack points into an array of (x, y, z) coordinates
# grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

# # Define cuboid exclusion conditions
# inside_cuboid = (
#     (-a / 2 <= grid_points[:, 0])
#     & (grid_points[:, 0] <= a / 2)
#     & (-b / 2 <= grid_points[:, 1])
#     & (grid_points[:, 1] <= b / 2)
#     & (-c / 2 <= grid_points[:, 2])
#     & (grid_points[:, 2] <= c / 2)
# )

# # Remove points inside the cuboid
# filtered_points = grid_points[~inside_cuboid]


def generate_points(num_points, a, b, c, scale_factor=3.5):
    """
    Generate points (x,y,z) in the first octant (x,y,z>=0) that lie outside a box
    centered at the origin with side lengths (a/2, b/2, c/2), with a higher density near the box.

    Parameters:
      num_points: number of points to generate.
      a, b, c: parameters defining the box side lengths.
      scale_factor: controls how tightly points cluster around the box (smaller = more concentrated).

    Returns:
      A list of tuples (x, y, z) that are outside the box.
    """
    points = []
    x_thresh, y_thresh, z_thresh = (
        a / 2,
        b / 2,
        c / 2,
    )  # Box boundaries in the positive octant

    while len(points) < num_points:
        # Generate exponentially distributed random numbers, shifting them by the threshold
        x = np.random.exponential(scale_factor * x_thresh)
        y = np.random.exponential(scale_factor * y_thresh)
        z = np.random.exponential(scale_factor * z_thresh)

        outside_box = x > a / 2 or y > b / 2 or z > c / 2
        inside_domain = x <= 2.5 * a and y <= 2.5 * b and z <= 2.5 * c
        if outside_box and inside_domain:
            points.append((x, y, z))

    return np.array(points)


filtered_points = generate_points(10000, 1, 1, 1, 3.5)

# Plot the filtered points
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the remaining grid points
ax.scatter(
    filtered_points[:, 0],
    filtered_points[:, 1],
    filtered_points[:, 2],
    color="b",
    s=10,
    label="Grid Points",
)

# Plot the excluded cuboid (wireframe)
cuboid_x = np.array([0, a / 2, a / 2, 0, 0, a / 2, a / 2, 0])
cuboid_y = np.array([0, 0, b / 2, b / 2, 0, 0, b / 2, b / 2])
cuboid_z = np.array([0, 0, 0, 0, c / 2, c / 2, c / 2, c / 2])

edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]
for edge in edges:
    ax.plot(
        [cuboid_x[edge[0]], cuboid_x[edge[1]]],
        [cuboid_y[edge[0]], cuboid_y[edge[1]]],
        [cuboid_z[edge[0]], cuboid_z[edge[1]]],
        color="r",
    )

ax.set_xlim((0, 2.5 * a))
ax.set_ylim((0, 2.5 * b))
ax.set_zlim((0, 2.5 * c))
# Labels and view settings
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Non-Uniform 3x3x3 Grid with Cuboid Excluded")
ax.legend()
plt.show()

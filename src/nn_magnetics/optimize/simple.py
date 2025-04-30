from nn_magnetics.utils.physics import Bfield_homogeneous
from nn_magnetics.data.create_data import generate_points_grid, generate_points_random
from magpylib import magnet
from magpylib_material_response import meshing, demag
import torch


def vector_field_correlation(B1, B2):
    """
    Calculate vector correlation between vector fields B1 and B2 according to:
                     Σ B1,i · B2,i
    CV_vec = -------------------------------
             (Σ |B1,i|² · Σ |B2,i|²)^(1/2)

    Args:
        B1 (torch.Tensor): First vector field with shape (N, 3)
        B2 (torch.Tensor): Second vector field with shape (N, 3)

    Returns:
        torch.Tensor: Scalar correlation value
    """
    # Check if shapes match and are of the expected form
    if B1.shape != B2.shape:
        raise ValueError(
            f"Input vector fields must have the same shape. Got {B1.shape} and {B2.shape}"
        )

    if B1.shape[1] != 3:
        raise ValueError(f"Expected vector fields with shape (N, 3), got {B1.shape}")

    # Calculate dot product at each point
    # (B1 * B2).sum(dim=1) gives the dot product for each point
    point_dot_products = torch.sum(B1 * B2, dim=1)

    # Sum all dot products for the numerator
    numerator = torch.sum(point_dot_products)

    # Calculate the squared magnitudes at each point
    B1_squared_magnitudes = torch.sum(B1 * B1, dim=1)
    B2_squared_magnitudes = torch.sum(B2 * B2, dim=1)

    # Sum the squared magnitudes
    sum_B1_squared = torch.sum(B1_squared_magnitudes)
    sum_B2_squared = torch.sum(B2_squared_magnitudes)

    # Calculate denominator: sqrt of product of sum of squared magnitudes
    denominator = torch.sqrt(sum_B1_squared * sum_B2_squared)

    # Calculate correlation coefficient
    correlation = numerator / denominator

    return correlation


DIMENSIONS = (1, 2, 1)
POLARIZATION = (0.0, 0.5, 1.0)
OBSERVERS = generate_points_random(26, 2, 1)


cuboid = magnet.Cuboid(dimension=DIMENSIONS, polarization=POLARIZATION)
B = cuboid.getB(OBSERVERS)
B_target = torch.from_numpy(B).to(torch.float64)

obs = torch.tensor(OBSERVERS)
# pols = torch.tensor(POLARIZATION)
dims = torch.tensor(DIMENSIONS)


def func(pols):
    dims_expanded = dims.unsqueeze(0).expand(obs.shape[0], -1)
    pols_expanded = pols.unsqueeze(0).expand(obs.shape[0], -1)

    B_pred = Bfield_homogeneous(
        observers=obs,
        polarizations=pols_expanded,
        dimensions=dims_expanded,
    )
    loss = torch.nn.functional.mse_loss(B_pred, B_target)

    return loss


pols = torch.nn.Parameter(torch.randn(3, dtype=torch.float32))
opt = torch.optim.Adam(params=[pols], lr=0.01)

for step in range(1001):
    opt.zero_grad()
    loss = func(pols)
    loss.backward()
    opt.step()


print(pols)

# print(
#     f"Avg dimensions:\n[{pol_mean[0]}±{pol_std[0]},\n{pol_mean[1]}±{pol_std[1]},\n{pol_mean[2]}±{pol_std[2]}]"
# )

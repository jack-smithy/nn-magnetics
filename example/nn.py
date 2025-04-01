import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Callable

from physics import Bfield_homogeneous, Dz_cuboid, batch_rotation_matrices, divB


class FieldCorrectionNetwork(nn.Module):
    def __init__(
        self,
        in_features: int = 8,
        hidden_dim_factor: int = 6,
        out_features: int = 4,
        activation=F.silu,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(
            in_features=in_features,
            out_features=4 * hidden_dim_factor,
        )
        self.linear2 = nn.Linear(
            in_features=4 * hidden_dim_factor,
            out_features=8 * hidden_dim_factor,
        )
        self.linear3 = nn.Linear(
            in_features=8 * hidden_dim_factor,
            out_features=4 * hidden_dim_factor,
        )
        self.linear4 = nn.Linear(
            in_features=4 * hidden_dim_factor,
            out_features=2 * hidden_dim_factor,
        )
        self.linear5 = nn.Linear(
            in_features=2 * hidden_dim_factor,
            out_features=hidden_dim_factor,
        )
        self.output = nn.Linear(
            in_features=hidden_dim_factor,
            out_features=out_features,
        )

        self.activation = activation

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        observers, dimensions, polarizations, susceptibilities = self._prepare_inputs(x)

        # calculate analytical solution
        B_analytical = Bfield_homogeneous(
            observers=observers,
            dimensions=dimensions,
            polarizations=polarizations,
        )

        # get nn correction factor
        B_correction = self._forward(
            observers=observers,
            dimensions=dimensions,
            susceptibilities=susceptibilities,
        )

        # correct analytical solution with nn prediction
        B = self._correct_ansatz(B_analytical, B_correction)

        # calculate divergence of B
        divergence_B = divB(B=B, x=observers)

        return B, divergence_B

    def _correct_ansatz(self, B_analytical: Tensor, prediction: Tensor) -> Tensor:
        assert prediction.shape[1] == 4

        angles = prediction[..., :3]
        amplitudes = prediction[..., 3]

        Rs_t = batch_rotation_matrices(angles)

        # Multiply each vector in B_reduced by the corresponding rotation matrix in Rs
        return amplitudes[:, None] * torch.einsum("nij,nj->ni", Rs_t, B_analytical)

    def _forward(
        self,
        observers: Tensor,
        dimensions: Tensor,
        susceptibilities: Tensor,
    ) -> Tensor:
        """
        Forward pass through the NN. Returns a matrix with columns `(alpha, beta, gamma, amplitude)`.
        """

        # the nn expects the features to be in one matrix
        x = torch.concat(
            (
                dimensions[..., :2],  # a, b
                susceptibilities,  # chi_x, chi_y, chi_z
                observers / dimensions,  # x/a, y/b, z
            ),
            dim=1,
        )

        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = self.activation(self.linear5(x))

        # no activation on output for regression
        prediction = self.output(x)

        return prediction

    @staticmethod
    def _prepare_inputs(x: Tensor) -> tuple[Tensor, ...]:
        """
        Constructs feature vectors for B-field calculation
        """
        n_samples = x.shape[0]

        # extract the spatial coordinates and make a new tensor with gradients
        observers = x[:, 5:].clone().requires_grad_(True)

        # make tensor containing dimensions of each magnet in the batch
        dimensions = torch.concat((x[:, :2], torch.ones((n_samples, 1))), dim=1)

        # calculate the reduced polarizations from the dimensions and χ_z
        J_z = 1 / (1 + x[:, 4] * Dz_cuboid(dimensions)).unsqueeze(-1)
        polarizations = torch.concat(
            (torch.zeros((n_samples, 1)), torch.zeros((n_samples, 1)), J_z),
            dim=1,
        )

        # make tensor containing the susceptibilities for each magnet
        susceptibilities = x[:, 2:5].clone()

        return observers, dimensions, polarizations, susceptibilities


class DivergenceLoss(nn.Module):
    def __init__(self, lambda_: float = 1, loss_fn: Callable = F.l1_loss) -> None:
        """
        Loss function which includes data loss and physics loss from ∇·B.

        Args:
            lambda_ (float): Weighting for the physics loss
            loss_fn (Callable): Loss function for the data loss. Defaults to F.l1_loss.
        """
        super().__init__()
        self.lambda_ = lambda_
        self.loss_fn = loss_fn

    def forward(self, B_true: Tensor, B_pred: Tensor, divB: Tensor) -> Tensor:
        data_loss = self.loss_fn(B_true, B_pred)
        physics_loss = divB.abs().mean()

        return data_loss + self.lambda_ * physics_loss

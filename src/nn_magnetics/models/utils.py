import torch
from torch import Tensor
from nn_magnetics.utils.physics import cartesian_to_spherical


def angular_loss(
    B_true: Tensor,
    B_pred: Tensor,
    loss=torch.nn.functional.l1_loss,
) -> Tensor:
    """
    Calculates loss in spherical coordinates
    """
    B_true_spherical = cartesian_to_spherical(B_true)
    B_pred_spherical = cartesian_to_spherical(B_pred)

    return loss(B_true_spherical, B_pred_spherical)


def get_num_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return (
        sum(p.numel() for p in model.parameters())
        if trainable_only
        else sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

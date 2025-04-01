import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class DivergenceLoss(nn.Module):
    def __init__(self, lambda_: float = 1, loss_fn=F.l1_loss) -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.loss_fn = loss_fn

    def forward(self, B_true: Tensor, B_pred: Tensor, divB: Tensor) -> Tensor:
        data_loss = self.loss_fn(B_true, B_pred)
        physics_loss = divB.abs().mean()

        return data_loss + self.lambda_ * physics_loss


def get_num_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return (
        sum(p.numel() for p in model.parameters())
        if trainable_only
        else sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

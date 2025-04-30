from typing import Type

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from nn_magnetics.models import BaseNetwork


def vector_field_correlation(B1, B2):
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


def _calc_loss(
    model: BaseNetwork,
    B_measured: Tensor,
    observers: Tensor,
    susceptibilities: Tensor,
    dimensions: Tensor,
) -> Tensor:
    """
    Calculate the loss for the predicted field vs the true field

    Args:
        model (BaseNetwork): Trained predictive model
        B_measured (Tensor): Ground truth B field
        observers (Tensor): Evaluation points
        susceptibilities (Tensor): Magnet susceptibility
        dimensions (Tensor): Magnet dimensions

    Returns:
        Tensor: Loss value
    """
    susc = susceptibilities.unsqueeze(0).expand((observers.shape[0], -1)).sigmoid()
    input = torch.cat([dimensions, susc, observers], dim=1).to(torch.float64)
    B_predicted = model(input)
    return F.mse_loss(B_measured, B_predicted)


def _build_model(model_cls: Type[BaseNetwork], path: str) -> BaseNetwork:
    """
    Load a trained model from saved weights. We can't do this beforehand as the model
    can't be shared between threads

    Args:
        model_cls (Type[BaseNetwork]): Model class
        path (str): Path to saved weights

    Returns:
        BaseNetwork: Model with loaded weights
    """
    return model_cls.load_from_path(
        path,
        activation=F.silu,
        save_path=None,
        save_weights=False,
    ).to(torch.float64)


def _optimize(
    model_cls: Type[BaseNetwork],
    path: str,
    observers: Tensor,
    B_measured: Tensor,
    dimensions: Tensor,
    n_steps: int,
    seed: int | None = None,
) -> Tensor:
    """_summary_

    Args:
        model_cls (Type[BaseNetwork]): _description_
        path (str): _description_
        observers (Tensor): _description_
        B_measured (Tensor): _description_
        dimensions (Tensor): _description_
        n_steps (int): _description_
        seed (int | None, optional): _description_. Defaults to None.

    Returns:
        Tensor: _description_
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = _build_model(model_cls=model_cls, path=path)

    def objective_func(susceptibilities):
        return _calc_loss(
            model=model,
            B_measured=B_measured,
            observers=observers,
            susceptibilities=susceptibilities,
            dimensions=dimensions,
        )

    susceptibilities = torch.nn.Parameter(torch.randn(3) * 0.1)
    optimizer = torch.optim.Adam(params=[susceptibilities], lr=0.001)
    for _ in tqdm(range(n_steps), disable=seed is not None):
        loss = objective_func(susceptibilities)
        loss.backward()
        optimizer.step()

    return susceptibilities.detach().clone().sigmoid()


def _optimize_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing"""
    seed, kwargs = args
    return _optimize(**kwargs, seed=seed)


def optimize(
    model_cls: Type[BaseNetwork],
    path: str,
    observers: Tensor,
    B_measured: Tensor,
    a: float,
    b: float,
    n_steps: int,
    n_repeats: int,
    num_workers: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Parallel version of the optimization function.

    Args:
        model_cls: The model class to optimize
        path: Path to the model weights
        X: Input tensor
        B_measured: Measured B field tensor
        a, b: Dimensions
        n_steps: Number of optimization steps
        n_repeats: Number of repeat optimizations to run in parallel
        num_workers: Number of parallel workers (defaults to min(n_repeats, cpu_count))
    """
    n_samples = observers.shape[0]
    dimensions = torch.tensor([a, b]).unsqueeze(0).expand((n_samples, -1))

    if num_workers is None:
        num_workers = min(n_repeats, mp.cpu_count())

    if mp.get_start_method(allow_none=True) != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    kwargs = {
        "model_cls": model_cls,
        "path": path,
        "observers": observers,
        "B_measured": B_measured,
        "dimensions": dimensions,
        "n_steps": n_steps,
    }

    # Generate args for each parallel run with different seeds for randomization
    args_list = [(i, kwargs) for i in range(n_repeats)]

    # Run optimizations in parallel
    if num_workers > 1:
        print(
            f"Running {n_repeats} optimizations for {n_steps} steps in parallel using {num_workers} workers"
        )
        with mp.Pool(processes=num_workers) as pool:
            suscs = pool.map(_optimize_wrapper, args_list)
    else:
        print(f"Running {n_repeats} optimizations for {n_steps} steps sequentially")
        suscs = [_optimize_wrapper(args) for args in args_list]

    suscs = torch.stack(suscs)

    return suscs.mean(0), suscs.std(0)

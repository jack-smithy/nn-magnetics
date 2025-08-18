from typing import Type

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from nn_magnetics.models import BaseNetwork


def format_results(susc_mean, susc_std, precision: int = 5) -> str:
    """Process and format the results"""

    a_mean, b_mean, c_mean = (
        round(susc_mean[0].item(), precision),
        round(susc_mean[1].item(), precision),
        round(susc_mean[2].item(), precision),
    )

    a_std, b_std, c_std = (
        round(susc_std[0].item(), precision),
        round(susc_std[1].item(), precision),
        round(susc_std[2].item(), precision),
    )

    errs = [
        round(
            abs(SUSCEPTIBILITY[i] - susc_mean[i].item()) / SUSCEPTIBILITY[i] * 100,
            precision,
        )
        for i in range(3)
    ]

    return f"""
    chi_x={a_mean}±{a_std}, chi_y={b_mean}±{b_std}, chi_z={c_mean}±{c_std}
    errors: x={errs[0]}%, y={errs[1]}%, z={errs[2]}%, overall: {sum(errs)/3}%
    """


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
    susc = susceptibilities.unsqueeze(0).expand((observers.shape[0], -1))
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
        p=0.2,
        do_output_activation=False,
    ).to(torch.float64)


def _optimize(
    model_cls: Type[BaseNetwork],
    path: str,
    observers: Tensor,
    B_measured: Tensor,
    dimensions: Tensor,
    n_steps: int,
    seed: int | None = None,
    verbose: bool = False,
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

    susceptibilities = torch.nn.Parameter(torch.rand(3))
    optimizer = torch.optim.Adagrad(params=[susceptibilities], lr=0.001)

    def objective_func(susceptibilities):
        return _calc_loss(
            model=model,
            B_measured=B_measured,
            observers=observers,
            susceptibilities=susceptibilities,
            dimensions=dimensions,
        )

    for i in tqdm(range(n_steps), disable=seed is not None):
        loss = objective_func(susceptibilities)
        loss.backward()
        optimizer.step()

        if verbose:
            if i % 10 == 0:
                print(
                    f"Step {i}: Loss={loss.item()}, Susceptibility={susceptibilities.tolist()}"
                )

    return susceptibilities.detach().clone()


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
    verbose: bool = False,
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

    kwargs = {
        "model_cls": model_cls,
        "path": path,
        "observers": observers,
        "B_measured": B_measured,
        "dimensions": dimensions,
        "n_steps": n_steps,
        "verbose": verbose,
    }

    # Generate args for each parallel run with different seeds for randomization
    args_list = [(i, kwargs) for i in range(n_repeats)]

    print(f"Running {n_repeats} optimizations for {n_steps} steps sequentially")
    suscs = [_optimize_wrapper(args) for args in args_list]

    suscs = torch.stack(suscs)

    return suscs.mean(0), suscs.std(0)

import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree, jaxtyped
from torch.utils.data import DataLoader
from beartype import beartype as typechecker

from nn_magnetics.data.dataset import IsotropicData
from nn_magnetics.jax.network import Model


@jaxtyped(typechecker=typechecker)
def L1loss(
    y: Float[Array, "batch 3"],
    y_pred: Float[Array, "batch 3"],
) -> Float[Array, ""]:
    return jnp.mean(jnp.abs(y - y_pred))


@eqx.filter_value_and_grad
def loss_fn(
    model: Model,
    X: Float[Array, "batch 6"],
    B: Float[Array, "batch 6"],
) -> Float:
    B_demag, B_ana = B[..., :3], B[..., 3:]
    B_corrections = jax.vmap(model)(X)

    B_pred = B_ana * B_corrections

    return L1loss(y=B_demag, y_pred=B_pred)


@eqx.filter_jit
def make_step(
    model: Model,
    opt_state: PyTree,
    optim: optax.GradientTransformation,
    x: Float[Array, "batch 6"],
    y: Float[Array, " batch 3"],
):
    loss_value, grads = loss_fn(model, x, y)

    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )

    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss_value


def angle_error(
    A: Float[Array, "batch 3"],
    B: Float[Array, "batch 3"],
) -> Float[Array, "batch 1"]:
    A_norm, B_norm = jnp.linalg.norm(A, axis=1), jnp.linalg.norm(B, axis=1)
    cos_angle = jnp.sum(A * B) / (A_norm * B_norm)
    cos_angle[cos_angle > 1] = 1
    cos_angle[cos_angle < -1] = -1
    angle = jnp.arccos(cos_angle)
    return jnp.rad2deg(angle)


def relative_amplitude_error(
    A: Float[Array, "batch 3"],
    B: Float[Array, "batch 3"],
    return_abs: bool = True,
) -> Float[Array, "batch 1"]:
    A_norm, B_norm = jnp.linalg.norm(A, axis=1), jnp.linalg.norm(B, axis=1)

    relative_error = (B_norm - A_norm) / A_norm * 100

    if return_abs:
        relative_error = jnp.abs(relative_error)

    return relative_error


def train(
    model: Model,
    trainloader: DataLoader,
    testloader: DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> tuple[Model, float]:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    start = time.perf_counter()
    for step in range(steps):
        loss_value = 0
        for X, B in trainloader:
            X, B = jnp.asarray(X), jnp.asarray(B)
            model, opt_state, loss_value = make_step(model, opt_state, optim, X, B)

        if step % print_every == 0:
            print(loss_value)

    end = time.perf_counter()
    return model, (end - start)


def main():
    DATA_DIR = Path("data/isotropic_chi")
    BATCH_SIZE = 64
    SEED = 1
    LEARNING_RATE = 1e-4
    STEPS = 500
    PRINT_EVERY = 1

    training_data = IsotropicData(path=DATA_DIR / "train_fast")
    testing_data = IsotropicData(path=DATA_DIR / "test_fast")

    trainloader = DataLoader(training_data, shuffle=True, batch_size=BATCH_SIZE)
    testloader = DataLoader(testing_data, shuffle=True, batch_size=BATCH_SIZE)

    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)
    model = Model(subkey)
    optim = optax.adam(LEARNING_RATE)

    model, time = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)
    print(time)


if __name__ == "__main__":
    main()

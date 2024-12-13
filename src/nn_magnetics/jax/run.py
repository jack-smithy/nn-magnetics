from ast import Tuple
from multiprocessing import dummy
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import List
import optax
from matplotlib import axis
from jaxtyping import PyTree, Float, Array, Int
import time

from nn_magnetics.data.dataset import IsotropicData
from torch.utils.data import DataLoader

DATA_DIR = Path("data/isotropic_chi")
BATCH_SIZE = 128
SEED = 1234
LEARNING_RATE = 1e-4
STEPS = 500
PRINT_EVERY = 10


class Model(eqx.Module):
    layers: List

    def __init__(self, key):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Linear(in_features=6, out_features=48, key=key1),
            jax.nn.silu,
            eqx.nn.Linear(in_features=48, out_features=24, key=key2),
            jax.nn.silu,
            eqx.nn.Linear(in_features=24, out_features=12, key=key3),
            jax.nn.silu,
            eqx.nn.Linear(in_features=12, out_features=6, key=key4),
            jax.nn.silu,
            eqx.nn.Linear(in_features=6, out_features=3, key=key5),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


def L1loss(y, y_pred):
    return jnp.mean(jnp.abs(y - y_pred))


@jax.jit
def amplitude_error(y, y_pred):
    norm_y = jnp.linalg.norm(y, axis=1)
    norm_y_pred = jnp.linalg.norm(y_pred, axis=1)

    return jnp.abs(norm_y_pred - norm_y) / norm_y


def loss_fn(model, x, y):
    x, y = jnp.asarray(x), jnp.asarray(y)
    y_pred = jax.vmap(model)(x)
    return L1loss(y=y, y_pred=y_pred)


# def evaluate(model, testloader):
#     errors = jnp.asarray([])
#     losses = jnp.asarray([])

#     for X, B in testloader:
#         X, B = jnp.asarray(X), jnp.asarray(B)
#         B_pred = jax.vmap(model)(X)
#         batch_errors = amplitude_error(B, B_pred)
#         batch_loss = loss_fn(model, X, B)
#         errors = jnp.concatenate((errors, batch_errors), axis=0)
#         losses = jnp.append(losses, batch_loss)

#     return jnp.mean(losses), jnp.mean(errors)


def train(
    model: Model,
    trainloader: DataLoader,
    testloader: DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> tuple[Model, float]:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: Model,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Float[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    @eqx.filter_jit
    def evaluate(model):
        errors = jnp.asarray([])
        losses = jnp.asarray([])

        for X, B in testloader:
            X, B = jnp.asarray(X), jnp.asarray(B)
            B_pred = jax.vmap(model)(X)
            batch_errors = amplitude_error(B, B_pred)
            batch_loss = loss_fn(model, X, B)
            errors = jnp.concatenate((errors, batch_errors), axis=0)
            losses = jnp.append(losses, batch_loss)

        return jnp.mean(losses), jnp.mean(errors)

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    start = time.perf_counter()
    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        # if (step % print_every) == 0 or (step == steps - 1):
        test_loss, test_error = evaluate(model)
        # print(
        #     f"{step=}, train_loss={train_loss.item()}, "
        #     f"test_loss={test_loss.item()}, test_error={test_error.item()}"
        # )
    end = time.perf_counter()
    return (model, (end - start))


def main():
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

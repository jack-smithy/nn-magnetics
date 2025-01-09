from typing import List

import equinox as eqx
import jax
from jaxtyping import Array, Float


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

    def __call__(self, x: Float[Array, "batch 6"]) -> Float[Array, "batch 3"]:
        for layer in self.layers:
            x = layer(x)

        return x

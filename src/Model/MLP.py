import jax.numpy as jnp

from jax.nn import silu
from flax import linen as nn
from typing import (Callable, Sequence)
import e3x


class MLP(nn.Module):
    features: Sequence[int]
    activation_fn: Callable = lambda x: x
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for i, ft in enumerate(self.features):
            x = e3x.nn.Dense(ft, use_bias=self.use_bias, name=f'layers_{i}')(x)
            if i != len(self.features) - 1:
                x = self.activation_fn(x)
        return x


class Residual(nn.Module):
    num_blocks: int = 2
    activation_fn: Callable = silu
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        ft = x.shape[-1]
        for i in range(self.num_blocks):
            x = self.activation_fn(x)
            x = e3x.nn.Dense(ft, use_bias=self.use_bias, name=f'layers_{i}')(x)

        return inputs + x


class ResidualMLP(nn.Module):
    num_residuals: int = 1
    num_blocks_per_residual: int = 2
    activation_fn: Callable = silu
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *args, **kwargs):
        """
        Args:
            inputs ():
            *args ():
            **kwargs ():

        Returns:
        """
        x = inputs
        ft = x.shape[-1]
        for i in range(self.num_residuals):
            x = Residual(num_blocks=self.num_blocks_per_residual,
                         activation_fn=self.activation_fn,
                         use_bias=self.use_bias)(x)
        x = self.activation_fn(x)
        x = e3x.nn.Dense(ft, use_bias=self.use_bias)(x)
        return x

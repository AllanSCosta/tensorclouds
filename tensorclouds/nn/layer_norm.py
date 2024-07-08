import e3nn_jax as e3nn
from flax import linen as nn
import jax.numpy as jnp
import jax

class EquivariantLayerNorm(nn.Module):
    @nn.compact
    def __call__(self, input: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        outputs = []
        for (_, ir), x in zip(input.irreps, input.list):
            if ir.l == 0:
                x = nn.LayerNorm()(x[..., 0])[..., None]
            else:
                x = x / (rms(x) + 1e-6)
            outputs.append(x)
        return e3nn.IrrepsArray.from_list(input.irreps, outputs, input.shape[:-1])


def rms(x: jax.Array) -> jax.Array:
    # x.shape == (..., mul, dim)
    norms_sqr = jnp.sum(x**2, axis=-1, keepdims=True)  # sum over dim
    mean_norm_sqr = jnp.mean(norms_sqr, axis=-2, keepdims=True)  # mean over mul
    vectors_rms = jnp.sqrt(jnp.where(mean_norm_sqr == 0.0, 1.0, mean_norm_sqr))
    assert vectors_rms.shape == x.shape[:-2] + (1, 1)
    return vectors_rms

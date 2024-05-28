import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp


class EquivariantLayerNorm(hk.Module):
    def __call__(self, input: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        outputs = []

        if None in input.list:
            return input
        for i, ((mul, ir), x) in enumerate(zip(input.irreps, input.list)):
            if ir.l == 0:
                x -= jnp.mean(x, axis=-2, keepdims=True)
                # x /= (jnp.std(x, axis=-2, keepdims=True) + 1e-6)
                # x = jnp.where(jnp.isnan(x), 0.0, x)
            #     x += hk.get_parameter(
            #         f"bias_{i}", (mul, ir.dim), init=jnp.zeros, dtype=jnp.bfloat16
            #     )
                # outputs.append(x)
                # continue
            x = x / (rms(x) + 1e-6)
            outputs.append(x)
        return e3nn.IrrepsArray.from_list(input.irreps, outputs, input.shape[:-1])


def rms(x: jnp.ndarray) -> jnp.ndarray:
    # x.shape == (..., mul, dim)
    norms_sqr = jnp.sum(x**2, axis=-1, keepdims=True)  # sum over dim
    mean_norm_sqr = jnp.mean(norms_sqr, axis=-2, keepdims=True)  # mean over mul
    vectors_rms = jnp.sqrt(jnp.where(mean_norm_sqr == 0.0, 1.0, mean_norm_sqr))
    assert vectors_rms.shape == x.shape[:-2] + (1, 1)
    return vectors_rms


from typing import Tuple
import flax.linen as nn
from tensorclouds.tensorcloud import TensorCloud

import e3nn_jax as e3nn
import jax.numpy as jnp


class ApproximateTimeEmbed(nn.Module):

    timesteps: int

    @nn.compact
    def __call__(self, state: TensorCloud, t: float) -> TensorCloud:
        irreps_array = state.irreps_array
        mask = state.mask_irreps_array

        num_scalars = irreps_array.filter("0e").array.shape[-1]
        t = jnp.floor(t * self.timesteps).astype(jnp.int32)
        t_emb = nn.Embed(self.timesteps, num_scalars)(t)

        t_emb = t_emb * mask[..., None]
        irreps_array = e3nn.concatenate(
            (t_emb, irreps_array),
            axis=-1,
        ).regroup()
        return state.replace(irreps_array=irreps_array)


class OnehotTimeEmbed(nn.Module):

    timesteps: int = 1000
    time_range: Tuple[int] = (0.0, 1.0)

    @nn.compact
    def __call__(self, state: TensorCloud, t: float) -> TensorCloud:
        irreps_array = state.irreps_array
        mask = state.mask
        t_emb = e3nn.soft_one_hot_linspace(
            t.astype(jnp.float32),
            start=self.time_range[0],
            end=self.time_range[1],
            number=self.timesteps,
            basis="cosine",
            cutoff=True,
        )

        t_emb = t_emb * mask[..., None]
        irreps_array = e3nn.concatenate(
            (t_emb, irreps_array),
            axis=-1,
        ).regroup()
        return state.replace(irreps_array=irreps_array)

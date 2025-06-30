from typing import Tuple

import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as jnp

from tensorclouds.tensorcloud import TensorCloud


class Embed(nn.Module):

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        pass


class PairwiseEmbed(nn.Module):

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        pass


class ApproximateTimeEmbed(Embed):

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


class OnehotTimeEmbed(Embed):

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


from einops import repeat
from jaxtyping import Array


class SequenceDistanceEmbed(PairwiseEmbed):

    k: int
    dim: int

    @nn.compact
    def __call__(self, state: TensorCloud) -> Array:
        seq_len = state.shape[0]
        mask = state.mask
        cross_mask = mask[..., None] * mask[:, None]

        seq_pos_i = repeat(jnp.arange(seq_len), "i -> i j", j=seq_len)
        seq_pos_j = repeat(jnp.arange(seq_len), "j -> i j", i=seq_len)

        relative_seq_pos = seq_pos_i - seq_pos_j
        k_seq = self.k

        relative_seq_pos = jnp.where(
            jnp.abs(relative_seq_pos) <= k_seq, relative_seq_pos, 0
        )
        relative_seq_pos = jnp.where(cross_mask, relative_seq_pos, 0)

        relative_seq_pos = relative_seq_pos + k_seq
        relative_seq_pos = nn.Embed(num_embeddings=2 * k_seq + 1, features=self.dim)(
            relative_seq_pos
        )

        return relative_seq_pos


class SpatialDistanceEmbed(PairwiseEmbed):

    dim: int
    basis: str = "gaussian"
    cut: float = 24.0

    @nn.compact
    def __call__(self, state: TensorCloud) -> Array:
        coord, seq_len = state.coord, state.shape[0]

        coord_i = repeat(coord, "i d -> i j d", j=seq_len)
        coord_j = repeat(coord, "j d -> i j d", i=seq_len)

        mask_coord_i = repeat(state.mask_coord, "i -> i j", j=seq_len)
        mask_coord_j = repeat(state.mask_coord, "j -> i j", i=seq_len)
        cross_mask = mask_coord_i & mask_coord_j

        vectors = (coord_i - coord_j) * cross_mask[..., None]
        norm_sqr = jnp.sum(vectors**2, axis=-1)
        norm = jnp.where(
            norm_sqr == 0.0, 0.0, jnp.sqrt(jnp.where(norm_sqr == 0.0, 1.0, norm_sqr))
        )

        rad_embed = e3nn.soft_one_hot_linspace(
            norm,
            start=0.0,
            end=self.cut,
            number=self.dim,
            basis=self.basis,
            cutoff=True,
        )

        return rad_embed

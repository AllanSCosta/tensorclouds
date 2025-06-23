from typing import Callable

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import repeat

from ..tensorcloud import TensorCloud


class EquivariantSelfAttention(nn.Module):

    irreps_out: e3nn.Irreps
    num_heads: int = 8
    k_seq: int = 16

    radial_cut: float = 20.0
    radial_bins: int = 32
    radial_basis: str = "gaussian"
    edge_irreps: e3nn.Irreps = e3nn.Irreps("0e + 1e")
    norm: bool = True
    activation: Callable = jax.nn.tanh
    envelope: bool = False
    move: bool = False

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_len = state.irreps_array.shape[0]
        assert state.mask.shape == (seq_len,)
        assert state.coord.shape == (seq_len, 3)

        features, coord = state.irreps_array, state.coord

        q = e3nn.flax.Linear(features.irreps)(features).mul_to_axis(self.num_heads)
        q = e3nn.IrrepsArray(q.irreps, repeat(q.array, "i ... -> i j ...", j=seq_len))

        k = e3nn.flax.Linear(features.irreps)(features).mul_to_axis(self.num_heads)
        k = e3nn.IrrepsArray(k.irreps, repeat(k.array, "j ... -> i j ...", i=seq_len))

        v = e3nn.flax.Linear(features.irreps)(features).mul_to_axis(self.num_heads)
        v = e3nn.IrrepsArray(v.irreps, repeat(v.array, "j ... -> i j ...", i=seq_len))

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

        ang_embed = e3nn.spherical_harmonics(
            self.edge_irreps, vectors, True, "component"
        )
        ang_embed = ang_embed * cross_mask[..., None].astype(ang_embed.array.dtype)

        # q = q + e3nn.flax.Linear(features.irreps)(ang_embed).mul_to_axis(self.num_heads)
        # k = k + e3nn.flax.Linear(features.irreps)(ang_embed).mul_to_axis(self.num_heads)
        # v = v + e3nn.flax.Linear(features.irreps)(ang_embed).mul_to_axis(self.num_heads)
        ang_embed = e3nn.flax.Linear(self.num_heads * ang_embed.irreps)(
            ang_embed
        ).mul_to_axis(self.num_heads)
        v = e3nn.concatenate((v, ang_embed), axis=-1).regroup()

        irreps_in = features.irreps
        score = (q.array * k.array).sum(-1) / jnp.sqrt(irreps_in.num_irreps)
        score = jnp.where(cross_mask[..., None], score, -jnp.inf)

        # bias attention weights based on invariants
        rad_embed = e3nn.soft_one_hot_linspace(
            norm,
            start=0.0,
            end=self.radial_cut,
            number=self.radial_bins,
            basis=self.radial_basis,
            cutoff=True,
        )

        seq_pos_i = repeat(jnp.arange(seq_len), "i -> i j", j=seq_len)
        seq_pos_j = repeat(jnp.arange(seq_len), "j -> i j", i=seq_len)

        relative_seq_pos = seq_pos_i - seq_pos_j
        k_seq = self.k_seq

        relative_seq_pos = jnp.where(
            jnp.abs(relative_seq_pos) <= k_seq, relative_seq_pos, 0
        )
        relative_seq_pos = jnp.where(cross_mask, relative_seq_pos, 0)

        relative_seq_pos = relative_seq_pos + k_seq
        relative_seq_pos = nn.Embed(
            num_embeddings=2 * k_seq + 1, features=self.radial_bins
        )(relative_seq_pos)

        invariants = e3nn.concatenate([relative_seq_pos, rad_embed], axis=-1).regroup()
        attention_bias = (
            e3nn.flax.MultiLayerPerceptron(
                [score.shape[-1]],
                self.activation,
                with_bias=True,
                output_activation=True,
            )(invariants)
            * cross_mask[..., None]
        )

        score = score + attention_bias.array
        attention_weights = jax.nn.softmax(score, where=cross_mask[..., None], axis=1)

        messages = attention_weights[..., None] * v
        messages = e3nn.IrrepsArray(messages.irreps, messages.array.sum(-3))

        messages = messages.axis_to_mul()
        new_features = e3nn.flax.Linear(self.irreps_out)(messages)

        if self.move:
            update = e3nn.flax.Linear("1e")(new_features).array
            new_coord = state.coord + update
            state = state.replace(coord=new_coord)

        return state.replace(irreps_array=new_features)

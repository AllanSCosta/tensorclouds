from typing import Callable, Tuple

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import repeat, rearrange

from tensorclouds.nn.embed import PairwiseEmbed

from ..tensorcloud import TensorCloud


class EquivariantSelfAttention(nn.Module):

    irreps_out: e3nn.Irreps
    num_heads: int = 8

    attn_bias: Tuple[PairwiseEmbed] = tuple()
    activation: Callable = jax.nn.silu
    
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
        vecs = (coord_i - coord_j) * cross_mask[..., None]

        eij = e3nn.spherical_harmonics('1e', vecs, True, "component")
        eij = eij * cross_mask[..., None].astype(eij.array.dtype)
        eij = repeat(eij.array, "i j d -> i j h () d", h=self.num_heads)
        eij = e3nn.IrrepsArray(f'1x1e', eij[..., 0, :])
        v = e3nn.concatenate([v, eij], axis=-1).regroup()

        irreps_in = features.irreps
        score = (q.array * k.array).sum(-1) / jnp.sqrt(irreps_in.num_irreps)
        score = jnp.where(cross_mask[..., None], score, -jnp.inf)

        # bias attention weights based on invariants
        attention_bias = [fn(state) for fn in self.attn_bias]
        attention_bias = e3nn.concatenate(attention_bias, axis=-1).regroup()
        attention_bias = (
            e3nn.flax.MultiLayerPerceptron(
                [score.shape[-1]],
                self.activation,
                with_bias=True,
                output_activation=True,
            )(attention_bias)
            * cross_mask[..., None]
        )

        score = score + attention_bias.array
        attention_weights = jax.nn.softmax(score, where=cross_mask[..., None], axis=1)

        messages = attention_weights[..., None] * v
        messages = e3nn.IrrepsArray(messages.irreps, messages.array.sum(-3))

        messages = messages.axis_to_mul()
        new_features = e3nn.flax.Linear(self.irreps_out)(messages)

        return state.replace(irreps_array=new_features)

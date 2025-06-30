from functools import reduce
from typing import List

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp

from ..tensorcloud import TensorCloud
from .attention import EquivariantSelfAttention
from .embed import Embed, PairwiseEmbed
from .feed_forward import FeedForward
from .residual import Residual


class TransformerBlock(nn.Module):

    irreps: e3nn.Irreps
    ff_factor: int
    attn_bias: List[PairwiseEmbed]
    move: bool = False

    @nn.compact
    def __call__(self, x: TensorCloud) -> TensorCloud:
        x = Residual(
            EquivariantSelfAttention(
                irreps_out=self.irreps, attn_bias=self.attn_bias, move=self.move
            )
        )(x)
        return Residual(FeedForward(self.irreps, self.ff_factor))(x)


class Transformer(nn.Module):

    irreps: e3nn.Irreps
    depth: int
    ff_factor: int

    attn_bias: List[PairwiseEmbed]
    move: bool = False

    @nn.compact
    def __call__(self, x: TensorCloud) -> TensorCloud:
        return reduce(
            lambda x, _: TransformerBlock(
                irreps=self.irreps,
                attn_bias=self.attn_bias,
                ff_factor=self.ff_factor,
                move=self.move,
            )(x),
            range(self.depth),
            x,
        )

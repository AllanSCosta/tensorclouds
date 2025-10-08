from functools import reduce
from typing import Tuple

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
    num_heads: int = 4

    attn_bias: Tuple[PairwiseEmbed] = tuple()
    ff: nn.Module = FeedForward
    move: bool = False

    @nn.compact
    def __call__(self, x: TensorCloud) -> TensorCloud:
        x = Residual(
            EquivariantSelfAttention(
                irreps_out=self.irreps,
                attn_bias=self.attn_bias,
                num_heads=self.num_heads,
            )
        )(x)

        if self.move:
            # update = e3nn.flax.Linear("1e")(x.irreps_array)
            # new_coord = x.coord + update.array
            # x = x.replace(coord=new_coord)

            vecs_irreps = e3nn.Irreps(self.irreps).filter(keep='1e')
            gate_irreps = e3nn.Irreps(f"{vecs_irreps.num_irreps}x0e")
            update = e3nn.flax.Linear("1e")(
                e3nn.gate(
                    e3nn.flax.Linear(gate_irreps + vecs_irreps)(x.irreps_array),
                    even_gate_act=jax.nn.gelu,
                )
            )
            new_coord = x.coord + update.array
            x = x.replace(coord=new_coord)

        return Residual(self.ff(self.irreps, self.ff_factor))(x)



class Transformer(nn.Module):

    irreps: e3nn.Irreps
    depth: int
        
    num_heads: int = 4
    attn_bias: Tuple[PairwiseEmbed] = tuple()

    ff: nn.Module = FeedForward
    ff_factor: int = 4
    pre_ff: bool = True

    move: bool = False

    @nn.compact
    def __call__(self, x: TensorCloud) -> TensorCloud:
        # x = x.replace(
        #     irreps_array=e3nn.flax.Linear(self.irreps)(x.irreps_array)
        # )
        # print('Transformer: ', x.irreps)
        # if self.pre_ff:
        #     x = Residual(self.ff(self.irreps, self.ff_factor))(x)
        return reduce(
            lambda x, _: TransformerBlock(
                irreps=self.irreps,
                attn_bias=self.attn_bias,
                num_heads=self.num_heads,
                ff=self.ff,
                ff_factor=self.ff_factor,
                move=self.move,
            )(x),
            range(self.depth),
            x,
        )

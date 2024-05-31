import functools
import jax
from typing import List, Tuple
from .mix import MixingBlock
from .self_interaction import SelfInteraction


import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp

from .sequence_convolution import SequenceConvolution
from .mix import MixingBlock
from moleculib.protein.datum import ProteinDatum
from .layer_norm import EquivariantLayerNorm

from ..tensorcloud import TensorCloud 
from .utils import multiscale_irreps

class DecoderBlock(hk.Module):
    def __init__(
        self,
        irreps_out,
        kernel_size,
    ):

        super().__init__()
        self.irreps_out = irreps_out
        self.kernel_size = kernel_size

    def __call__(self, state, skip, skip_mask):
        if skip is None:
            skip = state.replace(
                irreps_array=e3nn.IrrepsArray(
                    state.irreps_array.irreps, 
                    jnp.zeros_like(state.irreps_array.array)
                )
            )
        state = state.replace(
            irreps_array=EquivariantLayerNorm()(state.irreps_array + skip.irreps_array * skip_mask)
        )
        state = MixingBlock(
            irreps_out=self.irreps_out,
            kernel_size=self.kernel_size,
            residual=True,
            weighted_coords=False,
        )(state)
        return state, state


class Decoder(hk.Module):
    def __init__(
        self,
        irreps: e3nn.Irreps,
        layers: List[int],
        rescale: float,
        stride: int,
        kernel_size: int,
        skip_connections: bool = False,
    ):
        super().__init__()

        self.irreps = e3nn.Irreps(irreps)
        self.layers = layers

        self.tree_depth = len(layers)
        self.rescale = rescale
        self.stride = stride
        self.kernel_size = kernel_size
        self.skip_connections = skip_connections 

        self.dropout = 0.3
        self.list_irreps = multiscale_irreps(
            self.irreps, self.tree_depth - 1, self.rescale, 0
        )[::-1]
        assert len(self.list_irreps) == self.tree_depth

    def __call__(
        self,
        skips: List[TensorCloud] = None,
        ground: ProteinDatum = None,
        is_training: bool = False,
    ):

        acc_stride = self.stride**self.tree_depth

        if type(skips) == TensorCloud:
            skips = [skips] + [None] * (len(self.layers) - 1)

        state = skips[0]
        internals = [(state, )]

        if self.skip_connections:
            if is_training:
                will_skip = jax.random.uniform(
                    hk.next_rng_key(),
                    shape=(1,),
                    minval=0,
                    maxval=1,
                ) > self.dropout
                skip_bound = jax.random.randint(
                    hk.next_rng_key(),
                    shape=(1,),
                    minval=0,
                    maxval=(len(self.layers) + 1),
                )
                skip_masks = (
                    will_skip 
                    | (skip_bound <= jnp.arange(len(self.layers)))
                ).astype(jnp.bfloat16)[::-1]
            else:
                skip_masks = jnp.ones(
                    len(self.layers),
                ).astype(jnp.bfloat16)
        else:
            skip_masks = jnp.zeros(
                len(self.layers),
            ).astype(jnp.bfloat16)

        for idx, num_blocks in enumerate(self.layers):
            irreps_in = self.list_irreps[idx]

            skip = skips[idx]
            skip_mask = skip_masks[idx]
            
            decoder_block = hk.transform(
                DecoderBlock(
                    irreps_out=irreps_in,
                    kernel_size=self.kernel_size,
                )
            )

            init_block = jax.vmap(decoder_block.init, in_axes=(0, None, None, None))
            init_block = hk.lift(init_block, name=f"decoder_loop_{idx}")
            init_rng = hk.next_rng_keys(num_blocks)

            def apply_block(state, input):
                params, rng_key = input
                return decoder_block.apply(
                    params, rng_key, state, skip, skip_mask
                )

            state, _ = jax.lax.scan(
                apply_block,
                state,
                (init_block(init_rng, state, skip, skip_mask), init_rng),
            )

            internals.append(state)

            if idx < len(self.layers) - 1:
                irreps_out = self.list_irreps[idx + 1]
                state = SequenceConvolution(
                    irreps_out,
                    stride=self.stride,
                    kernel_size=self.kernel_size,
                    mode="valid",
                    transpose=True,
                )(state)
                acc_stride *= self.stride

        return state, internals
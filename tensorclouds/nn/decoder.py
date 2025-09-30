import functools
import jax
from typing import List, Tuple
from .mix import MixingBlock


import e3nn_jax as e3nn
from flax import linen as nn
import jax.numpy as jnp

from .sequence_convolution import TransposeSequenceConvolution
from .mix import MixingBlock
from moleculib.protein.datum import ProteinDatum
from tensorclouds.nn.layer_norm import EquivariantLayerNorm

from tensorclouds.tensorcloud import TensorCloud
from .utils import multiscale_irreps


class DecoderBlock(nn.Module):
    irreps_out: e3nn.Irreps
    kernel_size: int

    @nn.compact
    def __call__(self, state, skip, skip_mask):
        if skip is None:
            skip = state.replace(
                irreps_array=e3nn.IrrepsArray(
                    state.irreps_array.irreps, jnp.zeros_like(state.irreps_array.array)
                )
            )
        state = state.replace(
            irreps_array=EquivariantLayerNorm()(
                state.irreps_array + skip.irreps_array * skip_mask
            )
        )
        state = MixingBlock(
            irreps_out=self.irreps_out,
            kernel_size=self.kernel_size,
            residual=True,
            weighted_coords=False,
        )(state)
        return state, state


class Decoder(nn.Module):
    
    irreps: e3nn.Irreps
    layers: List[int]
    rescale: float
    stride: int
    kernel_size: int
    skip_connections: bool

    def setup(
        self,
    ):
        self.tree_depth = len(self.layers)
        self.dropout = 0.3
        self.list_irreps = multiscale_irreps(
            self.irreps, self.tree_depth - 1, self.rescale, 0
        )[::-1]

    @nn.compact
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
        internals = [(state,)]

        if self.skip_connections:
            if is_training:
                will_skip = (
                    jax.random.uniform(
                        self.make_rng("will_skip"),
                        shape=(1,),
                        minval=0,
                        maxval=1,
                    )
                    > self.dropout
                )
                skip_bound = jax.random.randint(
                    self.make_rng("skip_bound"),
                    shape=(1,),
                    minval=0,
                    maxval=(len(self.layers) + 1),
                )
                skip_masks = (
                    will_skip | (skip_bound <= jnp.arange(len(self.layers)))
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

            state, _ = DecoderBlock(
                irreps_out=irreps_in,
                kernel_size=self.kernel_size,
            )(state, skip, skip_mask)

            internals.append(state)

            if idx < len(self.layers) - 1:
                irreps_out = self.list_irreps[idx + 1]
                state = TransposeSequenceConvolution(
                    irreps_out,
                    stride=self.stride,
                    kernel_size=self.kernel_size,
                    mode="valid",
                )(state)
                acc_stride *= self.stride

        return state, internals

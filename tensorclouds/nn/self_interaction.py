from functools import reduce
from typing import List, Tuple

import e3nn_jax as e3nn
from flax import linen as nn
import jax

from .layer_norm import EquivariantLayerNorm
from .residual import Residual

from einops import rearrange
import jax.numpy as jnp

from ..tensorcloud import TensorCloud 



class SelfInteraction(nn.Module):

    layers: Tuple[e3nn.Irreps]
    chunk_factor: int = 0
    residual: bool = True
    full_square: bool = False
    norm: bool = True
    norm_last: bool = False

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_length = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_length, state.irreps_array.irreps.dim)
        assert state.mask_coord.shape == (seq_length,)
        assert state.coord.shape == (seq_length, 3)

        for idx, irreps in enumerate(self.layers):
            last_layer = idx == len(self.layers) - 1

            block = _SelfInteractionBlock(
                irreps, 
                chunk_factor=self.chunk_factor, 
                full_square=self.full_square, 
            )
            if self.residual:
                state = Residual(block)(state)
            else:
                state = block(state)

            if ((not last_layer) or self.norm_last) and self.norm:
                state = state.replace(
                    irreps_array=EquivariantLayerNorm()(state.irreps_array)
                )

        return state


class _SelfInteractionBlock(nn.Module):
    irreps_out: e3nn.Irreps 
    chunk_factor: int = 0 
    full_square: bool = False 
    symmetric_compression: bool = False

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_length = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_length, state.irreps_array.irreps.dim)
        assert state.coord.shape == (seq_length, 3)
        assert state.mask_coord.shape == (seq_length,)

        features = state.irreps_array

        dims = [irrep.mul for irrep in features.irreps]
        assert len(dims) > 0

        if self.full_square:
            if self.chunk_factor != 0.0:
                features = features.mul_to_axis(self.chunk_factor)
                channel_mix = e3nn.tensor_square(features)
                features = e3nn.concatenate((features, channel_mix), axis=-1).regroup()
                features = features.axis_to_mul()
            else:
                channel_mix = e3nn.tensor_square(features)
                features = e3nn.concatenate((features, channel_mix), axis=-1).regroup()
        else:
            if reduce(lambda x, y: x == y, dims):
                channel_mix = e3nn.tensor_square(
                    features.mul_to_axis(dims[0])
                ).axis_to_mul()
                features = e3nn.concatenate((features, channel_mix), axis=-1).regroup()

        invariants = features.filter(keep="0e").regroup()
        features *= e3nn.flax.MultiLayerPerceptron(
            [invariants.irreps.dim, features.irreps.num_irreps], act=jax.nn.silu
        )(invariants)
        features = e3nn.flax.Linear(self.irreps_out)(features)

        state = state.replace(irreps_array=features)

        return state

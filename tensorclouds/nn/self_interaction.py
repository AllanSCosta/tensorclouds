from functools import reduce
from typing import List

import e3nn_jax as e3nn
import haiku as hk
import jax

from .layer_norm import EquivariantLayerNorm
from .residual import Residual

from einops import rearrange
import jax.numpy as jnp

from ..tensor_cloud import TensorCloud 

class SelfInteractionUNet(hk.Module):

    def __init__(
        self, 
        irreps_in: e3nn.Irreps,
        num_layers: 4,
        rescale_factor: float = 0.8,
        *,
        norm: bool = True,
        norm_last: bool = False,
    ):
        super().__init__()
        if type(irreps_in) == str:
            irreps_in = e3nn.Irreps(irreps_in)
        self.irreps_in = irreps_in
        self.num_layers = num_layers
        self.rescale_factor = rescale_factor

        irreps = self.irreps_in
        irreps_list = []
        print('Building UNet')
        for i in range(self.num_layers):
            print(f'Layer [{i}] irreps: {irreps}')
            irreps_list.append(irreps)
            irreps = e3nn.Irreps([(int(mul * self.rescale_factor), ir) for mul, ir in irreps]) 
        self.layers = irreps_list

        self.norm = norm
        self.norm_last = norm_last
        
        self.full_square = True
        self.chunk_factor = 0

    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_length = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_length, state.irreps_array.irreps.dim)
        assert state.mask.shape == (seq_length,)
        assert state.coord.shape == (seq_length, 3)

        skips = []

        for idx, irreps in enumerate(self.layers):
            # reduce
            state = _SelfInteractionUNet(
                irreps, 
                full_square=False, 
            )(state)

            if self.norm:
                state = state.replace(
                    irreps_array=EquivariantLayerNorm()(state.irreps_array)
                )            

            skips.append(state)

        # update bottleneck
        state = _SelfInteractionUNet(
            state.irreps_array.irreps, 
            full_square=False, 
        )(state)
        
        decoder_layers = list(reversed(self.layers[:-1])) + [self.irreps_in]
        skips = list(reversed(skips))

        for idx, (irreps, skip) in enumerate(zip(decoder_layers, skips)):
            last_layer = idx == len(self.layers) - 1

            # add skip
            state = state.replace(
                irreps_array = state.irreps_array + skip.irreps_array
            )

            # expand
            state = _SelfInteractionUNet(
                irreps, 
                full_square=False, 
            )(state)

            if ((not last_layer) or self.norm_last) and self.norm:
                state = state.replace(
                    irreps_array=EquivariantLayerNorm()(state.irreps_array)
                )            

        return state



class _SelfInteractionUNet(hk.Module):
    def __init__(self, irreps_out: e3nn.Irreps, chunk_factor: int = 0, full_square: bool = False, symmetric_compression: bool = False):
        super().__init__()
        self.irreps_out = e3nn.Irreps(irreps_out)
        self.chunk_factor = chunk_factor
        self.full_square = full_square
        self.symmetric_compression = symmetric_compression
        

    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_length = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_length, state.irreps_array.irreps.dim)
        assert state.mask.shape == (seq_length,)
        assert state.coord.shape == (seq_length, 3)

        features = state.irreps_array

        dims = [irrep.mul for irrep in features.irreps]
        assert len(dims) > 0

        if reduce(lambda x, y: x == y, dims):
            if self.full_square:
                channel_mix = e3nn.tensor_square(features)
                features = e3nn.concatenate((features, channel_mix), axis=-1).regroup()
            else:
                channel_mix = e3nn.tensor_square(
                    features.mul_to_axis(dims[0])
                ).axis_to_mul()
                features = e3nn.concatenate((features, channel_mix), axis=-1).regroup()
        
        features = e3nn.haiku.Linear(self.irreps_out)(features)

        invariants = features.filter(keep="0e").regroup()
        features *= e3nn.haiku.MultiLayerPerceptron(
            [invariants.irreps.dim, features.irreps.num_irreps], act=jax.nn.silu
        )(invariants)

        state = state.replace(irreps_array=features)

        return state


class SelfInteraction(hk.Module):
    def __init__(
        self,
        layers: List[e3nn.Irreps],
        chunk_factor: int = 0,
        *,
        residual: bool = True,
        full_square: bool = False,
        norm: bool = True,
        norm_last: bool,
    ):
        super().__init__()
        if type(layers[0]) == str:
            layers = [e3nn.Irreps(layer) for layer in layers]
        self.layers = layers
        self.norm_last = norm_last
        self.chunk_factor = chunk_factor
        self.residual = residual
        self.full_square = full_square
        self.norm = norm

    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_length = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_length, state.irreps_array.irreps.dim)
        assert state.mask.shape == (seq_length,)
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


class _SelfInteractionBlock(hk.Module):
    def __init__(self, irreps_out: e3nn.Irreps, chunk_factor: int = 0, full_square: bool = False, symmetric_compression: bool = False):
        super().__init__()
        self.irreps_out = e3nn.Irreps(irreps_out)
        self.chunk_factor = chunk_factor
        self.full_square = full_square
        self.symmetric_compression = symmetric_compression
        

    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_length = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_length, state.irreps_array.irreps.dim)
        assert state.mask.shape == (seq_length,)
        assert state.coord.shape == (seq_length, 3)

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
        features *= e3nn.haiku.MultiLayerPerceptron(
            [invariants.irreps.dim, features.irreps.num_irreps], act=jax.nn.silu
        )(invariants)
        features = e3nn.haiku.Linear(self.irreps_out)(features)

        state = state.replace(irreps_array=features)

        return state

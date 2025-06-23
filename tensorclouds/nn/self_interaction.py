from functools import reduce

import e3nn_jax as e3nn
import flax.linen as nn
import jax

from ..tensorcloud import TensorCloud
from .layer_norm import EquivariantLayerNorm


class FullTensorSquareSelfInteraction(nn.Module):

    irreps: e3nn.Irreps
    norm: bool = True

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        features = state.irreps_array
        channel_mix = e3nn.tensor_square(features)
        features = e3nn.concatenate((features, channel_mix), axis=-1).regroup()

        invariants = features.filter(keep="0e").regroup()
        features *= e3nn.flax.MultiLayerPerceptron(
            [invariants.irreps.dim, features.irreps.num_irreps], act=jax.nn.silu
        )(invariants)
        features = e3nn.flax.Linear(self.irreps)(features)

        return state.replace(irreps_array=features)


class ChannelWiseTensorSquareSelfInteraction(nn.Module):

    irreps: e3nn.Irreps
    norm: bool = True

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        features = res = state.irreps_array

        dims = [irrep.mul for irrep in features.irreps]
        channel_mix = e3nn.tensor_square(features.mul_to_axis(dims[0])).axis_to_mul()
        features = e3nn.concatenate((features, channel_mix), axis=-1).regroup()

        scalars = features.filter(keep="0e").regroup()
        features *= e3nn.flax.MultiLayerPerceptron(
            [scalars.irreps.dim, features.irreps.num_irreps],
            act=jax.nn.silu,
        )(scalars)
        features = e3nn.flax.Linear(self.irreps)(features)

        if res.irreps == features.irreps:
            features = res + features
        else:
            features = e3nn.concatenate([res, features])
            features = e3nn.flax.Linear(self.irreps)(features)

        if self.norm:
            features = EquivariantLayerNorm()(features)

        return state.replace(irreps_array=features)


class SegmentedTensorSquareSelfInteraction(nn.Module):

    irreps: e3nn.Irreps
    norm: bool = True
    # num_heads: int =
    segment_size = 2

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        features = state.irreps_array

        channel_mix = [
            e3nn.tensor_square(
                features.filter(keep=ir).mul_to_axis(mul // self.segment_size)
            ).axis_to_mul()
            for (mul, ir) in features.irreps.filter(drop="0e")
        ]
        features = e3nn.concatenate([features, *channel_mix], axis=-1).regroup()

        invariants = features.filter(keep="0e").regroup()
        features *= e3nn.flax.MultiLayerPerceptron(
            [invariants.irreps.dim, features.irreps.num_irreps], act=jax.nn.silu
        )(invariants)
        features = e3nn.flax.Linear(self.irreps)(features)

        if self.norm:
            features = EquivariantLayerNorm()(features)

        return state.replace(irreps_array=features)


class SelfInteraction(nn.Module):

    irreps: e3nn.Irreps
    irreps_out: e3nn.Irreps = None
    depth: int = 1
    norm_last: bool = True

    base: nn.Module = SegmentedTensorSquareSelfInteraction

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        irreps_out = self.irreps_out
        if irreps_out is None:
            irreps_out = self.irreps

        for _ in range(self.depth - 1):
            state = self.base(self.irreps)(state)

        state = self.base(irreps_out)(state)

        return state

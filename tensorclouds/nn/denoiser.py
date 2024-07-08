import functools
import math
from flax import linen as nn
import jax.numpy as jnp
import e3nn_jax as e3nn
from typing import List, Tuple

from tensorclouds.nn.utils import safe_normalize

from .spatial_convolution import CompleteSpatialConvolution, kNNSpatialConvolution
from .self_interaction import SelfInteraction
from ..tensorcloud import TensorCloud 


from moleculib.assembly.datum import AssemblyDatum
from .layer_norm import EquivariantLayerNorm
import jax
from .sequence_convolution import SequenceConvolution


class ApproximateTimeEmbed(nn.Module):

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


class OnehotTimeEmbed(nn.Module):

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
            cutoff=True
        )

        t_emb = t_emb * mask[..., None]
        irreps_array = e3nn.concatenate(
            (t_emb, irreps_array),
            axis=-1,
        ).regroup()
        return state.replace(irreps_array=irreps_array)


import einops as ein


class Denoiser(nn.Module):

    layers: Tuple[e3nn.Irreps]
    k: int = 0
    k_seq: int = 0
    radial_cut: float = 32.0
    timesteps: int = 100
    time_range: Tuple = (0.0, 1.0)
    full_square: bool = False
    conservative: bool = False
    move: bool = False
    pos_encoding: bool = False

    @nn.compact
    def __call__(self, x, t=None, cond=None):
        if cond is not None:
            x = x.replace(irreps_array=e3nn.concatenate([x.irreps_array, cond], axis=-1).regroup())

        # first mix
        x = SelfInteraction(
            [self.layers[0]],
            full_square=True,
            norm_last=False,
        )(x)

        if self.pos_encoding:
            pos = nn.Embed(
                num_embeddings=x.irreps_array.shape[0], 
                features=x.irreps_array.filter('0e').shape[-1]
            )(jnp.arange(x.irreps_array.shape[0]))

            pos = e3nn.IrrepsArray(
                f"{x.irreps_array.filter('0e').shape[-1]}x0e", pos
            )

            x = x.replace(
                irreps_array=e3nn.concatenate((
                    x.irreps_array, pos
                ), axis=-1).regroup() * x.mask_coord[..., None]
            )

        if t is not None:
            x = OnehotTimeEmbed(self.timesteps, self.time_range)(x, t)

        # second mix
        x = SelfInteraction(
            [self.layers[0]],
            full_square=False,
            norm_last=True,
        )(x)

        states = [ x.irreps_array ]

        for irreps in self.layers[:-1]:
            res = x # residual connection

            x = SelfInteraction(
                [irreps],
                full_square=self.full_square,
                norm_last=True,
            )(x)

            if self.k > 0:
                x = kNNSpatialConvolution(
                    irreps_out=irreps,
                    envelope=False,
                    k=self.k,
                    k_seq=self.k_seq,
                    radial_cut=self.radial_cut,
                    radial_bins=32,
                    radial_basis='gaussian',
                    move=self.move,
                )(x)
            else:
                x = CompleteSpatialConvolution(
                    irreps_out=irreps,
                    radial_cut=self.radial_cut,
                    radial_bins=32,
                    k_seq=self.k_seq,
                    move=self.move,
                )(x)

            x = x.replace(
                irreps_array = e3nn.concatenate(
                    (res.irreps_array, x.irreps_array), axis=-1
                ).regroup() * x.mask[..., None]
            )

            x = x.replace(
                irreps_array=EquivariantLayerNorm()(x.irreps_array)
            )
            states.append(x.irreps_array)

        x = x.replace(
            irreps_array=e3nn.flax.Linear(self.layers[-2])(
                e3nn.concatenate(states, axis=-1).regroup()
            )
        )
        
        x = SelfInteraction(
            [self.layers[-2]],
            norm=True,
            norm_last=True,
            full_square=False,
        )(x)
        
        tc_out = SelfInteraction(
            [self.layers[-1]],
            norm=False,
            norm_last=False,
            full_square=True,
        )(x)

        if not self.move:
            coord_out = SelfInteraction(
                ['1x0e + 1x1e'],
                full_square=True,
                norm=False,
                norm_last=False,
            )(x).irreps_array.filter('1e').array
            tc_out = tc_out.replace(coord=coord_out)

        return tc_out



class TwoTrackDenoiser(nn.Module):

    feature_net: nn.Module
    coord_net: nn.Module

    @nn.compact
    def __call__(
        self, x, t, cond=None,
    ):
        pred_feature = self.feature_net(x, t, cond=cond).irreps_array
        pred_coord = self.coord_net(x, t, cond=cond).coord

        feature_mask = e3nn.IrrepsArray(
            f'{x.irreps_array.irreps.num_irreps}x0e', x.mask_irreps_array
        )
        coord_mask = x.mask_coord[..., None]
        return x.replace(
            irreps_array=feature_mask * pred_feature,
            coord=coord_mask * pred_coord,
        )



class FourTrackDenoiser(nn.Module):

    feature_net1: nn.Module
    feature_net2: nn.Module

    coord_net1: nn.Module
    coord_net2: nn.Module

    @nn.compact
    def __call__(
        self, x, t, cond=None,
    ):
        pred_feature1 = self.feature_net1(x, t, cond=cond).irreps_array
        pred_feature2 = self.feature_net2(x, t, cond=cond).irreps_array

        pred_coord1 = self.coord_net1(x, t, cond=cond).coord
        pred_coord2 = self.coord_net2(x, t, cond=cond).coord

        feature_mask = e3nn.IrrepsArray(
            f'{x.irreps_array.irreps.num_irreps}x0e', x.mask_irreps_array
        )
        coord_mask = x.mask_coord[..., None]

        x1 = x.replace(
            irreps_array=feature_mask * pred_feature1,
            coord=coord_mask * pred_coord1,
        )

        x2 = x.replace(
            irreps_array=feature_mask * pred_feature2,
            coord=coord_mask * pred_coord2,
        )

        return x1, x2
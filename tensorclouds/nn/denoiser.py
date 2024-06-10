import functools
import math
import haiku as hk
import jax.numpy as jnp
import e3nn_jax as e3nn
from typing import List

from .spatial_convolution import IPA, CompleteSpatialConvolution, kNNSpatialConvolution
from .self_interaction import SelfInteraction
from ..tensorcloud import TensorCloud 


from moleculib.assembly.datum import AssemblyDatum
from .layer_norm import EquivariantLayerNorm
import jax
from .sequence_convolution import SequenceConvolution


class ApproximateTimeEmbed(hk.Module):
    def __init__(self, timesteps: int):
        super().__init__()
        self.timesteps = timesteps

    def __call__(self, state: TensorCloud, t: float) -> TensorCloud:
        irreps_array = state.irreps_array
        mask = state.mask_irreps_array

        num_scalars = irreps_array.filter("0e").array.shape[-1]
        t = jnp.floor(t * self.timesteps).astype(jnp.int32)
        t_emb = hk.Embed(self.timesteps, num_scalars)(t)

        t_emb = t_emb * mask[..., None]
        irreps_array = e3nn.concatenate(
            (t_emb, irreps_array),
            axis=-1,
        ).regroup()
        return state.replace(irreps_array=irreps_array)


class OnehotTimeEmbed(hk.Module):
    def __init__(self, timesteps: int, time_range=(0.0, 1.0)):
        super().__init__()
        self.timesteps = timesteps
        self.time_range = time_range

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

class Denoiser(hk.Module):

    def __init__(
        self,
        layers: List[e3nn.Irreps],
        k: int = 0,
        k_seq: int = 0,
        radial_cut = 32.0,
        timesteps = 100,
        time_range = (0.0, 1.0),
        full_square=False,
        conservative=False,
        move=False,
    ):
        super().__init__()
        self.layers = layers
        self.radial_cut = radial_cut
        self.time_embed = OnehotTimeEmbed(timesteps, time_range=time_range)
        self.full_square=full_square
        self.conservative = conservative
        self.k = k
        self.k_seq = k_seq
        self.move = False

    def mix(self, x, t, cond):

        if cond is not None:
            x = x.replace(irreps_array=e3nn.concatenate([x.irreps_array, cond], axis=-1).regroup())                
        
        # frame = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # frame = ein.repeat(frame, 'e c -> n (e c)', n=x.irreps_array.shape[0])
        # reference_frame = e3nn.IrrepsArray(
        #     "3x1e", frame)
        # x = x.replace(
        #     irreps_array=e3nn.concatenate(
        #         (x.irreps_array, reference_frame), axis=-1
        #     ).regroup()
        # )

        # first mix
        x = SelfInteraction(
            [self.layers[0]],
            full_square=True,
            norm_last=False,
        )(x)

        pos = hk.Embed(
            vocab_size=x.irreps_array.shape[0], 
            embed_dim=x.irreps_array.filter('0e').shape[-1]
        )(jnp.arange(x.irreps_array.shape[0]))

        pos = e3nn.IrrepsArray(
            f"{x.irreps_array.filter('0e').shape[-1]}x0e", pos
        )

        x = x.replace(
            irreps_array=e3nn.concatenate((
                x.irreps_array, pos
            ), axis=-1).regroup() * x.mask_irreps_array[..., None]
        )

        x = self.time_embed(x, t)

        # second mix
        x = SelfInteraction(
            [self.layers[0]],
            full_square=False,
            norm_last=True,
        )(x)

        states = [ x.irreps_array ]

        for irreps in self.layers:
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
                ).regroup() * x.mask_irreps_array[..., None]
            )

            x = x.replace(
                irreps_array=EquivariantLayerNorm()(x.irreps_array)
            )
            states.append(x.irreps_array)

        x = x.replace(
            irreps_array=e3nn.haiku.Linear(self.layers[-2])(
                e3nn.concatenate(states, axis=-1).regroup()
            )
        )
        return x

    def predict_noise(self, x, t, cond):
        x = self.mix(x, t, cond=cond)
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

    def predict_energy(self, x, t, cond=None):
        x = self.mix(x, t, cond=cond)
        energy_out = SelfInteraction(
            [x.irreps_array.irreps, '1x0e'],
            full_square=False,
            norm_last=False,
        )(x).irreps_array.array[..., 0].sum(-1)
        return energy_out

    def predict_force(self, x, cond, t):
        gradients = jax.grad(functools.partial(self.predict_energy, t=t, cond=cond), allow_int=True)(x)
        gradients = TensorCloud(
            irreps_array=gradients.irreps_array,
            mask_irreps_array=x.mask_irreps_array,
            coord=gradients.coord,
            mask_coord=x.mask_coord,
        )
        return gradients

    def __call__(self, x, t, cond=None):
        if self.conservative:
            return self.predict_force(x, t=t, cond=cond)
        else:
            return self.predict_noise(x, t=t, cond=cond)

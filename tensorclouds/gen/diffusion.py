import functools
import jax
import jax.numpy as jnp
import haiku as hk

import e3nn_jax as e3nn
from ..base.utils import TensorCloud, inner_split
from einops import rearrange

from typing import List

from .so3_diffusion import SO3Diffuser
from .r3_diffusion import R3Diffuser
from .constants import compute_constants
from .utils import DiffusionStepOutput


class TensorCloudDiffuser(hk.Module):

    def __init__(
        self,
        feature_net: hk.Module,
        coord_net: hk.Module,
        irreps: e3nn.Irreps,
        rescale = 10.0,
        timesteps=1000,
        start_at=1.0,
        leading_shape=(1,),
    ):
        super().__init__()

        self.feature_net = feature_net()
        self.coord_net = coord_net()

        self.start_at = start_at
        self.num_timesteps = timesteps
        self.leading_shape = leading_shape

        self.feature_diffuser = SO3Diffuser(
            irreps=irreps,
            timesteps=timesteps,
            leading_shape=leading_shape,
        )

        self.coord_diffuser = R3Diffuser(
            rescale=rescale,
            timesteps=timesteps,
            leading_shape=leading_shape,
        )

        for key, val in compute_constants(timesteps, start_at=self.start_at).items():
            setattr(self, key, val)

    def p_sample(self, z, t, cond=None, conditioners=[]):
        feature_noise_pred = self.feature_net(z, t=t, cond=cond).irreps_array
        coord_noise_pred = self.coord_net(z, t=t, cond=cond).coord

        irreps_array = self.feature_diffuser.p_sample(z.irreps_array, t, feature_noise_pred, z.mask_irreps_array)[0]
        coord = self.coord_diffuser.p_sample(z.coord, t, coord_noise_pred, z.mask_coord)[0]

        if len(conditioners) > 0:
            new_coord = coord
            for conditioner in conditioners:
                conditioner_velocity = conditioner(coord)
                new_coord = new_coord + conditioner_velocity
            coord = new_coord

        z = TensorCloud(
            irreps_array=irreps_array,
            mask_irreps_array=z.mask_irreps_array,
            coord=coord,
            mask_coord=z.mask_coord,
        )
        return z, z

    def sample_prior(self):
        xT = TensorCloud(
            irreps_array=self.feature_diffuser.sample_prior(),
            mask_irreps_array=jnp.ones(self.leading_shape, dtype=jnp.bool_),
            coord=self.coord_diffuser.sample_prior(),
            mask_coord=jnp.ones(self.leading_shape, dtype=jnp.bool_),
        )
        return xT

    def sample(
            self, 
            cond: e3nn.IrrepsArray = None, 
            conditioners: List = [],
            x0: TensorCloud = None,
        ):
        if x0 is not None:
            zT, _, _ = self.q_sample(x0, self.num_timesteps - 1)
        else:
            zT = self.sample_prior()

        return hk.scan(
            functools.partial(self.p_sample, conditioners=conditioners, cond=cond),
            zT,
            jnp.arange(0, self.num_timesteps)[::-1],
        )

    def q_sample(self, z0, t: int):
        features, so3_noise = self.feature_diffuser.q_sample(z0.irreps_array, z0.mask_irreps_array, t)
        coord, coord_noise = self.coord_diffuser.q_sample(z0.coord, z0.mask_coord, t)
        z0 = TensorCloud(
            irreps_array=features,
            mask_irreps_array=z0.mask_irreps_array,
            coord=coord,
            mask_coord=z0.mask_coord,
        )
        return z0, so3_noise, coord_noise

    def __call__(
            self, 
            x0: TensorCloud, 
            cond: e3nn.IrrepsArray = None,
            is_training = False
        ):
        t = jax.random.randint(hk.next_rng_key(), (), 0, self.num_timesteps)

        xt, feature_noise, coord_noise = self.q_sample(x0, t)
        feature_noise_pred = self.feature_net(xt, t=t, cond=cond).irreps_array
        coord_noise_pred = self.coord_net(xt, t=t, cond=cond).coord

        return DiffusionStepOutput(
            noise_prediction=TensorCloud(
                irreps_array=feature_noise_pred,
                mask_irreps_array=x0.mask_irreps_array,
                coord=coord_noise_pred,
                mask_coord=x0.mask_irreps_array,
            ),
            noise=TensorCloud(
                irreps_array=feature_noise,
                mask_irreps_array=x0.mask_irreps_array,
                coord=coord_noise,
                mask_coord=x0.mask_irreps_array,
            ),
            reweight=self.loss_weight[t][None]
        )




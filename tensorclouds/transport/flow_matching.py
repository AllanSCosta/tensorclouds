import functools
import jax
import jax.numpy as jnp
from flax import linen as nn

import e3nn_jax as e3nn
from tensorclouds.random.normal import NormalDistribution
from tensorclouds.random.harmonic import HarmonicDistribution
from ..tensorcloud import TensorCloud
from tensorclouds.utils import align_with_rotation

from typing import List


import chex


@chex.dataclass
class ModelPrediction:
    prediction: TensorCloud
    target: TensorCloud
    reweight: float = 1.0


from typing import Tuple



class TensorCloudFlowMatcher(nn.Module):

    network: nn.Module
    irreps: e3nn.Irreps
    leading_shape: Tuple[int]
    var_features: float
    var_coords: float

    def setup(self):
        self.dist = NormalDistribution(
            irreps_in=self.irreps,
            irreps_mean=e3nn.zeros(self.irreps),
            irreps_scale=self.var_features,
            coords_mean=jnp.zeros(3),
            coords_scale=self.var_coords,
        )

        # self.dist = HarmonicDistribution(
        #     irreps=self.irreps,
        #     var_features=self.var_features,
        #     N = leading_shape[-1],
        # )
    

    def sample(
        self,
        cond: e3nn.IrrepsArray = None,
        num_steps: int = 100,
        mask_features: jnp.array = None,
        mask_coord: jnp.array = None,
    ):
        dt = 1 / num_steps
      
        def update_one_step(network: nn.Module, xt: TensorCloud, t: float) -> TensorCloud:
            v̂t = network(xt, t, cond=cond) 
            next_xt = xt + dt * v̂t
            return next_xt, next_xt

        x0 = self.dist.sample(
            self.make_rng(),
            leading_shape=self.leading_shape,
            mask_features=mask_features,
            mask_coord=mask_coord,
        )

        ts = jnp.arange(0, 1, dt)

        return nn.scan(
            update_one_step,
            variable_broadcast="params",
            split_rngs={"params": True},
        )(self.network, x0, ts)

    def p_t(self, x1, t: int, sigma_min: float = 1e-2):
        x0 = self.dist.sample(
            self.make_rng(),
            leading_shape=self.leading_shape,
            mask_coord=x1.mask_coord,
            mask_features=x1.mask_irreps_array,
        )
        x0 = x0.centralize()
        x0, x1 = align_with_rotation(x0, x1)
        xt = t * x1 + (1 - t) * x0
        vt = (x1 + (-x0))
        return xt, vt
    
    def __call__(
        self, x1: TensorCloud, cond: e3nn.IrrepsArray = None, is_training=False
    ):
        x1 = x1.centralize()
        t = jax.random.uniform(self.make_rng())
        xt, vt = self.p_t(x1, t)
        v̂t = self.network(xt, t, cond=cond)

        return ModelPrediction(
            prediction=v̂t,
            target=vt,
            reweight=1,
        )

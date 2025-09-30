import functools
from typing import List

import chex
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import linen as nn

from tensorclouds.random.harmonic import HarmonicDistribution
from tensorclouds.random.normal import NormalDistribution
from tensorclouds.utils import align_with_rotation

from ..tensorcloud import TensorCloud


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

    def sample(
        self,
        cond: e3nn.IrrepsArray = None,
        num_steps: int = 100,
        mask_features: jnp.array = None,
        mask_coord: jnp.array = None,
    ):
        dt = 1 / num_steps

        def update_one_step(
            network: nn.Module, xt: TensorCloud, t: float
        ) -> TensorCloud:
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
        vt = x1 + (-x0)
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


class ReparameterizedTensorCloudFlowMatcher(nn.Module):

    network: nn.Module  # must output two tensorclouds
    leading_shape: Tuple[int] = None
    var_features: float = 1.0
    var_coords: float = 1.0

    def sample(
        self,
        x0=None,
        cond=None,
        eps: float = 1.0,
        num_steps: int = 1000,
    ) -> Tuple[TensorCloud, TensorCloud]:
        dt = 1.0 / num_steps

        def update_one_step(
            network: nn.Module, zt: TensorCloud, t: float
        ) -> TensorCloud:
            x1_hat = network(zt, t, cond=cond)
            s = t + dt
            coeff = 1 / (1 - t + 1e-4)
            next_zt = coeff * (s - t) * x1_hat + coeff * (1 - s) * zt
            next_zt = next_zt.centralize()
            next_zt = next_zt.replace(irreps_array=next_zt.irreps_array * zt.mask_irreps_array)
            return next_zt, next_zt

        z = NormalDistribution(
            irreps_in=x0.irreps,
            irreps_mean=e3nn.zeros(x0.irreps),
            irreps_scale=self.var_features,
            coords_mean=jnp.zeros(3),
            coords_scale=self.var_coords,
        ).sample(
            self.make_rng(),
            leading_shape=x0.mask_coord.shape,
            mask_coord=x0.mask_coord,
            mask_features=x0.mask_irreps_array,
        )
        x0 = x0 + z

        return nn.scan(
            update_one_step,
            variable_broadcast="params",
            split_rngs={"params": True},
        )(self.network, x0, jnp.arange(0, 1, dt))

    def compute_xt(
        self, t: float, x0: TensorCloud, x1: TensorCloud, eps: float = 1e-4
    ) -> TensorCloud:
        """Computes xt at time t."""
        z = NormalDistribution(
            irreps_in=x1.irreps,
            irreps_mean=e3nn.zeros(x1.irreps),
            irreps_scale=self.var_features,
            coords_mean=jnp.zeros(3),
            coords_scale=self.var_coords,
        ).sample(
            self.make_rng(),
            leading_shape=x1.mask_coord.shape,
            mask_coord=x1.mask_coord,
            mask_features=x1.mask_irreps_array,
        )
        x0 = x0 + z
        interpolant = (1 - t) * x0 + t * x1
        return interpolant, (x1 + (-x0))

    def __call__(
        self,
        x0: TensorCloud,
        x1: TensorCloud,
        is_training=False,
        cond: TensorCloud = None,
        eps: float = 1e-4,
    ):
        # Sample time.
        t = jax.random.uniform(self.make_rng(), minval=0.0 + eps, maxval=1.0 - eps)
        x0 = x0.centralize()
        x1 = x1.centralize()

        # Compute xt at time t.
        xt, b = self.compute_xt(t, x0, x1)
        # drift = self.dtIt(x0, x1) + self.gamma_dot(t) * z

        # Compute the predicted velocity ut(xt) at time t and location xt.
        x1_hat = self.network(xt, t, cond=cond)
        # x1_hat = x1_hat.replace(coord=x1_hat.coord + x0.coord)

        return x1_hat
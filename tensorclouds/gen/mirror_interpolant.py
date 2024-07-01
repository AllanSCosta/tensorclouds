from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
import e3nn_jax as e3nn

from ..tensorcloud import TensorCloud
from ..random.normal import NormalDistribution

import chex


@chex.dataclass
class NoisePrediction:
    noise_prediction: TensorCloud
    noise: TensorCloud


class TensorCloudMirrorInterpolant(nn.Module):

    network: nn.Module
    leading_shape: Tuple[int]
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
            network: nn.Module, zt: TensorCloud, tk: float
        ) -> TensorCloud:
            t, key = tk

            z_hat = network(zt, t, cond=cond)

            gamma_dot = (1 / (2 * jnp.sqrt(t * (1 - t) + 1e-4))) * (1 - 2 * t)
            gamma = jnp.sqrt(t * (1 - t) + 1e-4)

            b_hat = gamma_dot * z_hat

            z = NormalDistribution(
                irreps_in=z_hat.irreps,
                irreps_mean=e3nn.zeros(z_hat.irreps),
                irreps_scale=self.var_features,
                coords_mean=jnp.zeros(3),
                coords_scale=self.var_coords,
            ).sample(
                key,
                leading_shape=self.leading_shape,
                mask_coord=zt.mask_coord,
                mask_features=zt.mask_irreps_array,
            )

            dW = jnp.sqrt(dt) * z

            drift = dt * b_hat
            denoise = -((eps / gamma) * dt) * z_hat
            noise = jnp.sqrt(2 * eps) * dW

            next_zt = zt + drift + (t < 0.99) * (t > 0.01) * (denoise + noise)
            next_zt = next_zt.centralize()

            return next_zt, next_zt

        ts = jnp.arange(0, 1, dt)
        ks = jax.random.split(self.make_rng(), num_steps)

        return nn.scan(
            update_one_step,
            variable_broadcast="params",
            split_rngs={"params": True},
        )(self.network, x0, [ts, ks])

    def compute_xt(self, t: float, x0: TensorCloud, eps: float = 1e-4) -> TensorCloud:
        z = NormalDistribution(
            irreps_in=x0.irreps,
            irreps_mean=e3nn.zeros(x0.irreps),
            irreps_scale=self.var_features,
            coords_mean=jnp.zeros(3),
            coords_scale=self.var_coords,
        ).sample(
            self.make_rng(),
            leading_shape=self.leading_shape,
            mask_features=x0.mask_irreps_array,
            mask_coord=x0.mask_coord,
        )
        interpolant = x0 + jnp.sqrt((1 - t) * t + eps) * z
        return interpolant, z

    def __call__(
        self,
        x0: TensorCloud,
        is_training=False,
        cond: TensorCloud = None,
    ):
        t = jax.random.uniform(self.make_rng())

        x0 = x0.centralize()
        xt, z = self.compute_xt(t, x0)
        pred = self.network(xt, t, cond=cond)

        return NoisePrediction(noise_prediction=pred, noise=z)

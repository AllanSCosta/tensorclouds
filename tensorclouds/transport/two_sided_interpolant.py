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
    prediction: TensorCloud
    target: TensorCloud


@chex.dataclass
class DriftPrediction:
    prediction: TensorCloud
    target: dict


class TensorCloudTwoSidedInterpolant(nn.Module):

    network: nn.Module # must output two tensorclouds
    leading_shape: Tuple[int]
    var_features: float = 1.0
    var_coords: float = 1.0

    def setup(self):
        self.prediction_type = "velocity"
        scheduler = "bsquare"
        if scheduler == "brownian":
            self.gamma = lambda t: jnp.sqrt(t * (1 - t))
            self.gamma_dot = lambda t: (1 / (2 * jnp.sqrt(t * (1 - t) + 1e-2))) * (
                1 - 2 * t
            )
            self.dtIt = lambda x0, x1: -x0 + x1
        elif scheduler == "bsquare":
            self.gamma = lambda t: t * (1 - t) + 1e-4
            self.gamma_dot = lambda t: 1 - 2 * t + 1e-4
            self.dtIt = lambda x0, x1: -x0 + x1

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
            b_hat, z_hat = network(zt, t, cond=cond)

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

            drift = b_hat * dt
            denoise = -((eps / self.gamma(t))) * z_hat * dt
            noise = jnp.sqrt(2 * eps) * dW

            next_zt = zt + drift + (t < 0.9) * (t > 0.1) * (denoise + noise)
            next_zt = next_zt.centralize()

            return next_zt, next_zt

        ts = jnp.arange(0, 1, dt)
        ks = jax.random.split(self.make_rng(), num_steps)

        return nn.scan(
            update_one_step,
            variable_broadcast="params",
            split_rngs={"params": True},
        )(self.network, x0, [ts, ks])

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
            leading_shape=self.leading_shape,
            mask_coord=x1.mask_coord,
            mask_features=x1.mask_irreps_array,
        )

        interpolant = t * x1
        interpolant += (1 - t) * x0
        interpolant += self.gamma(t) * z

        return interpolant, z

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
        xt, z = self.compute_xt(t, x0, x1)
        drift = self.dtIt(x0, x1) + self.gamma_dot(t) * z

        # Compute the predicted velocity ut(xt) at time t and location xt.
        drift_pred, noise_pred = self.network(xt, t, cond=cond)

        return (
            NoisePrediction(prediction=noise_pred, target=z),
            DriftPrediction(prediction=drift_pred, target=drift),
        )


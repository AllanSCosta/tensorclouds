from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import haiku as hk
import e3nn_jax as e3nn

from ..tensorcloud import TensorCloud
from ..random.normal import NormalDistribution
from ..utils import align_with_rotation

import chex


@chex.dataclass
class NoisePrediction:
    prediction: TensorCloud
    target: TensorCloud


@chex.dataclass
class DriftPrediction:
    prediction: TensorCloud
    target: dict


class TensorCloudStepInterpolant(hk.Module):

    def __init__(
        self,
        feature_drift: hk.Module,
        coord_drift: hk.Module,
        feature_denoise: hk.Module,
        coord_denoise: hk.Module,
        leading_shape: tuple[int, ...],
        var_features: float = 1.0,
        var_coords: float = 1.0,
        name: str | None = None,
    ):
        super().__init__(name)
        self.leading_shape = leading_shape

        self.feature_drift = feature_drift()
        self.coord_drift = coord_drift()

        self.feature_denoise = feature_denoise()
        self.coord_denoise = coord_denoise()

        self.var_features = var_features
        self.var_coords = var_coords

        self.prediction_type = "velocity"

        scheduler = "brownian"

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
        elif scheduler == "const":
            self.gamma = lambda t: 1.0
            self.gamma_dot = lambda t: 0.0
            self.dtIt = lambda x0, x1: -x0 + x1

    def predict_drift(
        self,
        x,
        t,
        cond=None,
    ):
        return x.replace(
            irreps_array=self.feature_drift(x, t, cond=cond).irreps_array,
            coord=self.coord_drift(x, t, cond=cond).coord,
        )

    def predict_denoise(
        self,
        x,
        t,
        cond=None,
    ):
        return x.replace(
            irreps_array=self.feature_denoise(x, t, cond=cond).irreps_array,
            coord=self.coord_denoise(x, t, cond=cond).coord,
        )

    def sample(
        self,
        x0=None,
        cond=None,
        eps: float = 1.0,
        num_steps: int = 1000,
    ) -> Tuple[TensorCloud, TensorCloud]:
        """Sample from the base distribution and then push forward along the velocity field."""

        def update_one_step(zt: TensorCloud, t: float) -> TensorCloud:
            """Use a Euler integrator to update z(t) to z(t + 1)."""
            dt = 1.0 / num_steps

            z_hat = self.predict_denoise(zt, t, cond=cond)
            b_hat = self.predict_drift(zt, t, cond=cond)

            z = NormalDistribution(
                irreps_in=z_hat.irreps,
                irreps_mean=e3nn.zeros(z_hat.irreps),
                irreps_scale=self.var_features,
                coords_mean=jnp.zeros(3),
                coords_scale=self.var_coords,
            ).sample(hk.next_rng_key(), leading_shape=self.leading_shape)

            dW = jnp.sqrt(dt) * z
            drift = dt * b_hat
            denoise = -((eps / self.gamma(t)) * dt) * z_hat
            noise = jnp.sqrt(2 * eps) * dW

            next_zt = zt + drift #+ (t < 0.8) * (t > 0.2) * (denoise + noise)
            next_zt = next_zt.centralize()

            next_zt = next_zt.replace(
                irreps_array=e3nn.IrrepsArray(
                    next_zt.irreps_array.irreps,
                    next_zt.irreps_array.array * (zt.irreps_array.array != 0.0),
                )
            )

            return next_zt, next_zt
        x0 = self.q_sample(x0)
        x0 = x0.centralize()
        
        return hk.scan(
            update_one_step,
            x0,
            jnp.linspace(0, 1, num_steps),
        )

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
            hk.next_rng_key(),
            leading_shape=self.leading_shape,
            mask=x0.mask_irreps_array,
        )

        interpolant = t * x1
        interpolant += (1 - t) * x0
        interpolant += self.gamma(t) * z

        return interpolant, z

    def q_sample(self, x1):
        z = NormalDistribution(
            irreps_in=x1.irreps,
            irreps_mean=e3nn.zeros(x1.irreps),
            irreps_scale=self.var_features,
            coords_mean=jnp.zeros(3),
            coords_scale=self.var_coords,
        ).sample(
            hk.next_rng_key(),
            leading_shape=self.leading_shape,
            mask=x1.mask_irreps_array,
        )
        return z

    def __call__(
        self,
        x0: TensorCloud,
        x1: TensorCloud = None,
        is_training=False,
        cond: TensorCloud = None,
        eps: float = 1e-1,
    ):

        if x1 is None:
            x1 = x0

        x0 = self.q_sample(x1)
        x0, x1 = align_with_rotation(x0, x1)

        # Sample time.
        t = jax.random.uniform(hk.next_rng_key())
        x0 = x0.centralize()
        x1 = x1.centralize()

        # Compute xt at time t.
        xt, z = self.compute_xt(t, x0, x1)
        drift = self.dtIt(x0, x1) + self.gamma_dot(t) * z

        # Compute the predicted velocity ut(xt) at time t and location xt.
        drift_pred = self.predict_drift(xt, t, cond=cond)
        noise_pred = self.predict_denoise(xt, t, cond=cond)

        return (
            NoisePrediction(prediction=noise_pred, target=z),
            DriftPrediction(prediction=drift_pred, target=drift),
        )

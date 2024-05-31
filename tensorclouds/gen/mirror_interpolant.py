from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import haiku as hk
import e3nn_jax as e3nn

from ..tensorcloud import TensorCloud
from .r3_diffusion import _centralize_coord
from ..random.normal import NormalDistribution

import chex

@chex.dataclass
class NoisePrediction:
    noise_prediction: TensorCloud
    noise: TensorCloud


class TensorCloudMirrorInterpolant(hk.Module):

    def __init__(
        self,
        feature_net: hk.Module,
        coord_net: hk.Module,
        leading_shape: tuple[int, ...],
        var_features: float = 1.0,
        var_coords: float = 1.0,
        name: str | None = None,
    ):
        super().__init__(name)
        self.leading_shape = leading_shape
        
        self.feature_net = feature_net()
        self.coord_net = coord_net()
                
        self.var_features = var_features
        self.var_coords = var_coords  

    def make_prediction(
        self, x, t, cond=None,
    ):
        pred_feature = self.feature_net(x, t, cond=cond)
        pred_coord = self.coord_net(x, t, cond=cond)
        return x.replace(
            irreps_array=pred_feature.irreps_array,
            coord=pred_coord.coord,
        )

    def sample(
        self, 
        x0=None, 
        cond=None, 
        eps: float=1.0,
        num_steps: int=1000,
    ) -> Tuple[TensorCloud, TensorCloud]:
        """Sample from the base distribution and then push forward along the velocity field."""

        def update_one_step(zt: TensorCloud, t: float) -> TensorCloud:
            """Use a Euler integrator to update z(t) to z(t + 1)."""
            dt = 1.0 / num_steps
            z_hat = self.make_prediction(zt, t, cond=cond)
            
            gamma_dot = (1/(2*jnp.sqrt(t*(1-t)+1e-4))) * (1-2*t)
            gamma = jnp.sqrt(t*(1-t) + 1e-4)

            b_hat = gamma_dot * z_hat

            z = NormalDistribution(
                irreps_in=z_hat.irreps,
                irreps_mean=e3nn.zeros(z_hat.irreps),
                irreps_scale=self.var_features,
                coords_mean=jnp.zeros(3),
                coords_scale=self.var_coords,
            ).sample(
                hk.next_rng_key(), 
                leading_shape=self.leading_shape
            ) 

            dW = jnp.sqrt(dt) * z

            drift = dt * b_hat
            denoise = - ((eps / gamma) * dt) * z_hat
            noise = jnp.sqrt(2 * eps) * dW

            next_zt = zt + drift + (t < 0.9) * (t > 0.1) * (denoise + noise) 

            next_zt = next_zt.replace(
                coord=_centralize_coord(next_zt.coord, next_zt.mask_coord)
            )
            return next_zt, next_zt

        return hk.scan(
            update_one_step,
            self.sample_base_distribution() if x0 is None else x0,
            jnp.linspace(0, 1, num_steps),
        )

    def compute_xt(
        self, t: float, x0: TensorCloud, eps: float = 1e-4
    ) -> TensorCloud:
        
        """Computes xt at time t."""
        z = NormalDistribution(
            irreps_in=x0.irreps,
            irreps_mean=e3nn.zeros(x0.irreps),
            irreps_scale=self.var_features,
            coords_mean=jnp.zeros(3),
            coords_scale=self.var_coords,
        ).sample(
            hk.next_rng_key(), 
            leading_shape=self.leading_shape,
            mask=x0.mask_irreps_array
        )
        interpolant = x0 + jnp.sqrt((1-t) * t + eps) * z
        return interpolant, z

    def __call__(
        self,
        x0: TensorCloud,
        is_training=False,
        cond: TensorCloud = None,
        eps: float = 1e-4,
    ):
        t = jax.random.uniform(hk.next_rng_key())

        x0 = x0.replace(coord=_centralize_coord(x0.coord, mask=x0.mask_coord))
        xt, z = self.compute_xt(t, x0)

        pred = self.make_prediction(xt, t, cond=cond)

        return NoisePrediction(
            noise_prediction=pred,
            noise=z
        )

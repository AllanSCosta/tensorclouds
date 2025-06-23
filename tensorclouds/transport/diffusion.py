import functools
from typing import List, Tuple

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import linen as nn

from tensorclouds.random.normal import NormalDistribution

from ..tensorcloud import TensorCloud

# ==========================
# ADAPTED BY ALLAN COSTA
# ORIGINAL AUTHOR OF THE SNIPPET: lucidrains
# github.com/lucidrains/denoising-diffusion-pytorch/
# ==========================


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return jnp.linspace(beta_start, beta_end, timesteps)


def sigmoid_beta_schedule(timesteps, start=0, end=3, tau=0.3, clamp_min=1e-5):
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps, dtype=jnp.float32) / timesteps
    v_start = jax.nn.sigmoid(jnp.array(start / tau))
    v_end = jax.nn.sigmoid(jnp.array(end / tau))
    alphas_cumprod = (-jax.nn.sigmoid((t * (end - start) + start) / tau) + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


def compute_constants(timesteps, start_at=1.0, scheduler=linear_beta_schedule):
    assert start_at > 0.0 and start_at <= 1.0

    betas = scheduler(int(timesteps / start_at))
    alphas = 1.0 - betas

    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], (1, 0), constant_values=1.0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = jnp.sqrt(1.0 / alphas)
    sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod - 1)
    posterior_mean_coef1 = (
        betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod)
    )
    posterior_log_variance_clipped = jnp.log(jnp.clip(posterior_variance, a_min=1e-20))
    snr = alphas_cumprod / (1 - alphas_cumprod)
    clipped_snr = jnp.clip(snr, a_max=5.0)
    constants = dict(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        posterior_variance=posterior_variance,
        sqrt_recip_alphas=sqrt_recip_alphas,
        sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
        snr=clipped_snr,
        loss_weight=clipped_snr / snr,
    )
    for key, value in constants.items():
        constants[key] = value[:timesteps]
    return constants


import chex


@chex.dataclass
class ModelPrediction:
    prediction: TensorCloud
    target: dict
    reweight: float


class TensorCloudDiffuser(nn.Module):

    network: nn.Module
    irreps: e3nn.Irreps
    var_features: float
    var_coords: float
    timesteps: int = 1000
    leading_shape: Tuple = (1,)

    def setup(self):
        for key, val in compute_constants(self.timesteps, start_at=1.0).items():
            setattr(self, key, val)
        self.normal = NormalDistribution(
            irreps_in=self.irreps,
            irreps_mean=e3nn.zeros(self.irreps),
            irreps_scale=self.var_features,
            coords_mean=jnp.zeros(3),
            coords_scale=self.var_coords,
        )

    def sample(
        self,
        cond: e3nn.IrrepsArray = None,
        mask_coord=None,
        mask_features=None,
    ):

        def update_one_step(
            network: nn.Module, xt: TensorCloud, tk: Tuple
        ) -> TensorCloud:
            t, key = tk
            z = self.normal.sample(
                key,
                leading_shape=self.leading_shape,
                mask_coord=xt.mask_coord,
                mask_features=xt.mask_irreps_array,
            )

            ϵ̂ = network(xt, t, cond=cond)

            αt = self.alphas[t]
            ᾱt = self.alphas_cumprod[t]
            σt = jnp.exp(0.5 * self.posterior_log_variance_clipped[t])

            sqrt = lambda x: jnp.sqrt(jnp.maximum(x, 1e-6))

            next_xt = (1 / sqrt(αt)) * (
                xt + (-((1 - αt) / sqrt(1 - ᾱt))) * ϵ̂
            ).centralize() + (t != 0) * σt * z
            next_xt = next_xt.centralize()

            return next_xt, next_xt

        zT = self.normal.sample(
            self.make_rng(),
            leading_shape=self.leading_shape,
            mask_coord=mask_coord,
            mask_features=mask_features,
        )

        ts = jnp.arange(0, self.timesteps)[::-1]
        ks = jax.random.split(self.make_rng(), self.timesteps)

        return nn.scan(
            update_one_step,
            variable_broadcast="params",
            split_rngs={"params": True},
        )(self.network, zT, [ts, ks])

    def q_sample(self, x0, t: int):
        z = self.normal.sample(
            self.make_rng(),
            leading_shape=self.leading_shape,
            mask_coord=x0.mask_coord,
            mask_features=x0.mask_irreps_array,
        )
        return (
            self.sqrt_alphas_cumprod[t] * x0 + self.sqrt_one_minus_alphas_cumprod[t] * z
        ), z

    def __call__(
        self, x0: TensorCloud, cond: e3nn.IrrepsArray = None, is_training=False
    ):
        t = jax.random.randint(self.make_rng(), (), 0, self.timesteps)

        x0 = x0.centralize()
        xt, z = self.q_sample(x0, t)
        ẑ = self.network(xt, t, cond=cond)

        return ModelPrediction(
            prediction=ẑ, target=z, reweight=self.loss_weight[t][None]
        )

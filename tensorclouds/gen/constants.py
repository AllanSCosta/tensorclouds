# ==========================
# ADAPTED BY ALLAN COSTA
# ORIGINAL AUTHOR OF THE SNIPPET: lucidrains
# github.com/lucidrains/denoising-diffusion-pytorch/
# ==========================

import jax
import jax.numpy as jnp


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

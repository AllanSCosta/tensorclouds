import haiku as hk
import jax.numpy as jnp
import jax
import functools

from .constants import compute_constants

def _centralize_coord(coord, mask):
    center = (coord * mask[..., None]).sum(-2) / mask.sum(-1)
    coord = (coord - center[None, ...]) * mask[..., None]
    coord = jnp.where(jnp.isnan(coord), 0.0, coord)
    return coord


class R3Diffuser(hk.Module):
    def __init__(
        self,
        rescale = 5.0,
        timesteps=1000,
        leading_shape=(14,),
    ):
        super().__init__()
        self.leading_shape = leading_shape
        self.rescale = rescale

        self.num_timesteps = timesteps
        for key, val in compute_constants(timesteps).items():
            setattr(self, key, val)

    def p_sample(self, z, t, noise_pred, mask):
        z0_pred = (
            self.sqrt_recip_alphas_cumprod[t] * z -
            self.sqrt_recipm1_alphas_cumprod[t] * noise_pred
        )
        z0_pred = _centralize_coord(z0_pred, mask)
        log_var = self.posterior_log_variance_clipped[t]

        mean_coord = (
            self.posterior_mean_coef1[t] * z0_pred
            + self.posterior_mean_coef2[t] * z
        )

        noise = jax.random.normal(
            hk.next_rng_key(), 
            self.leading_shape + (3,)
        ) * self.rescale
        z = (mean_coord + jnp.exp(0.5 * log_var) * noise) * mask[..., None] 
        
        return z, z

    def q_sample(self, z0, mask, t: int):
        z0 = _centralize_coord(z0, mask)
        coord_noise = jax.random.normal(
            hk.next_rng_key(), 
            self.leading_shape + (3,)
        ).astype(z0.dtype)
        coord_noise = coord_noise * self.rescale
        coord_noise = coord_noise * mask[..., None]
        return (
            self.sqrt_alphas_cumprod[t] * z0
            + self.sqrt_one_minus_alphas_cumprod[t] * coord_noise
        ), coord_noise

    def sample_prior(self):
        zT = jax.random.normal(
            hk.next_rng_key(), 
            self.leading_shape + (3,)
         ) * self.rescale
        return zT

    def sample(self, mask):
        return hk.scan(
            functools.partial(self.p_sample, mask=mask),
            self.sample_prior(),
            jnp.arange(0, self.num_timesteps)[::-1],
        )

    def __call__(self, z0, mask):
        t = jax.random.randint(hk.next_rng_key(), (), 0, self.num_timesteps)
        z0 = _centralize_coord(z0, mask)   
        zt, noise = self.q_sample(z0, t)
        state = self.time_embed(zt, t)
        noise_pred = self.f(state)
        return self.loss(noise_pred, noise, z0.mask_coord, t)  
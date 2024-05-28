import e3nn_jax as e3nn
import haiku as hk
from .constants import compute_constants
import jax.numpy as jnp
import functools 
import jax


class SO3Diffuser(hk.Module):

    def __init__(
        self,
        irreps: e3nn.Irreps,
        timesteps=1000,
        leading_shape=(1,),
        f=None
    ):
        super().__init__()
        self.irreps = irreps
        self.leading_shape = leading_shape

        self.num_timesteps = timesteps
        for key, val in compute_constants(timesteps).items():
            setattr(self, key, val)

    def p_sample(self, z: e3nn.IrrepsArray, t: int, noise_pred: e3nn.IrrepsArray, mask: jnp.ndarray):
        z0_pred = (
            self.sqrt_recip_alphas_cumprod[t] * z -
            self.sqrt_recipm1_alphas_cumprod[t] * noise_pred
        )

        mean = (
            self.posterior_mean_coef1[t] * z0_pred
            + self.posterior_mean_coef2[t] * z
        ) 

        noise = e3nn.normal(
            self.irreps, hk.next_rng_key(), self.leading_shape
        )
        log_var = self.posterior_log_variance_clipped[t]
        z = (mean + jnp.exp(0.5 * log_var) * noise * (jnp.array([t]) > 0)[..., None])
        z = z * mask[..., None]

        return z, z

    def sample_prior(self):
        zT = e3nn.normal(self.irreps, hk.next_rng_key(), self.leading_shape)
        return zT

    def sample(self):
        return hk.scan(
            functools.partial(self.p_sample),
            self.sample_prior(),
            jnp.arange(0, self.num_timesteps)[::-1],
        )

    def q_sample(self, z0, mask: jnp.ndarray, t: int):
        noise = e3nn.normal(
            self.irreps,
            hk.next_rng_key(),
            self.leading_shape,
            dtype=z0.dtype,
        ) * mask[..., None]
        zt = (
            self.sqrt_alphas_cumprod[t] * z0
            + self.sqrt_one_minus_alphas_cumprod[t] * noise
        )
        return zt, noise

    def __call__(self, z0: e3nn.IrrepsArray, condition: e3nn.IrrepsArray = None):
        t = jax.random.randint(hk.next_rng_key(), (), 0, self.num_timesteps)
        zt, noise = self.q_sample(z0, t)
        noise_pred = self.f(zt, t)
        return self.loss(noise_pred, noise, t)

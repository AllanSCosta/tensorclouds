from ..tensorcloud import TensorCloud
import jax.numpy as jnp
import e3nn_jax as e3nn
import jax


class HarmonicDistribution:
    """
    Harmonic Prior by Jing et al
    Modified from https://github.com/bjing2016/alphaflow/blob/2c27c69a8adc24d77e1f583b78e0a8d675c76869/alphaflow/utils/diffusion.py#L40
    """

    def __init__(
        self,
        irreps,
        N=256,
        a=3 / (3.8**2),
        var_features=1.0,
    ):

        J = jnp.zeros((N, N))
        for i, j in zip(jnp.arange(N - 1), jnp.arange(1, N)):
            J = J.at[i, i].set(a + J[i, i])
            J = J.at[j, j].set(a + J[j, j])
            J = J.at[j, i].set(-a + J[j, i])
        D, P = jnp.linalg.eigh(J)
        D_inv = 1 / D
        D_inv = D_inv.at[0].set(0)
        self.P, self.D_inv = P, D_inv
        self.N = N

        self.irreps = irreps
        self.var_features = var_features

    def sample(self, key, leading_shape=(), mask_coord=None, mask_features=None):
        if mask_coord is None:
            mask_coord = jnp.ones(leading_shape, dtype=bool)
        if mask_features is None:
            mask_features = jnp.ones(leading_shape + self.irreps.shape, dtype=bool)

        irreps_key, coords_key = jax.random.split(key)
        features = (
            e3nn.normal(self.irreps, leading_shape=leading_shape, key=irreps_key)
            * self.var_features
        )

        coord = self.P @ (
            jnp.sqrt(self.D_inv)[:, None]
            * jax.random.normal(coords_key, (*leading_shape, 3))
        )

        return TensorCloud(
            irreps_array=features,
            mask_irreps_array=mask_features,
            coord=coord,
            mask_coord=mask_coord,
        )

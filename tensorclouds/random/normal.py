import chex
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from typing import Optional, Tuple

from ..tensorcloud import TensorCloud, TensorClouds


class NormalDistribution:
    """A normal distribution for TensorCloud."""

    def __init__(
        self,
        irreps_in: e3nn.Irreps,
        irreps_scale: float = 1.0,
        coords_scale: float = 1.0,
        coords_mean: Optional[chex.Array] = None,
        irreps_mean: Optional[e3nn.IrrepsArray] = None,
    ):
        irreps_in = e3nn.Irreps(irreps_in)
        self.irreps_in = irreps_in
        if irreps_mean is None:
            irreps_mean = e3nn.zeros(irreps_in)
        self.irreps_mean = irreps_mean
        if coords_mean is None:
            coords_mean = jnp.zeros(3)
        self.coords_mean = coords_mean
        self.irreps_scale = irreps_scale
        self.coords_scale = coords_scale

    def sample(
        self,
        key: chex.PRNGKey,
        leading_shape: Tuple[int, ...] = (),
        mask_coord: jax.Array = None,
        mask_features: jax.Array = None,
    ) -> TensorCloud:
        """Sample from the distribution."""

        if mask_coord is None:
            mask_coord = jnp.ones(leading_shape, dtype=bool)
        if mask_features is None:
            mask_features = e3nn.IrrepsArray(
                self.irreps_in,
                jnp.ones(leading_shape + (self.irreps_in.num_irreps,), dtype=bool),
            )

        irreps_key, coords_key = jax.random.split(key)
        irreps = (
            e3nn.normal(self.irreps_in, leading_shape=leading_shape, key=irreps_key)
            * self.irreps_scale
        )
        coords = jax.random.normal(coords_key, (*leading_shape, 3)) * self.coords_scale
        # mask_features_arr = (e3nn.ones(self.irreps_in, leading_shape) * mask_features[..., None]).array

        mask_features_arr = (e3nn.ones(self.irreps_in) * mask_features).array
        coords = mask_coord[..., None] * coords
        irreps = e3nn.IrrepsArray(
            self.irreps_in,
            mask_features_arr * irreps.array,
        )
        return TensorCloud(
            irreps_array=irreps,
            mask_irreps_array=mask_features,
            coord=coords,
            mask_coord=mask_coord,
        )

    def sample_like(
        self,
        key: chex.PRNGKey,
        tensorclouds: TensorClouds,
        # leading_shapes: Tuple[int, ...] = (),
        # masks_coord: jax.Array = None,
        # masks_features: jax.Array = None,
    ) -> TensorCloud:
        """Sample from the distribution."""
        tcs = []

        attr = tensorclouds._tensorclouds.keys()
        leading_shapes = tensorclouds.shapes
        mask_coords = tensorclouds.mask_coord
        mask_features_ = tensorclouds.mask_irreps_array

        for attr in attr:

            mask_coord = mask_coords[attr]
            mask_features = mask_features_[attr]
            leading_shape = leading_shapes[attr]

            if mask_coord is None:
                mask_coord = jnp.ones(leading_shape, dtype=bool)
            if mask_features is None:
                mask_features = e3nn.IrrepsArray(
                    self.irreps_in,
                    jnp.ones(leading_shape + (self.irreps_in.num_irreps,), dtype=bool),
                )

            irreps_key, coords_key = jax.random.split(key)
            irreps = (
                e3nn.normal(self.irreps_in, leading_shape=leading_shape, key=irreps_key)
                * self.irreps_scale
            )
            coords = (
                jax.random.normal(coords_key, (*leading_shape, 3)) * self.coords_scale
            )
            # mask_features_arr = (e3nn.ones(self.irreps_in, leading_shape) * mask_features[..., None]).array

            mask_features_arr = (e3nn.ones(self.irreps_in) * mask_features).array
            coords = mask_coord[..., None] * coords
            irreps = e3nn.IrrepsArray(
                self.irreps_in,
                mask_features_arr * irreps.array,
            )
            tcs.append(
                (
                    attr,
                    TensorCloud(
                        irreps_array=irreps,
                        mask_irreps_array=mask_features,
                        coord=coords,
                        mask_coord=mask_coord,
                    ),
                )
            )
        return TensorClouds.create(**dict(tcs))

    def log_likelihood(self, x: TensorCloud) -> chex.Array:
        """Compute the log probability of the input."""
        irreps_log_likelihood = jax.scipy.stats.norm.logpdf(
            x.irreps_array, loc=self.irreps_mean.array, scale=self.irreps_scale
        ).sum(-1)
        coords_log_likelihood = jax.scipy.stats.norm.logpdf(
            x.coord, loc=self.coords_mean, scale=self.coords_scale
        ).sum(-1)
        return irreps_log_likelihood + coords_log_likelihood

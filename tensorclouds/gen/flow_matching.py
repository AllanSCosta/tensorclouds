import functools
import jax
import jax.numpy as jnp
from flax import linen as nn

import e3nn_jax as e3nn
from tensorclouds.random.normal import NormalDistribution
from tensorclouds.random.harmonic import HarmonicDistribution
from ..tensorcloud import TensorCloud

from typing import List


import chex


@chex.dataclass
class ModelPrediction:
    prediction: TensorCloud
    target: TensorCloud
    reweight: float = 1.0


from typing import Tuple


def compute_rotation_for_alignment(x: TensorCloud, y: TensorCloud):
    """Computes the rotation matrix that aligns two point clouds."""

    # We are only interested in the coords.
    x = x.coord * x.mask_coord[:, None]
    y = y.coord * y.mask_coord[:, None]

    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    assert x.shape == y.shape

    # Compute the covariance matrix.
    covariance = jnp.matmul(y.T, x)
    ndim = x.shape[-1]
    assert covariance.shape == (ndim, ndim)

    # Compute the SVD of the covariance matrix.
    u, _, v = jnp.linalg.svd(covariance, full_matrices=True)

    # Compute the rotation matrix that aligns the two point clouds.
    rotation = u @ v
    rotation = rotation.at[:, 0].set(
        rotation[:, 0] * jnp.sign(jnp.linalg.det(rotation))
    )
    assert rotation.shape == (ndim, ndim)

    return rotation


def align_with_rotation(
    x0: TensorCloud,
    x1: TensorCloud,
) -> Tuple[TensorCloud, TensorCloud]:
    """Aligns x0 to x1 via a rotation."""
    R = compute_rotation_for_alignment(x0, x1)
    coord = e3nn.IrrepsArray("1o", x0.coord)
    rotated_coord = coord.transform_by_matrix(R).array
    x0 = x0.replace(
        coord=rotated_coord,
        irreps_array=x0.irreps_array.transform_by_matrix(R),
    )
    return x0, x1


class TensorCloudFlowMatcher(nn.Module):

    feature_net: nn.Module
    coord_net: nn.Module
    irreps: e3nn.Irreps
    var_features: float
    var_coords: float
    timesteps = 1000
    leading_shape = (1,)

    def setup(self):
        self.dist = NormalDistribution(
            irreps_in=self.irreps,
            irreps_mean=e3nn.zeros(self.irreps),
            irreps_scale=self.var_features,
            coords_mean=jnp.zeros(3),
            coords_scale=self.var_coords,
        )
        # self.dist = HarmonicDistribution(
        #     irreps=self.irreps,
        #     var_features=self.var_features,
        #     N = leading_shape[-1],
        # )

    def make_prediction(self, x, t, cond=None):
        pred_feature = self.feature_net(x, t, cond=cond)
        pred_coord = self.coord_net(x, t, cond=cond)
        return x.replace(
            irreps_array=x.mask_irreps_array * pred_feature.irreps_array,
            coord=x.mask_coord * pred_coord.coord,
        )

    def sample(
        self,
        cond: e3nn.IrrepsArray = None,
        num_steps: int = 1000,
        mask_features: jnp.array = None,
        mask_coord: jnp.array = None,
    ):
        dt = 1 / num_steps

        def update_one_step(xt: TensorCloud, t: float) -> TensorCloud:
            s = t + dt
            x̂t = self.make_prediction(xt, t, cond=cond)
            next_xt = ((s - t) / (1 - t)) * x̂t + ((1 - s) / (1 - t)) * xt
            next_xt = (t < 1.0) * next_xt + (t >= 1.0) * x̂t
            # v̂t = self.make_prediction(xt, t, cond=cond)
            # next_xt = xt + dt * v̂t
            return next_xt, next_xt

        x0 = self.dist.sample(
            self.make_rng(),
            leading_shape=self.leading_shape,
            mask_features=mask_features,
            mask_coord=mask_coord,
        )

        ts = jnp.arange(0, 1, dt)

        return nn.scan(
            update_one_step,
            variable_broadcast="params",
            split_rngs={"params": True},
        )(self.network, x0, ts)

    def p_t(self, x1, t: int, sigma_min: float = 1e-2):
        x0 = self.dist.sample(
            self.make_rng(),
            leading_shape=self.leading_shape,
            mask_coord=x1.mask_coord,
            mask_features=x1.mask_irreps_array,
        )
        x0 = x0.centralize()
        x0, x1 = align_with_rotation(x0, x1)
        xt = t * x1 + (1 - t) * x0
        vt = x1 + (-x0)
        return xt, vt, x0

    def __call__(
        self, x1: TensorCloud, cond: e3nn.IrrepsArray = None, is_training=False
    ):
        x1 = x1.centralize()
        t = jax.random.randint(self.make_rng(), (), 0, self.num_timesteps)

        xt, v1, x0 = self.p_t(x1, t)
        x̂1 = self.make_prediction(xt, t, cond=cond)

        # xt, vt = self.p_t(x1, t)
        # v̂t = self.make_prediction(xt, t, cond=cond)

        return ModelPrediction(
            # prediction=v̂t,
            # target=vt,
            prediction=x̂1,
            target=x1,
            reweight=1,
        )

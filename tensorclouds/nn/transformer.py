import jax
import flax.linen as nn
import e3nn_jax as e3nn
import jax.numpy as jnp
from ..tensorcloud import TensorCloud


class FeedForward(nn.Module):

    irreps: e3nn.Irreps
    factor: int = 4

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        features = res = state.irreps_array

        features = e3nn.flax.Linear(self.factor * self.irreps)(features)
        
        gate_ = []
        for (_, ir), x in zip(features.irreps, features.list):
            norms_sqr = jnp.sum(x**2, axis=-1)
            norms_ = jnp.sqrt(jnp.where(norms_sqr == 0.0, 1.0, norms_sqr))
            norms_ = jnp.where(norms_sqr == 0.0, 0.0, norms_)
            gate = e3nn.flax.MultiLayerPerceptron([norms_.shape[-1]], act=jax.nn.sigmoid)(norms_)
            gate_.append(gate)
        gate = jnp.concatenate(gate_, axis=-1)
        features = (gate * features)



        features = e3nn.flax.Linear(self.irreps)(features)

        return state.replace(
            irreps_array=features,
            # mask_irreps_array=jnp.ones_like(gate).astype(jnp.bool_),
        )


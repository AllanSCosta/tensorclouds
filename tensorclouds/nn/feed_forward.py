import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
from flax import linen as nn
from ..tensorcloud import TensorCloud


class FeedForward(nn.Module):

    irreps: e3nn.Irreps
    factor: int = 4

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        features = state.irreps_array

        expansion_irreps = self.factor * features.irreps
        features = e3nn.flax.Linear(
            expansion_irreps,
            parameter_initializer=jax.nn.initializers.he_normal,
        )(features)
        
        features = e3nn.norm_activation(features, [jax.nn.silu, jnp.tanh], normalization='norm')

        features = e3nn.flax.Linear(
            self.irreps, 
            parameter_initializer=jax.nn.initializers.he_normal,
        )(features)

        return state.replace(
            irreps_array=features,
        )

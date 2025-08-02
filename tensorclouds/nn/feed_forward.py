import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
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
        )(features)
        features = e3nn.norm_activation(
            features, [jax.nn.gelu, jnp.tanh], normalization="norm"
        )
        features = e3nn.flax.Linear(
            self.irreps,
        )(features)
        return state.replace(
            irreps_array=features,
        )
        
import einops as ein


class CrossedFeedForward(nn.Module):

    irreps: e3nn.Irreps
    factor: int = 4

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        features = state.irreps_array
        expansion_irreps = (self.factor // 2) * features.irreps 
        
        vecs = features.filter("1e")
        gate_mul = vecs.irreps.num_irreps
        gate_irreps = e3nn.Irreps(f'{gate_mul}x0e')
        features = e3nn.flax.Linear(
            gate_irreps + expansion_irreps,
        )(features)
        features = e3nn.gate(features)

        vecs = ein.rearrange(vecs.array, "... (c e) -> ... c e", e=3)
        norms2 = jnp.sum(vecs**2, axis=-1)
        features = e3nn.concatenate([features, norms2], axis=-1).regroup()

        features = e3nn.flax.Linear(
            self.irreps,
        )(features)

        return state.replace(
            irreps_array=features,
        )

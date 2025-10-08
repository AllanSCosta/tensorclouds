import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import linen as nn
import einops as ein
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





class CrossedFeedForward(nn.Module):

    irreps: e3nn.Irreps
    factor: int = 4

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        features = state.irreps_array
        factor = self.factor // 2

        expansion_irreps = factor * features.irreps 

        vecs = features.filter("1e")
        gate_mul = vecs.irreps.num_irreps * (factor // 2)
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
    


# class CrossedFeedForward(nn.Module):

#     irreps: e3nn.Irreps
#     factor: int = 4

#     @nn.compact
#     def __call__(self, state: TensorCloud) -> TensorCloud:        
#         feats = state.irreps_array

#         expansion_irreps = self.factor * feats.irreps
#         feats = e3nn.flax.Linear(expansion_irreps)(feats)
#         feats = feats.mul_to_axis(2)
#         clear_feats, cross_feats = feats[..., 0, :], feats[..., 1, :]

#         cleared = []
#         for x in clear_feats.list:
#             cleared.append(jnp.tanh(jnp.sum(x**2, axis=-1, keepdims=True)) * x)
#         cleared = e3nn.IrrepsArray.from_list(clear_feats.irreps, cleared, clear_feats.shape[:-1])

#         acts = []
#         for x in cross_feats.list: 
#             acts.append(jnp.tanh(jnp.sum(x**2, axis=-1)))

#         crossed = []
#         for ir, act in zip(cross_feats.irreps, acts[::1]): 
#             crossed.append(cross_feats.filter(keep=ir) * act)
#         crossed = e3nn.concatenate(crossed, axis=-1).regroup()

#         feats = e3nn.concatenate([cleared, crossed], axis=-1).regroup()
#         feats = e3nn.flax.Linear(
#             self.irreps,
#         )(feats)

#         return state.replace(
#             irreps_array=feats,
#         )

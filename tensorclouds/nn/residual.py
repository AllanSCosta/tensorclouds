from typing import Callable, Union

import e3nn_jax as e3nn
import jax
from flax import linen as nn

from ..tensorcloud import TensorCloud
from .layer_norm import EquivariantLayerNorm


class Residual(nn.Module):

    function: Callable[[TensorCloud], TensorCloud]
    norm: bool = True

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        assert state.irreps_array.ndim == 2
        assert state.mask.ndim == 1
        assert state.coord.ndim == 2

        new_state = self.function(state)

        seq_len = state.irreps_array.shape[0]
        new_seq_len = new_state.irreps_array.shape[0]

        if new_seq_len != seq_len:
            raise ValueError("Residual block cannot change sequence length")

        if state.irreps_array.irreps == new_state.irreps_array.irreps:
            features = state.irreps_array + new_state.irreps_array
        else:
            features = e3nn.flax.Linear(new_state.irreps_array.irreps)(
                e3nn.concatenate(
                    [
                        state.irreps_array,
                        new_state.irreps_array,
                    ]
                )
            )

        if self.norm:
            features = EquivariantLayerNorm()(features)

        return new_state.replace(irreps_array=features)

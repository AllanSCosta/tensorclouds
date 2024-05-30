import e3nn_jax as e3nn
import haiku as hk
import jax

from .layer_norm import EquivariantLayerNorm
from .residual import Residual
from .self_interaction import SelfInteraction
from .sequence_convolution import SequenceConvolution
from ..tensorcloud import TensorCloud 


class MixingBlock(hk.Module):
    def __init__(
        self,
        irreps_out: e3nn.Irreps,
        kernel_size: int = 3,
        residual: bool = True,
        weighted_coords: bool = False,
    ):
        super().__init__()
        self.irreps_out = irreps_out
        self.residual = residual
        self.kernel_size = kernel_size
        self.weighted_coords = weighted_coords

    def __call__(
        self,
        state: TensorCloud,
    ):
        @hk.remat
        def f(state: TensorCloud) -> TensorCloud:
            state = SequenceConvolution(
                self.irreps_out,
                stride=1,
                kernel_size=3,
                mode="same",
                norm=True,
                weighted=self.weighted_coords,
            )(state)
            state = SelfInteraction(
                [self.irreps_out],
                norm_last=True,
                residual=True,
            )(state)
            return state

        if self.residual:
            state = Residual(f)(state)
        else:
            state = f(state)

        state = state.replace(
            irreps_array=EquivariantLayerNorm()(state.irreps_array)
        )

        return state

from typing import List

import jax
from .mix import MixingBlock

import e3nn_jax as e3nn
from flax import linen as nn

from .sequence_convolution import SequenceConvolution
from .utils import multiscale_irreps
from .layer_norm import EquivariantLayerNorm
from ..tensorcloud import TensorCloud 

from .self_interaction import SelfInteraction


class Encoder(nn.Module):

    def __init__(
        self,
        irreps: e3nn.Irreps,
        layers: List[int],
        rescale: float,
        stride: int,
        kernel_size: int,
    ):
        super().__init__()
        self.irreps = e3nn.Irreps(irreps)
        self.tree_depth = len(layers)
        self.layers = layers
        self.rescale = rescale
        self.stride = stride
        self.kernel_size = kernel_size

    def __call__(
        self,
        state: TensorCloud,
    ) -> List[TensorCloud]:
        assert input is not None

        seq_len = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_len, state.irreps_array.irreps.dim)
        assert state.mask_irreps_array.shape == (seq_len,)
        assert state.coord.shape == (seq_len, 3)
        assert state.mask_coord.shape == (seq_len,)

        list_irreps = multiscale_irreps(
            self.irreps, self.tree_depth - 1, self.rescale, 0
        )
        assert len(list_irreps) == self.tree_depth
        acc_stride = 1
        skips = []

        # rc = jnp.sqrt(4.5 * self.knn * acc_stride)  # k=16 => rc=8.5, 12, 17, 24
        # state = SpatialConvolution(list_irreps[0], radial_cut=rc, k=self.knn)(state)
        # state = SpatialConvolution(list_irreps[0], radial_cut=rc, k=self.knn)(state)
        state = SelfInteraction([list_irreps[0]], norm_last=True, residual=True)(state)

        for idx, num_blocks in enumerate(self.layers):
            irreps_in = list_irreps[idx]
            residual = state

            # rc = jnp.sqrt(4.5 * self.knn * acc_stride)  # k=16 => rc=8.5, 12, 17, 24
            # state = SpatialConvolution(
            #     irreps_in, radial_cut=rc, k=self.knn, move=False
            # )(state)

            encoder_block = hk.without_apply_rng(
                hk.transform(
                    MixingBlock(
                        irreps_out=irreps_in,
                        kernel_size=self.kernel_size,
                        residual=True,
                        weighted_coords=True,
                    )
                )
            )

            init_block = jax.vmap(encoder_block.init, in_axes=(0, None))
            init_block = hk.lift(init_block, name=f"encoder_loop_{idx}")
            init_rng = hk.next_rng_keys(num_blocks) if hk.running_init() else None

            def apply_block(state, params):
                return encoder_block.apply(params, state), None

            state, _ = jax.lax.scan(apply_block, state, init_block(init_rng, state))

            state.replace(irreps_array=EquivariantLayerNorm()(residual.irreps_array + state.irreps_array))            
            skip = state.replace(irreps_array=state.irreps_array.filter('0e + 1e'))
            skips.append(skip)

            if idx < len(self.layers) - 1:
                irreps_out = list_irreps[idx + 1]
                prev_size = state.irreps_array.shape[0]
                state = SequenceConvolution(
                    irreps_out,
                    stride=self.stride,
                    kernel_size=self.kernel_size,
                    mode="valid",
                )(state)
                print(f"compressing [{prev_size}] {irreps_in} --> [{state.irreps_array.shape[0]}] {irreps_out}")
                acc_stride *= self.stride

        return skips

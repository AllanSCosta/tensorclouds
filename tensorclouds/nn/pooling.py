from flax import linen as nn
import jax.numpy as jnp

from model.base.sequence_convolution import convolution_indices
from model.base.utils import down_conv_seq_len, up_conv_seq_len
import e3nn_jax as e3nn
from ..tensorcloud import TensorCloud

from einops import repeat
import jax


class Upsample(nn.Module):
    def __init__(
        self,
        irreps_out: e3nn.Irreps,
        stride: int,
        kernel_size: int,
        *,
        mode: str = "valid",
    ):
        assert kernel_size % 2 == 1, "only odd sizes"
        super().__init__()
        self.irreps_out = irreps_out
        self.stride = stride
        self.kernel_size = kernel_size
        self.mode = mode

    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_len = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_len, state.irreps_array.irreps.dim)
        assert state.mask.shape == (seq_len,)
        assert state.coord.shape == (seq_len, 3)

        k = self.kernel_size
        irreps_array = state.irreps_array

        # set up and get convolution indices
        reverse_seq_len = up_conv_seq_len(seq_len, k, self.stride, self.mode)
        dst = convolution_indices(reverse_seq_len, k, self.stride, self.mode)
        dst = jnp.where(dst != -1, dst, reverse_seq_len)

        irreps_array_dst = jnp.where(
            state.mask_irreps_array[:, None], dst, reverse_seq_len
        )
        coord_dst = jnp.where(state.mask_coord[:, None], dst, reverse_seq_len)
        assert dst.shape == (seq_len, k)

        # compute num neighbors and new nmask
        def _num_neighbors(dst):
            num_neighbors = jnp.zeros((reverse_seq_len,))
            num_neighbors = num_neighbors.at[dst].add(1)
            new_mask = num_neighbors > 0
            num_neighbors = jnp.where(num_neighbors == 0.0, 1.0, num_neighbors)
            assert num_neighbors.shape == (reverse_seq_len,)
            return num_neighbors, new_mask

        coord_num_neigh, new_mask_coord = _num_neighbors(coord_dst)
        irreps_array_num_neigh, new_mask_irreps_array = _num_neighbors(irreps_array_dst)

        new_coord = e3nn.scatter_sum(
            repeat(state.coord, "i d -> i k d", k=k),
            dst=coord_dst,
            output_size=reverse_seq_len,
        ) / (coord_num_neigh[:, None] + 1e-6)
        assert new_coord.shape == (reverse_seq_len, 3)

        irreps_array = e3nn.flax.Linear(self.irreps_out)(state.irreps_array)
        output_windows = e3nn.IrrepsArray(
            self.irreps_out, repeat(irreps_array.array, "i d -> i k d", k=k)
        )

        # aggregate features over intersecting windows
        new_irreps_array = (
            e3nn.scatter_sum(
                output_windows, dst=irreps_array_dst, output_size=reverse_seq_len
            )
            / irreps_array_num_neigh[:, None]
        )
        assert new_irreps_array.shape == (reverse_seq_len, new_irreps_array.irreps.dim)

        return TensorCloud(
            irreps_array=new_irreps_array,
            mask_irreps_array=new_mask_irreps_array,
            coord=new_coord,
            mask_coord=new_mask_coord,
        )


class Downsample(nn.Module):
    def __init__(
        self,
        irreps_out: e3nn.Irreps,
        stride: int,
        kernel_size: int,
    ):
        assert kernel_size % 2 == 1, "only odd sizes"
        super().__init__()
        self.irreps_out = irreps_out
        self.stride = stride
        self.kernel_size = kernel_size
        self.mode = "valid"

    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_len = state.irreps_array.shape[0]

        assert state.irreps_array.shape == (seq_len, state.irreps_array.irreps.dim)
        assert state.mask.shape == (seq_len,)
        assert state.coord.shape == (seq_len, 3)

        k = self.kernel_size

        new_seq_len = down_conv_seq_len(seq_len, k, self.stride, self.mode)

        # compute source indices:
        src = convolution_indices(seq_len, k, self.stride, self.mode)
        assert src.shape == (new_seq_len, k)

        # create convolution masks:
        conv_mask_irreps_array = state.mask_irreps_array[src] & (src != -1)
        conv_mask_coord = state.mask_coord[src] & (src != -1)
        assert conv_mask_irreps_array.shape == (new_seq_len, k)
        assert conv_mask_coord.shape == (new_seq_len, k)

        # collect irreps_array and coordinates:
        conv_irreps_array = (
            conv_mask_irreps_array[:, :, None] * state.irreps_array[src, :]
        )
        assert conv_irreps_array.shape == (new_seq_len, k, conv_irreps_array.irreps.dim)
        conv_coord = conv_mask_coord[:, :, None] * state.coord[src, :]
        assert conv_coord.shape == (new_seq_len, k, 3)

        num_neighbors = jnp.sum(conv_mask_coord, axis=1)
        # have everyone present scores and use them as weights next coordinate:
        relative_weights = e3nn.flax.Linear("0e")(conv_irreps_array).array
        minus_inf = jnp.finfo(relative_weights.dtype).min
        relative_weights = jnp.where(
            conv_mask_coord[:, :, None], relative_weights, minus_inf
        )
        assert relative_weights.shape == (new_seq_len, k, 1)
        relative_weights = jax.nn.softmax(relative_weights, axis=1)
        new_coord = jnp.sum(conv_coord * relative_weights, axis=1)

        assert new_coord.shape == (new_seq_len, 3)

        # compute new mask coordinates:
        new_mask_coord = num_neighbors > 0

        new_irreps_array = e3nn.flax.Linear(self.irreps_out)(conv_irreps_array)
        new_irreps_array = e3nn.IrrepsArray(
            new_irreps_array.irreps, new_irreps_array.array.mean(-2)
        )
        assert new_irreps_array.shape == (new_seq_len, new_irreps_array.irreps.dim)

        # compute new mask for irreps_array:
        new_mask_irreps_array = jnp.sum(conv_mask_irreps_array, axis=-1) > 0
        assert new_mask_irreps_array.shape == (new_seq_len,)

        return TensorCloud(
            irreps_array=new_irreps_array,
            coord=new_coord,
            mask_irreps_array=new_mask_irreps_array,
            mask_coord=new_mask_coord,
        )

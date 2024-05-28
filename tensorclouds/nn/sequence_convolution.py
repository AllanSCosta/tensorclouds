import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from .layer_norm import EquivariantLayerNorm
from ..tensor_cloud import TensorCloud
from .utils import up_conv_seq_len, down_conv_seq_len


def moving_window(a, kernel: int, stride: int):
    """
    >>> moving_window([1, 2, 3, 4, 5], 3, 2)
    [[1, 2, 3], [3, 4, 5]]
    """
    starts = jnp.arange(0, len(a) - kernel + 1, stride)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (kernel,)))(starts)


def convolution_indices(
    sequence_length: int, kernel: int, stride: int, mode: str
) -> jnp.ndarray:
    indices = jnp.arange(0, sequence_length)

    if mode.lower() == "same":
        pad_width = kernel // 2
        indices = jnp.pad(
            indices, [(pad_width, pad_width)], constant_values=((-1, -1),)
        )
    elif mode.lower() == "valid":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return moving_window(indices, kernel, stride)


class SequenceConvolution(hk.Module):
    def __init__(
        self,
        irreps_out: e3nn.Irreps,
        stride: int,
        kernel_size: int,
        *,
        transpose: bool = False,
        weighted: bool = False,
        norm: bool = True,
        mode: str = "same",
    ):
        # NOTE(Allan): Maybe this assert is not required now
        # NOTE(Mario): I think it is, at least for the `same` mode
        assert kernel_size % 2 == 1, "only odd sizes"
        super().__init__()

        self.irreps_out = e3nn.Irreps(irreps_out)
        self.stride = stride
        self.kernel_size = kernel_size
        self.transpose = transpose
        self.norm = norm
        self.mode = mode
        self.weighted = weighted

    def forward(self, state: TensorCloud) -> TensorCloud:
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
        # compute new coordinates:
        # if self.weighted:
        #     # have everyone present scores and use them as weights next coordinate:
        #     relative_weights = e3nn.haiku.Linear("0e")(conv_irreps_array).array
        #     minus_inf = jnp.finfo(relative_weights.dtype).min
        #     relative_weights = jnp.where(
        #         conv_mask_coord[:, :, None], relative_weights, minus_inf
        #     )
        #     assert relative_weights.shape == (new_seq_len, k, 1)
        #     relative_weights = jax.nn.softmax(relative_weights, axis=1)
        #     new_coord = jnp.sum(conv_coord * relative_weights, axis=1)
        # else:
        #     relative_arrows = (
        #         e3nn.haiku.Linear(f"1x1e")(conv_irreps_array).array * 0.001
        #     )
        #     new_coord = (
        #         (relative_arrows + conv_coord) * conv_mask_coord[:, :, None]
        #     ).sum(1) / (num_neighbors[:, None] + 1e-6)
        new_coord = state.coord 
        assert new_coord.shape == (new_seq_len, 3)

        # compute new mask coordinates:
        new_mask_coord = num_neighbors > 0

        # present convolution internal coordinate differences to network:
        conv_vector = conv_coord[:, :, None, :] - conv_coord[:, None, :, :]
        conv_vector_mask = conv_mask_coord[:, :, None] & conv_mask_coord[:, None, :]
        assert conv_vector.shape == (new_seq_len, k, k, 3)
        assert conv_vector_mask.shape == (new_seq_len, k, k)

        conv_vector = conv_vector_mask[..., None] * conv_vector
        conv_vector = e3nn.spherical_harmonics("1e", conv_vector, True)
        assert conv_vector.shape == (new_seq_len, k, k, 3)

        conv_vector = conv_vector.axis_to_mul().axis_to_mul()
        assert conv_vector.shape == (new_seq_len, k**2 * 3)

        # mix irreps_array and vector to get new irreps_array:
        new_irreps_array = e3nn.haiku.Linear(self.irreps_out)(
            e3nn.concatenate([conv_irreps_array.axis_to_mul(), conv_vector]).regroup()
        )
        assert new_irreps_array.shape == (new_seq_len, new_irreps_array.irreps.dim)

        # compute new mask for irreps_array:
        new_mask_irreps_array = jnp.sum(conv_mask_irreps_array, axis=-1) > 0
        assert new_mask_irreps_array.shape == (new_seq_len,)

        if self.norm:
            new_irreps_array = EquivariantLayerNorm()(new_irreps_array)

        return TensorCloud(
            irreps_array=new_irreps_array,
            coord=new_coord,
            mask_irreps_array=new_mask_irreps_array,
            mask_coord=new_mask_coord,
        )

    def backward(self, state: TensorCloud) -> TensorCloud:
        seq_len = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_len, state.irreps_array.irreps.dim)
        assert state.mask.shape == (seq_len,)
        assert state.coord.shape == (seq_len, 3)

        k = self.kernel_size

        irreps_array = state.irreps_array
        if self.norm:
            irreps_array = EquivariantLayerNorm()(irreps_array)

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

        # predict global position for each new leaf
        relative_arrows = e3nn.haiku.Linear(f"{k}x1e")(irreps_array)
        relative_arrows = relative_arrows.mul_to_axis(k).array
        global_arrows = relative_arrows + state.coord[:, None, :]
        assert global_arrows.shape == (seq_len, k, 3)

        # aggregate predicted coordinates over intersecting windows
        new_coords = e3nn.scatter_sum(
            global_arrows, dst=coord_dst, output_size=reverse_seq_len
        ) / coord_num_neigh[:, None].astype(global_arrows.dtype)
        assert new_coords.shape == (reverse_seq_len, 3)

        # transpose-convolve the features
        output_windows = (
            e3nn.haiku.Linear(k * self.irreps_out)(irreps_array)
            .mul_to_axis(k)
            .remove_nones()
        )
        assert output_windows.shape == (seq_len, k, output_windows.irreps.dim)

        # aggregate features over intersecting windows
        features = e3nn.scatter_sum(
            output_windows, dst=irreps_array_dst, output_size=reverse_seq_len
        ) / irreps_array_num_neigh[:, None].astype(output_windows.dtype)
        assert features.shape == (reverse_seq_len, features.irreps.dim)

        return TensorCloud(
            irreps_array=features,
            mask_irreps_array=new_mask_irreps_array,
            coord=new_coords,
            mask_coord=new_mask_coord,
        )

    def __call__(self, state: TensorCloud) -> TensorCloud:
        if self.transpose:
            return self.backward(state)
        else:
            return self.forward(state)

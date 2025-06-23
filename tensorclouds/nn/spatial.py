from typing import Callable, List, Tuple

from einops import repeat

import e3nn_jax as e3nn
from flax import linen as nn
import jax
import jax.numpy as jnp

from ..tensorcloud import TensorCloud

from .utils import safe_normalize

class CompleteSpatialConvolution(nn.Module):

    irreps_out: e3nn.Irreps
    radial_cut: float
    radial_bins: int = 32
    radial_basis: str = "gaussian"
    edge_irreps: e3nn.Irreps = e3nn.Irreps("0e + 1e + 2e")
    norm: bool = True
    k_seq: int = 16
    attention: bool = False
    activation: Callable = jax.nn.silu
    move: bool = True

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_len = state.irreps_array.shape[0]
        assert state.mask.shape == (seq_len,)
        assert state.coord.shape == (seq_len, 3)
        features = state.irreps_array
        coord = state.coord

        if seq_len == 1:
            print("[WARNING] Skipping Spatial Convolution - seq_len == 1")
            return state

        features_i = e3nn.IrrepsArray(
            features.irreps, repeat(features.array, "i h -> i j h", j=seq_len)
        )
        features_j = e3nn.IrrepsArray(
            features.irreps, repeat(features.array, "j h -> i j h", i=seq_len)
        )
        coord_i = repeat(coord, "i d -> i j d", j=seq_len)
        coord_j = repeat(coord, "j d -> i j d", i=seq_len)

        mask_coord_i = repeat(state.mask_coord, "i -> i j", j=seq_len)
        mask_coord_j = repeat(state.mask_coord, "j -> i j", i=seq_len)
        cross_mask = mask_coord_i & mask_coord_j

        vectors = (coord_i - coord_j) * cross_mask[..., None]
        norm_sqr = jnp.sum(vectors**2, axis=-1)
        norm = jnp.where(
            norm_sqr == 0.0, 0.0, jnp.sqrt(jnp.where(norm_sqr == 0.0, 1.0, norm_sqr))
        )

        # Angular embedding:
        # edge_irreps = [ irrep for (mul, irrep) in self.edge_irreps ]
        # edge_irreps = e3nn.Irreps(edge_irreps)
        ang_embed = e3nn.spherical_harmonics(
            self.edge_irreps, vectors, False, "component"
        )
        ang_embed = ang_embed * cross_mask[..., None].astype(ang_embed.array.dtype)

        # messages_i = e3nn.flax.Linear(self.irreps_out)(
        #     e3nn.tensor_product(ang_embed, features_i)
        # )

        messages_j = e3nn.flax.Linear(self.irreps_out)(
            e3nn.tensor_product(ang_embed, features_j)
        )

        messages = e3nn.concatenate(
            [
                # messages_i,
                messages_j,
                ang_embed,
            ],
            axis=-1,
        ).regroup()

        # Radial part:
        rad_embed = (
            e3nn.soft_one_hot_linspace(
                norm,
                start=0.0,
                end=self.radial_cut,
                number=self.radial_bins,
                basis=self.radial_basis,
                cutoff=True,
            )
            * cross_mask[..., None]
        )

        seq_pos_i = repeat(jnp.arange(seq_len), "i -> i j", j=seq_len)
        seq_pos_j = repeat(jnp.arange(seq_len), "j -> i j", i=seq_len)

        relative_seq_pos = seq_pos_i - seq_pos_j
        k_seq = self.k_seq

        relative_seq_pos = jnp.where(
            jnp.abs(relative_seq_pos) <= k_seq, relative_seq_pos, 0
        )
        relative_seq_pos = jnp.where(cross_mask, relative_seq_pos, 0)

        relative_seq_pos = relative_seq_pos + k_seq
        relative_seq_pos = nn.Embed(num_embeddings=2 * k_seq + 1, features=32)(
            relative_seq_pos
        )

        rad_embed = e3nn.concatenate(
            [relative_seq_pos, rad_embed, messages.filter("0e")], axis=-1
        ).regroup()
        rad_embed = e3nn.flax.MultiLayerPerceptron(
            [self.radial_bins, messages.irreps.num_irreps],
            self.activation,
            with_bias=True,
            output_activation=False,
        )(rad_embed)

        messages = (
            messages * rad_embed * cross_mask[..., None].astype(messages.array.dtype)
        )

        features_aggr = e3nn.sum(messages, axis=1) / (
            jnp.sum(cross_mask, axis=1, keepdims=True) + 1e-6
        )
        features_aggr = features_aggr * (jnp.sum(cross_mask, axis=1, keepdims=True) > 1)
        features = e3nn.flax.Linear(self.irreps_out)(features_aggr)

        if self.move:
            update = 1e-3 * e3nn.flax.Linear("1e")(features).array
            new_coord = state.coord + update
            state = state.replace(coord=new_coord)

        return state.replace(irreps_array=features)



def knn(coord: jax.Array, mask: jax.Array, k: int, k_seq: int = None):
    n, d = coord.shape
    distance_matrix = jnp.sum(
        jnp.square(coord[:, None, :] - coord[None, :, :]), axis=-1
    )
    assert distance_matrix.shape == (n, n)
    matrix_mask = mask[:, None] & mask[None, :]
    assert matrix_mask.shape == (n, n)

    # if k sequence nearest neighbors is on
    # go up and down identity matrix and force everyone to be listed
    if k_seq != None:
        assert k_seq % 2 == 0
        seq_nei_matrix = jnp.zeros((n, n))
        eye = jnp.eye(n)
        for i in range(1, k_seq // 2 + 1):
            up = jnp.roll(eye, i, axis=0)
            down = jnp.roll(eye, -i, axis=0)
            up = up - jnp.triu(up, k=0)
            down = down - jnp.tril(down, k=0)
            seq_nei_matrix += up + down
        seq_nei_matrix = jnp.where(matrix_mask, seq_nei_matrix, 0)
        distance_matrix = jnp.where(seq_nei_matrix, -jnp.inf, distance_matrix)

    distance_matrix = jnp.where(matrix_mask, distance_matrix, jnp.inf)
    neg_dist, neighbors = jax.lax.top_k(-distance_matrix, k)
    mask = neg_dist != -jnp.inf

    assert neighbors.shape == (n, k)
    assert mask.shape == (n, k)

    return neighbors, mask


from e3nn_jax.experimental.linear_shtp import LinearSHTP

import einops as ein


class kNNSpatialConvolution(nn.Module):

    irreps_out: e3nn.Irreps
    k_seq: int = 4
    k: int = 16
    radial_cut: float = 20.0
    radial_bins: int = 32
    radial_basis: str = "gaussian"
    edge_irreps: e3nn.Irreps = e3nn.Irreps("0e + 1e + 2e")
    norm: bool = True
    activation: Callable = jax.nn.silu
    envelope: bool = False
    move: bool = False

    @nn.compact
    def __call__(self, state: TensorCloud) -> TensorCloud:
        seq_len = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_len, state.irreps_array.irreps.dim)
        assert state.mask.shape == (seq_len,)
        assert state.coord.shape == (seq_len, 3)
        irreps_in = state.irreps_array.irreps

        features = state.irreps_array
        if seq_len == 1:
            print("[WARNING] Skipping Spatial Convolution - seq_len == 1")
            return state

        # SCATTER INPUTS
        k = self.k
        k = min(k + 1, seq_len)

        nei_indices, nei_mask = knn(
            state.coord,
            state.mask_coord,
            k=k,
            k_seq=self.k_seq,
        )

        features_i = e3nn.IrrepsArray(
            features.irreps, repeat(features.array, "i h -> i k h", k=k)
        )

        features_j = nei_mask[:, :, None] * features[nei_indices]

        coord = state.coord
        coord_i = coord[:, None, :]
        coord_j = coord[nei_indices, :]

        mask_coord_i = state.mask_coord[:, None]
        mask_coord_j = state.mask_coord[nei_indices]
        cross_mask = mask_coord_i & mask_coord_j

        # MAKE VECTORS
        vectors = (coord_i - coord_j) * nei_mask[..., None]
        norm_sqr = jnp.sum(vectors**2, axis=-1)
        norm = jnp.where(
            norm_sqr == 0.0, 0.0, jnp.sqrt(jnp.where(norm_sqr == 0.0, 1.0, norm_sqr))
        )

        edge_irreps = e3nn.Irreps([mulir.ir for mulir in features_j.irreps])
        # edge_irreps = features_j.irreps
        # edge_irreps = e3nn.Irreps('1x1e')

        # ANGULAR EMBED
        ang_embed = e3nn.spherical_harmonics(edge_irreps, vectors, True, "component")
        ang_embed = ang_embed * nei_mask[..., None]

        # RADIAL EMBED
        rad_embed = (
            e3nn.soft_one_hot_linspace(
                norm,
                start=0.0,
                end=self.radial_cut,
                number=self.radial_bins,
                basis=self.radial_basis,
                cutoff=True,
            )
            * nei_mask[..., None]
        )

        # RELATIVE POS ENCODING
        seq_pos_i = jnp.arange(seq_len)[:, None]
        seq_pos_j = jnp.arange(seq_len)[nei_indices]

        relative_seq_pos = seq_pos_i - seq_pos_j
        k_seq = self.k_seq

        relative_seq_pos = jnp.where(
            jnp.abs(relative_seq_pos) <= k_seq, relative_seq_pos, 0
        )
        relative_seq_pos = jnp.where(cross_mask, relative_seq_pos, 0)

        relative_seq_pos = relative_seq_pos + k_seq
        relative_seq_pos = nn.Embed(num_embeddings=2 * k_seq + 1, features=32)(
            relative_seq_pos
        )
        # MAKE MESSAGES

        # STANDARD TFN
        # messages = e3nn.flax.Linear(self.irreps_out)(
        #     e3nn.concatenate(
        #         [e3nn.tensor_product(ang_embed, features_j), ang_embed], axis=-1
        #     ).regroup()
        # )

        # eSCN
        conv = LinearSHTP(self.irreps_out)
        vectors = e3nn.IrrepsArray("1e", vectors)
        messages = jax.vmap(jax.vmap(conv))(features_j, vectors)

        # SIMPLE CAT
        # messages = e3nn.flax.Linear(self.irreps_out)(
        #     e3nn.concatenate(
        #         [features_j, ang_embed], axis=-1
        #     ).regroup()
        # )

        # L-WISE
        # tps = []
        # for (mulir) in features_j.irreps:
        #     ir = mulir.ir
        #     tps.append(
        #         e3nn.tensor_product(features_j.filter(keep=ir), ang_embed.filter(keep=ir))
        #     )
        # messages = e3nn.concatenate(tps + [ang_embed], axis=-1).regroup()

        # ELEMENT-WISE
        # messages = e3nn.flax.Linear(self.irreps_out)(
        # e3nn.concatenate(
        # [e3nn.elementwise_tensor_product(features_j, ang_embed)], axis=-1
        # ).regroup()
        # )

        # GATE MESSAGES
        gate = e3nn.concatenate(
            [
                relative_seq_pos,
                rad_embed,
                messages.filter("0e"),
            ],
            axis=-1,
        ).regroup()

        gate = e3nn.flax.MultiLayerPerceptron(
            [messages.irreps.num_irreps],
            act=self.activation,
            with_bias=True,
            output_activation=True,
        )(gate)

        # AGGREGATE
        messages = messages * gate * cross_mask[..., None].astype(messages.array.dtype)

        new_features = e3nn.sum(messages, axis=1) / (
            jnp.sum(nei_mask, axis=1, keepdims=True) + 1e-6
        )
        new_features = new_features * (jnp.sum(cross_mask, axis=1, keepdims=True) > 1)

        if self.move:
            update = e3nn.flax.Linear("1e")(new_features).array
            state = state.replace(coord=state.coord + update)

        return state.replace(irreps_array=new_features)



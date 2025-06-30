import re
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import optax
from einops import rearrange, repeat
from moleculib.assembly.datum import AssemblyDatum
from moleculib.protein.datum import ProteinDatum

from tensorclouds.nn.utils import ModelOutput, safe_norm, safe_normalize

# from tensorclouds.train.schedulers import Scheduler

CA_INDEX = 1


def vector_cloud_matching_loss(
    self,
    rng_key,
    model_output: ModelOutput,
    _: ProteinDatum,
) -> Tuple[ModelOutput, jax.Array, Dict[str, float]]:
    pred, target = model_output.prediction, model_output.target

    vec_irreps = "1e"
    pred_vectors = rearrange(
        pred.irreps_array.filter(vec_irreps).array, "... (v e) -> ... v e", e=3
    )
    target_vectors = rearrange(
        target.irreps_array.filter(vec_irreps).array, "... (v e) -> ... v e", e=3
    )
    vec_mask = target_vectors.sum(-1) != 0.0

    pred_vectors = pred_vectors.at[..., ca_index, :].set(0.0) + pred.coord[:, None, :]
    target_vectors = (
        target_vectors.at[..., ca_index, :].set(0.0) + target.coord[:, None, :]
    )

    pred_vectors = rearrange(pred_vectors, "r v ... -> (r v) ...")
    target_vectors = rearrange(target_vectors, "r v ... -> (r v) ...")
    vec_mask = rearrange(vec_mask, "r v -> (r v)")

    def ij_map(x, distance=True):
        x_ij = rearrange(x, "... i c -> ... i () c") - rearrange(
            x, "... j c -> ... () j c"
        )
        return safe_norm(x_ij)[..., None] if distance else x_ij

    pred_dist_map = ij_map(pred_vectors)
    target_dist_map = ij_map(target_vectors)

    cross_mask = rearrange(vec_mask, "i -> i ()") & rearrange(vec_mask, "j -> () j")
    vectors_loss = jnp.square(pred_dist_map - target_dist_map)

    vectors_loss = jnp.sum(vectors_loss * cross_mask[..., None]) / (
        jnp.sum(cross_mask) + 1e-6
    )

    pred_coord_dist_map = ij_map(pred.coord)
    target_coord_dist_map = ij_map(target.coord)

    cross_mask = rearrange(target.mask_coord, "r -> r ()") & rearrange(
        target.mask_coord, "r -> () r"
    )
    coord_loss = jnp.square(pred_coord_dist_map - target_coord_dist_map)

    coord_loss = jnp.sum(coord_loss * cross_mask[..., None]) / (
        jnp.sum(cross_mask) + 1e-6
    )

    return (
        model_output,
        vectors_loss + coord_loss,
        {"vectors_loss": vectors_loss, "coord_loss": coord_loss},
    )


class InternalVectorLoss:
    def __init__(self, weight=1.0, start_step=0, norm_only=False):
        super().__init__(weight=weight, start_step=start_step)
        self.norm_only = norm_only

    def _call(
        self, rng_key, model_output: ModelOutput, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jax.Array, Dict[str, float]]:
        def flip_atoms(coord, flips, mask):
            flips_list = jnp.where(mask[..., None], flips, 15)
            p, q = flips_list.T
            p_coords, q_coords = coord[p], coord[q]
            coord = coord.at[p].set(q_coords, mode="drop")
            coord = coord.at[q].set(p_coords, mode="drop")
            return coord

        def _internal_vector_loss(coords, ground_coords, mask):
            cross_mask = rearrange(mask, "... i -> ... i ()") & rearrange(
                mask, "... j -> ... () j"
            )
            cross_mask = cross_mask.astype(coords.dtype)

            cross_vectors = rearrange(coords, "... i c -> ... i () c") - rearrange(
                coords, "... j c -> ... () j c"
            )
            cross_ground_vectors = rearrange(
                ground_coords, "... i c -> ... i () c"
            ) - rearrange(ground_coords, "... j c -> ... () j c")
            if self.norm_only:
                cross_vectors = safe_norm(cross_vectors)[..., None]
                cross_ground_vectors = safe_norm(cross_ground_vectors)[..., None]

            error = optax.huber_loss(
                cross_vectors, cross_ground_vectors, delta=1.0
            ).mean(-1)
            error = (error * cross_mask).sum((-1, -2)) / (
                cross_mask.sum((-1, -2)) + 1e-6
            )
            error = error * (cross_mask.sum() > 0).astype(error.dtype)
            return error

        coords = model_output.datum.atom_coord
        alternative = flip_atoms(coords, ground.flips_list, ground.flips_mask)
        loss = _internal_vector_loss(coords, ground.atom_coord, ground.atom_mask)
        alternate_loss = _internal_vector_loss(
            alternative, ground.atom_coord, ground.atom_mask
        )

        coords = jnp.where(
            (loss < alternate_loss)[..., None, None], coords, alternative
        )

        loss = jnp.where(loss < alternate_loss, loss, alternate_loss)

        new_datum = dict(vars(model_output.datum).items())
        new_datum.update(atom_coord=coords)
        datum = ProteinDatum(**new_datum)

        model_output = model_output.replace(datum=datum)

        return model_output, loss.mean(), {"internal_vector_loss": loss.mean()}

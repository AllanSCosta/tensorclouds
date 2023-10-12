from collections import defaultdict
from typing import List, NamedTuple, Any
import e3nn_jax as e3nn
from model.base.utils import ModelOutput, inner_split, inner_stack, multiscale_irreps

import jax.numpy as jnp
import jax
import functools

import haiku as hk
from einops import rearrange



class TrackerState(NamedTuple):
    sample_pool: Any
    codebook_frequencies: Any


class Tracker:
    def init(self, params):
        codebooks = hk.data_structures.filter(
            lambda _, name, __: name == "codebook", params
        )
        pool = jax.tree_util.tree_map(
            lambda w: jnp.zeros_like(w, dtype=w.dtype), codebooks
        )
        frequencies = jax.tree_util.tree_map(
            lambda w: jnp.zeros_like(w[..., 0], dtype=w.dtype), codebooks
        )
        return TrackerState(sample_pool=pool, codebook_frequencies=frequencies)

    def reset(self, state):
        return TrackerState(
            sample_pool=state.sample_pool,
            codebook_frequencies=jax.tree_util.tree_map(
                lambda w: jnp.zeros_like(w, dtype=w.dtype), state.codebook_frequencies
            ),
        )

    def maybe_update_codebook(self, params, tracker_state):
        codebooks = hk.data_structures.filter(
            lambda _, name, __: name == "codebook", params
        )
        updated_codebooks = jax.tree_util.tree_map(
            lambda w, f, p: jnp.where((f > 0)[..., None], w, p),
            codebooks,
            tracker_state.codebook_frequencies,
            tracker_state.sample_pool,
        )
        updated_params = hk.data_structures.merge(params, updated_codebooks)
        return updated_params

    def update(self, rng_key, tracker_state, model_output: ModelOutput):
        frequencies, fre_keys = jax.tree_util.tree_flatten(
            tracker_state.codebook_frequencies
        )
        frequencies = [[frequencies[-1]]] + [inner_split(f) for f in frequencies[:-1]]

        pool, pool_keys = jax.tree_util.tree_flatten(tracker_state.sample_pool)
        pool = [[pool[-1]]] + [inner_split(p) for p in pool[:-1]]

        metrics = {}

        for level, level_quants in enumerate(model_output.decoder_internals):
            for level_layer, quants in enumerate(level_quants):
                _, posterior = quants

                mask = posterior.state.mask_irreps_array
                mask = rearrange(mask, "... -> (...)")

                counts = jax.nn.one_hot(
                    jnp.argmax(posterior.logits, axis=-1), posterior.logits.shape[-1]
                )
                counts = rearrange(counts, "... c -> (...) c")

                counts = (counts * mask[..., None]).sum(-2) / (
                    jnp.sum(mask, axis=0) + 1e-6
                )
                counts = counts * (jnp.sum(mask) > 0).astype(jnp.float32)

                frequencies[level][level_layer] += counts
                metrics[f"level_{level}/layer_{level_layer}/current_usage"] = jnp.sum(
                    counts > 0
                )
                metrics[
                    f"level_{level}/layer_{level_layer}/accumulated_usage"
                ] = jnp.sum(frequencies[level][level_layer] > 0)

                pre_samples = posterior.pre_quant
                batch_size = pre_samples.shape[0]

                pre_samples = rearrange(pre_samples.array, "... d -> (...) d")

                local_pool = pool[level][level_layer]

                # random shuffle
                k1, k2, rng_key = jax.random.split(rng_key, 3)
                pre_samples_idx = jax.random.permutation(
                    k1, jnp.arange(pre_samples.shape[0])
                )
                pre_samples = pre_samples[pre_samples_idx][:batch_size]
                pre_samples_mask = mask[pre_samples_idx][:batch_size]

                # get new potential indices
                candidate_indices = jax.random.permutation(
                    k2, jnp.arange(local_pool.shape[0])
                )[:batch_size]
                local_pool = local_pool.at[candidate_indices].set(
                    jnp.where(
                        pre_samples_mask[..., None],
                        pre_samples,
                        local_pool[candidate_indices],
                    )
                )
                pool[level][level_layer] = local_pool

        frequencies = [inner_stack(f) for f in frequencies[1:]] + [frequencies[0][0]]
        frequencies = jax.tree_util.tree_unflatten(fre_keys, frequencies)

        pool = [inner_stack(p) for p in pool[1:]] + [pool[0][0]]
        pool = jax.tree_util.tree_unflatten(pool_keys, pool)

        return (
            TrackerState(
                sample_pool=tracker_state.sample_pool, codebook_frequencies=frequencies
            ),
            metrics,
        )

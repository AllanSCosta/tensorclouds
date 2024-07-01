from typing import List, Tuple, Union

import numpy as np

import chex
import e3nn_jax as e3nn
import jax.numpy as jnp

from moleculib.protein.datum import ProteinDatum

import jax
from jax.tree_util import tree_map, tree_flatten


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def l2_norm(tree):
    """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = tree_flatten(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def clip_grads(grad_tree, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
    norm = l2_norm(grad_tree)
    normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
    return tree_map(normalize, grad_tree)


def inner_stack(pytrees):
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values, axis=0), *pytrees)


def inner_split(pytree):
    leaves, defs = tree_flatten(pytree)
    splits = [
        [arr.squeeze(0) for arr in jnp.split(leaf, len(leaf), axis=0)]
        for leaf in leaves
    ]
    splits = list(zip(*splits))
    return [jax.tree_util.tree_unflatten(defs, split) for split in splits]


from ..tensorcloud import TensorCloud


@chex.dataclass
class ModelOutput:
    datum: ProteinDatum
    encoder_internals: List[TensorCloud]
    decoder_internals: List[TensorCloud]
    atom_perm_loss: jnp.ndarray
    diff_loss: List[jnp.ndarray]


def rescale_irreps(irreps: e3nn.Irreps, rescale: float, chunk_factor: int = 0):
    irreps = e3nn.Irreps([(int(mul * rescale), ir) for mul, ir in irreps])
    if chunk_factor != 0:
        irreps = e3nn.Irreps(
            [(next_multiple(mul, chunk_factor), ir) for mul, ir in irreps]
        )
    return irreps


def multiscale_irreps(
    irreps: e3nn.Irreps, depth: int, rescale: float, chunk_factor: int = 0
) -> List[e3nn.Irreps]:
    list_irreps = [irreps]
    for _ in range(depth):
        list_irreps.append(rescale_irreps(list_irreps[-1], rescale, chunk_factor))
    return list_irreps


def next_multiple(x: int, factor: int) -> int:
    """next multiple of factor"""
    if x % factor == 0:
        return x
    return x + factor - (x % factor)


def up_conv_seq_len(size: int, kernel: int, stride: int, mode: str) -> int:
    """output size of a convolutional layer"""
    if mode.lower() == "same":
        return stride * (size - 1) + 1
    if mode.lower() == "valid":
        return stride * (size - 1) + kernel

    raise ValueError(f"Unknown mode: {mode}")


def down_conv_seq_len(size: int, kernel: int, stride: int, mode: str) -> int:
    """output size of a convolutional layer"""
    if mode.lower() == "same":
        assert kernel % 2 == 1
        if (size - 1) % stride != 0:
            raise ValueError(
                (
                    f"Not a perfect convolution: "
                    f"size={size}, kernel={kernel}, stride={stride} mode={mode}."
                    f"({size} - 1) % {stride} == {(size - 1) % stride} != 0"
                )
            )
        return (size - 1) // stride + 1
    if mode.lower() == "valid":
        if (size - kernel) % stride != 0:
            raise ValueError(
                (
                    f"Not a perfect convolution: "
                    f"size={size}, kernel={kernel}, stride={stride} mode={mode}."
                    f"({size} - {kernel}) % {stride} == {(size - kernel) % stride} != 0"
                )
            )
        return (size - kernel) // stride + 1

    raise ValueError(f"Unknown mode: {mode}")


def safe_norm(vector: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """safe_norm(x) = norm(x) if norm(x) != 0 else 1.0"""
    norms_sqr = jnp.sum(vector**2, axis=axis)
    norms = jnp.where(norms_sqr == 0.0, 1.0, norms_sqr) ** 0.5
    return norms


def safe_normalize(vector: jnp.ndarray) -> jnp.ndarray:
    return vector / safe_norm(vector)[..., None]


import os
import pickle

from moleculib.assembly.datum import AssemblyDatum
from moleculib.abstract.dataset import PreProcessedDataset


class EmbeddingsDataset(PreProcessedDataset):

    def __init__(self, path, transform=[]):
        self.path = path
        with open(os.path.join(path, "embeddings.pyd"), "rb") as f:
            splits = pickle.load(f)
        super().__init__(splits, transform=transform)

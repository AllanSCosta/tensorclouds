from functools import partial
from typing import Any, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import e3nn_jax as e3nn
from ..nn.self_interaction import SelfInteraction
from ..nn.utils import dotdict, safe_norm
from ..tensorcloud import TensorCloud

from einops import rearrange, repeat
from moleculib.nucleic.alphabet import (
    all_nucs, #all_residues,
    all_nucs_atom_mask, # all_residues_atom_mask,
    all_nucs_atom_tokens, # all_residues_atom_tokens,
    # flippable_arr,
    # flippable_mask,
)

from jax import tree_util
from moleculib.nucleic.datum import NucleicDatum



def nuc_to_tensor_cloud(nuc):
    res_token = nuc["nuc_token"]
    res_mask = nuc["atom_mask"][..., 8] ##TODO: Do we want it to be atom mask or res mask??...

    scalars = jax.nn.one_hot(res_token, 14)
    scalars = scalars * res_mask[..., None]

    mask = nuc["atom_mask"]
    #the ca coord must be present for the atoms in the residue to be available
    #so if the c5' coord is masked, the entire nucleotide is masked:
    mask = mask & res_mask[:, None] 
    
    vectors = nuc["atom_coord"]
    ca_coord = vectors[..., 8, :] #taking C5' atom as center
    vectors = vectors - ca_coord[..., None, :]
    vectors = vectors * mask[..., None]
    vectors = rearrange(vectors, "r a c -> r (a c)")

    irreps_array = e3nn.IrrepsArray(
        "14x0e + 24x1e", jnp.concatenate([scalars, vectors], axis=-1)
    )


    state = TensorCloud(
        irreps_array=irreps_array, #TODO:  check if we want irreps_array * ca_mask[..., None],
        mask_irreps_array=mask,
        coord=ca_coord,  #TODO:  check if we want ca_coord * ca_mask[..., None],
        mask_coord=res_mask,
    ).centralize()

    return state


def tensor_cloud_to_nuc(state, nuc=None, backbone_only=False):
    irreps_array = state.irreps_array
    ca_coord = state.coord
    res_mask = state.mask_coord

# Depending on the irreps configuration, different processing paths are taken:
    if str(irreps_array.irreps) == "14x0e+24x1e":
        #Directly filter out scalar features (0e irreps) and converting the scalars into prob distribution
        res_logits = jax.nn.softmax(irreps_array.filter("0e").array)
        eos_logits = None
    elif str(irreps_array.irreps) == "24x1e":# no scalar in this config
        res_logits = None
        eos_logits = None
    # else:  
    #     # TODO: When do we use it? cuz like the logits are not taken anywhere
    #     # Filter to keep scalar irreps and apply a MultiLayerPerceptron to obtain logits
    #     invariants = irreps_array.filter(keep="0e")
    #     logits = e3nn.haiku.MultiLayerPerceptron(
    #         [invariants.irreps.dim, invariants.irreps.num_irreps, 25],
    #         act=jax.nn.silu,
    #         output_activation=False,
    #     )(invariants).array

        # res_logits, sos_logits, eos_logits = (
        #     logits[..., :14],
        #     logits[..., -1], #TODO: why -1 and -2?
        #     logits[..., -2],
        # )
        # irreps_array = e3nn.haiku.Linear("24x1e")(irreps_array)

    atom_coord = irreps_array.filter("1e").array
    atom_coord = rearrange(atom_coord, "r (a c) -> r a c", a=24)

    if nuc is None and not backbone_only:
        sequence_token = jnp.argmax(res_logits, axis=-1)
    elif nuc is None and backbone_only:
        sequence_token = jnp.full(res_logits.shape[0], all_nucs.index("G")) #NOTE changed to G but i think its completely irrelevant to mine
    else:
        sequence_token = nuc["nuc_token"]

    logit_extract = repeat(sequence_token, "r -> r l", l=14) == repeat(
        jnp.arange(0, 14), "l -> () l"
    )

    if nuc is None:
        atom_token = (logit_extract[..., None] * all_nucs_atom_tokens[None]).sum(-2)
        atom_mask = (logit_extract[..., None] * all_nucs_atom_mask[None]).sum(-2)
    else:
        atom_token = nuc["atom_token"]
        atom_mask = nuc["atom_mask"]

    atom_coord = atom_coord.at[..., 8, :].set(0.0)
    atom_coord = atom_coord + ca_coord[..., None, :]
    atom_coord = atom_coord * atom_mask[..., None]

    return dict(
        idcode=None,
        resolution=None,
        sequence=None,
        nuc_token=sequence_token,
        nuc_index=jnp.arange(sequence_token.shape[0]),
        nuc_mask=res_mask,
        chain_token=jnp.zeros(sequence_token.shape[0], dtype=jnp.int32),
        atom_token=atom_token,
        atom_coord=atom_coord,
        atom_mask=atom_mask,
        residue_logits=res_logits,
        eos_logits=eos_logits,
    )
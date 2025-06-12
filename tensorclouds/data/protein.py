from functools import partial

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
from ..tensorcloud import TensorCloud

from einops import rearrange, repeat
from moleculib.protein.alphabet import (
    all_residues,
    all_residues_atom_mask,
    all_residues_atom_tokens,
    flippable_arr,
    flippable_mask,
)


from moleculib.protein.datum import ProteinDatum

def protein_to_tensor_cloud(protein):
    res_token = protein.residue_token
    res_mask = protein.atom_mask[..., 1]
    vectors = protein.atom_coord
    mask = protein.atom_mask

    ca_coord = vectors[..., 1, :]

    vectors = vectors - ca_coord[..., None, :]
    vectors = vectors * mask[..., None]
    vectors = rearrange(vectors, "r a c -> r (a c)")

    irreps_array = e3nn.IrrepsArray("14x1e", jnp.array(vectors))

    state = TensorCloud(
        irreps_array=irreps_array,
        mask_irreps_array=jnp.array(mask),
        coord=jnp.array(ca_coord),
        mask_coord=jnp.array(res_mask),
        label=jnp.array(res_token * res_mask),
    )

    return state


def tensor_cloud_to_protein(state, backbone_only=False):
    irreps_array = state.irreps_array
    ca_coord = state.coord
    res_mask = state.mask_coord

    atom_coord = irreps_array.filter("1e").array
    atom_coord = rearrange(atom_coord, "r (a c) -> r a c", a=14)

    labels = state.label
    logit_extract = repeat(labels, "r -> r l", l=23) == repeat(
        jnp.arange(0, 23), "l -> () l"
    )

    atom_token = (logit_extract[..., None] * all_residues_atom_tokens[None]).sum(-2)
    atom_mask = (logit_extract[..., None] * all_residues_atom_mask[None]).sum(-2)

    atom_coord = atom_coord.at[..., 1, :].set(0.0)
    atom_coord = atom_coord + ca_coord[..., None, :]
    atom_coord = atom_coord * atom_mask[..., None]

    return ProteinDatum(
        idcode=None,
        resolution=None,
        sequence=None,
        residue_token=labels,
        residue_index=jnp.arange(labels.shape[0]),
        residue_mask=res_mask,
        chain_token=jnp.zeros(labels.shape[0], dtype=jnp.int32),
        atom_token=atom_token,
        atom_coord=atom_coord,
        atom_mask=atom_mask,
    )

import einops as ein

def ligand_to_point_cloud(ligand):
    atom_type = jnp.array(ligand.atom_type)
    atom_coord = ligand.atom_coord
    atom_mask = ligand.atom_mask

    irreps_array = e3nn.zeros('14x1e', (atom_type.shape[0], ))
    mask_irreps_array = jnp.zeros((atom_type.shape[0], irreps_array.irreps.num_irreps), dtype=bool)
    mask_irreps_array = mask_irreps_array.at[:, 1].set(True)
    
    state = TensorCloud(
        irreps_array=irreps_array,
        mask_irreps_array=mask_irreps_array,
        coord=jnp.array(atom_coord),
        mask_coord=jnp.array(atom_mask),
        label=jnp.array(atom_type),
    )

    return state

from moleculib.assembly.datum import AssemblyDatum
from moleculib.molecule.datum import MoleculeDatum

def tensor_cloud_to_ligand(state):
    atom_coord = state.coord
    atom_mask = state.mask_coord
    atom_type = state.label

    atom_coord = atom_coord * atom_mask[..., None]

    return MoleculeDatum(
        idcode=None,
        atom_type=atom_type,
        atom_coord=atom_coord,
        atom_mask=atom_mask,
    )

from tensorclouds.tensorcloud import TensorCloud, TensorClouds

def assembly_to_tensor_cloud(assembly):
    protein_clouds = []
    for protein in assembly.proteins:
        prot_cloud = protein_to_tensor_cloud(protein)
        protein_clouds.append(prot_cloud)
    
    ligand_cloud = ligand_to_point_cloud(assembly.ligands[0])
    assembly_cloud = TensorClouds.create(
        proteins=TensorCloud.concatenate(protein_clouds),
        ligands=ligand_cloud
    )

    return assembly_cloud


def tensor_cloud_to_assembly(state):
    protein_clouds = state._tensorclouds['proteins']
    ligand_cloud = state._tensorclouds['ligands']

    protein = tensor_cloud_to_protein(protein_clouds)
    ligand = tensor_cloud_to_ligand(ligand_cloud)

    return AssemblyDatum(
        idcode=None,
        proteins=protein,
        ligands=ligand
    )
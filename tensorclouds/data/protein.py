from functools import partial

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from moleculib.protein.alphabet import (all_residues, all_residues_atom_mask,
                                        all_residues_atom_tokens,
                                        flippable_arr, flippable_mask)
from moleculib.protein.datum import ProteinDatum

from ..tensorcloud import TensorCloud




import einops as ein


def ligand_to_point_cloud(ligand):
    atom_type = jnp.array(ligand.atom_type)
    atom_coord = ligand.atom_coord
    atom_mask = ligand.atom_mask

    irreps_array = e3nn.zeros("14x1e", (atom_type.shape[0],))
    mask_irreps_array = jnp.zeros(
        (atom_type.shape[0], irreps_array.irreps.num_irreps), dtype=bool
    )
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
        proteins=TensorCloud.concatenate(protein_clouds), ligands=ligand_cloud
    )

    return assembly_cloud


def tensor_cloud_to_assembly(state):
    protein_clouds = state._tensorclouds["proteins"]
    ligand_cloud = state._tensorclouds["ligands"]

    protein = tensor_cloud_to_protein(protein_clouds)
    ligand = tensor_cloud_to_ligand(ligand_cloud)

    return AssemblyDatum(idcode=None, proteins=protein, ligands=ligand)

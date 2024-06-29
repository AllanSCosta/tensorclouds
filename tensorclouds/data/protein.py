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
from moleculib.protein.alphabet import (
    all_residues,
    all_residues_atom_mask,
    all_residues_atom_tokens,
    flippable_arr,
    flippable_mask,
)

from jax import tree_util
from moleculib.protein.datum import ProteinDatum


class Embed(hk.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.w_init = hk.initializers.TruncatedNormal()

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        params = hk.get_parameter(
            "embeddings",
            [self.vocab_size, self.embed_dim],
            init=self.w_init,
            dtype=jnp.bfloat16,
        )
        tokens = jnp.asarray(tokens)
        return jnp.asarray(params)[(tokens,)]


class ProteinDatumEncoder(hk.Module):
    def __init__(self, irreps: e3nn.Irreps, interact=True, depth: int = 0,rescale=2.0):
        super().__init__()
        self.depth = depth
        if not str(irreps.filter('2e')):
            dim = irreps[0].mul
            irreps = irreps + e3nn.Irreps(f'{dim}x2e')
        self.irreps = irreps
        self.interact = interact
        self.rescale = rescale

    def __call__(self, datum: ProteinDatum) -> TensorCloud:
        seq_len, num_atoms = datum.atom_mask.shape
        assert num_atoms == 14, num_atoms
        assert datum.residue_token.shape == (seq_len,)
        assert datum.atom_coord.shape == (seq_len, num_atoms, 3)
        assert datum.atom_mask.shape == (seq_len, num_atoms)
        assert datum.flips_list.shape == (seq_len, 2, 2), datum.flips_list.shape
        assert datum.flips_mask.shape == (seq_len, 2), datum.flips_mask.shape

        embed_dim = self.irreps.filter("0e").dim

        residue_token = datum.residue_token_masked if hasattr(datum, "residue_token_masked") else datum.residue_token
        residue_token_embed = Embed(
            vocab_size=23, 
            embed_dim=embed_dim
        )(residue_token)

        residue_index_embed = Embed(vocab_size=seq_len, embed_dim=embed_dim)(
            jnp.arange(seq_len)
        )

        residue_scalar_embed = residue_token_embed + residue_index_embed

        # if datum.boundary_token is not None:
        #     sos_eos_embed = Embed(vocab_size=3, embed_dim=embed_dim)(
        #         datum.boundary_token
        #     )
        #     residue_scalar_embed = residue_scalar_embed + sos_eos_embed

        flips_index_take = jax.vmap(lambda x, idx: x[idx])
        flippable = (
            flips_index_take(datum.atom_coord, datum.flips_list)
            * datum.flips_mask[..., None, None]
        )
        flip_diffs = flippable[..., 1, :] - flippable[..., 0, :]
        flip_embed = e3nn.spherical_harmonics("2e", flip_diffs, False).array
        assert flip_embed.shape == (seq_len, 2, 5), flip_embed.shape

        def collapse_flips_into_centers(vecs, flip, flip_mask):
            centers = vecs[flip].mean(-2) * flip_mask[:, None]
            indices = jnp.where(flip_mask[:, None], flip, 15)
            vecs = vecs.at[indices[:, 0]].set(centers, mode="drop")
            vecs = vecs.at[indices[:, 1]].set(0.0, mode="drop")
            return vecs


        res_coord = datum.atom_coord_masked[:, 1, :] if hasattr(datum, "atom_coord_masked") else datum.atom_coord[:, 1, :]
        mask_coord = res_coord.sum(-1) != 0.0 

        vector_embed = jnp.zeros_like(datum.atom_coord, dtype=jnp.bfloat16)
        vector_embed = jnp.where(
            repeat(datum.atom_mask, "... -> ... e", e=3),
            datum.atom_coord - res_coord[..., None, :],
            vector_embed,
        )
        vector_embed = jax.vmap(collapse_flips_into_centers)(
            vector_embed, datum.flips_list, datum.flips_mask
        )

        norm_scalar_embed = safe_norm(vector_embed)
        norm_scalar_embed = e3nn.haiku.MultiLayerPerceptron(
            [embed_dim] * 2, jax.nn.silu
        )(norm_scalar_embed)

        scalar_embed = residue_scalar_embed + norm_scalar_embed
        vector_embed = rearrange(vector_embed, "... a e -> ... (a e)")
        flip_embed = rearrange(flip_embed, "... a e -> ... (a e)")

        irreps_array = e3nn.IrrepsArray(
            f"{embed_dim}x0e + 14x1e + 2x2e",
            jnp.concatenate([scalar_embed, vector_embed, flip_embed], axis=-1),
        )

        state = TensorCloud(
            irreps_array=irreps_array,
            coord=res_coord,
            mask_coord=mask_coord,
            mask_irreps_array=datum.residue_mask,
        )

        if self.interact:
            state = SelfInteraction(
                [self.irreps * int(self.rescale)] * self.depth + [self.irreps], norm_last=True
            )(state)

        return state


@jax.custom_vjp
def safe_eigh(x):
    return jnp.linalg.eigh(x)


def safe_eigh_fwd(x):
    w, v = safe_eigh(x)
    return (w, v), (w, v)


def safe_eigh_bwd(res, g):
    w, v = res
    wct, vct = g
    deltas = w[..., jnp.newaxis, :] - w[..., :, jnp.newaxis]
    on_diagonal = jnp.eye(w.shape[-1], dtype=bool)
    F = jnp.where(on_diagonal, 0, 1 / jnp.where(on_diagonal, 1, deltas))
    matmul = partial(jnp.matmul, precision=jax.lax.Precision.HIGHEST)
    vT_ct = matmul(v.T.conj(), vct)
    F_vT_vct = jnp.where(vT_ct != 0, F * vT_ct, 0)  # ignore values that would give NaN
    g = matmul(v, matmul(jnp.diag(wct) + F_vT_vct, v.T.conj()))
    g = (g + g.T.conj()) / 2
    return (g,)


safe_eigh.defvjp(safe_eigh_fwd, safe_eigh_bwd)


def sqrt_2tensor(y):
    """
    sqrt_2tensor(e3nn.sh(2, x, False)) == x or -x
    e3nn.sh(2, sqrt_2tensor(y), False) == y
    """
    assert y.shape == (5,)
    A = e3nn.generators(2) @ y
    A = jnp.conj(A) @ A.T
    val, vec = safe_eigh(A)
    x = vec.T[0]  # first is the smallest eigenvalue
    safe_sqrt = lambda x: jnp.sqrt(jnp.maximum(x, 1e-7))
    x = x * safe_sqrt(safe_sqrt(jnp.mean(y**2)))
    return x, val[0]


class ProteinDatumDecoder(hk.Module):
    def __init__(
        self,
        irreps: e3nn.Irreps,
        interact: bool = True,
        depth: int = 0,
        rescale: float = 2.0,
        ca_only: bool = False,
    ):
        super().__init__()
        self.depth = depth
        if not str(irreps.filter('2e')):
            dim = irreps[0].mul
            irreps = irreps + e3nn.Irreps(f'{dim}x2e')
        self.irreps = irreps
        self.interact = interact
        self.rescale = rescale
        self.ca_only = ca_only

    def __call__(
        self, state: TensorCloud, sequence_token: jnp.ndarray = None,
    ) -> ProteinDatum:
        seq_len = state.irreps_array.shape[0]
        assert state.irreps_array.shape == (seq_len, state.irreps_array.irreps.dim)
        assert state.mask.shape == (seq_len,)
        assert state.coord.shape == (seq_len, 3)
        
        ca_coord = state.coord

        invariants = state.irreps_array.filter(keep="0e")
        logits = e3nn.haiku.MultiLayerPerceptron(
            [invariants.irreps.dim, invariants.irreps.num_irreps, 25], act=jax.nn.silu,
            output_activation=False
        )(invariants).array
        
        res_logits, sos_logits, eos_logits = (
            logits[..., :23],
            logits[..., -1],
            logits[..., -2],
        )

        if sequence_token is None:
            sequence_token = jnp.argmax(res_logits, axis=-1)
            sequence_token = jnp.where(
                jnp.arange(len(sequence_token)) > eos_logits.argmax(-1),
                0, sequence_token
            )

        seq_len = sequence_token.shape[0]
        assert sequence_token.shape == (seq_len,)
        state = jax.tree_util.tree_map(lambda x: x[:seq_len], state)

        logit_extract = repeat(sequence_token, "r -> r l", l=23) == repeat(
            jnp.arange(0, 23), "l -> () l"
        )

        atom_token = (logit_extract[..., None] * all_residues_atom_tokens[None]).sum(-2)
        atom_mask = (logit_extract[..., None] * all_residues_atom_mask[None]).sum(-2)
        assert atom_token.shape == (seq_len, 14), atom_token.shape
        assert atom_mask.shape == (seq_len, 14), atom_mask.shape

        flips = (logit_extract[..., None, None] * flippable_arr[None]).sum(-3)
        flips_mask = (
            (logit_extract[..., None, None] * flippable_mask[None]).sum(-3).squeeze(-1)
        )

        flips = jnp.where(flips_mask[..., None], flips, 0)
        assert flips.shape == (seq_len, 2, 2), flips.shape

        if not self.ca_only: 
            all_vecs_decoded = SelfInteraction(
                [self.irreps * int(self.rescale)] * self.depth
                + [e3nn.Irreps(f"{25}x0e + {23 * 14}x1e + {23}x2e").regroup()],
                chunk_factor=2,
                norm_last=False,
            )(state)

            all_vecs_decoded = all_vecs_decoded.irreps_array.filter("1e + 2e").mul_to_axis(
                23
            )
            vecs_decoded = jax.vmap(lambda arr, idx: arr[idx])(
                all_vecs_decoded, sequence_token
            )
            vecs3 = rearrange(
                vecs_decoded.filter("1e").array, "... (a e) -> ... a e", a=14, e=3
            )
            vecs5 = vecs_decoded.filter("2e").array
            vecs5 = repeat(vecs5, "... e -> ... a e", a=2)

            vecs3 = vecs3 * atom_mask[..., None]
            vecs5 = vecs5 * flips_mask[..., None]
            
            assert vecs3.shape == (seq_len, 14, 3), vecs3.shape
            assert vecs5.shape == (seq_len, 2, 5), vecs5.shape

            flips_extract = (
                repeat(flips, "... -> ... a", a=14)
                == repeat(jnp.arange(0, 14), "a -> () a")
            ) * flips_mask[..., None, None]

            flippable = (vecs3[..., None, None, :, :] * flips_extract[..., None]).sum(-2)
            center = flippable[..., 0, :]

            diff, atom_perm_loss = jax.vmap(jax.vmap(sqrt_2tensor))(vecs5)
            # diff_scale = e3nn.haiku.Linear('0e')(state.irreps_array).array
            # diff = diff * diff_scale[..., None]
            sym_vecs3 = jnp.stack([center + diff, center - diff], axis=-2)
            assert sym_vecs3.shape == (seq_len, 2, 2, 3), sym_vecs3.shape

            atom_perm_loss = (atom_perm_loss * flips_mask).sum() / (flips_mask.sum() + 1e-6)

            sym_vecs3_aggregate = (sym_vecs3[..., None, :] * flips_extract[..., None]).sum(
                (-3, -4)
            )
            subs_mask = sym_vecs3_aggregate.sum(-1) > 0

            vecs3 = jnp.where(subs_mask[..., None], sym_vecs3_aggregate, vecs3)
            vecs3 = vecs3.at[..., 1, :].set(0.0)
        else:
            vecs3 = jnp.zeros((seq_len, 14, 3), dtype=ca_coord.dtype)
            atom_perm_loss = jnp.array([0.0])

        atom_coord = ca_coord[..., None, :] + vecs3 * atom_mask[..., None]
        assert atom_coord.shape == (seq_len, 14, 3), atom_coord.shape

        datum = ProteinDatum(
            idcode=None,
            resolution=None,  # ophiuchus doesn't label resolution
            sequence=None,  # str and jax dont like each other
            residue_token=sequence_token,
            residue_index=jnp.arange(res_logits.shape[0]),
            residue_mask=sequence_token != 0,
            chain_token=None,  # TODO(Allan)
            atom_token=atom_token,  # TODO(Allan)
            atom_coord=atom_coord,
            atom_mask=atom_mask,
        )

        return (
            datum,
            (res_logits, sos_logits, eos_logits),
            atom_perm_loss,
        )


def protein_to_tensor_cloud(protein):
    res_token = protein['residue_token']
    res_mask = protein['atom_mask'][..., 1]

    scalars = jax.nn.one_hot(res_token, 23)
    scalars = scalars * res_mask[..., None]

    mask = protein['atom_mask']
    vectors = protein['atom_coord']
    ca_coord = vectors[..., 1, :]

    vectors = (vectors - ca_coord[..., None, :])
    vectors = vectors * mask[..., None]
    vectors = rearrange(vectors, 'r a c -> r (a c)')

    irreps_array = e3nn.IrrepsArray(
        '23x0e + 14x1e', 
        jnp.concatenate([scalars, vectors], axis=-1)
    )
    
    # if type(mask) == np.ndarray:
        # mask[..., 1] = False
    # else:
        # mask.at[..., 1].set(False)

    state = TensorCloud(
        irreps_array=irreps_array,
        mask_irreps_array=mask,
        coord=ca_coord,
        mask_coord=res_mask,
    )

    return state 


def tensor_cloud_to_protein(state, protein=None, backbone_only=False):
    irreps_array = state.irreps_array
    ca_coord = state.coord
    res_mask = state.mask_coord

    if (str(irreps_array.irreps) == '23x0e+14x1e'):
        res_logits = jax.nn.softmax(irreps_array.filter('0e').array)
        eos_logits = None
    elif (str(irreps_array.irreps) == '14x1e'):
        res_logits = None
        eos_logits = None
    else:
        invariants = irreps_array.filter(keep='0e')
        logits = e3nn.haiku.MultiLayerPerceptron(
            [invariants.irreps.dim, invariants.irreps.num_irreps, 25], act=jax.nn.silu,
            output_activation=False
        )(invariants).array

        res_logits, sos_logits, eos_logits = (
            logits[..., :23],
            logits[..., -1],
            logits[..., -2],
        )
        irreps_array = e3nn.haiku.Linear('14x1e')(irreps_array)

    atom_coord = irreps_array.filter("1e").array
    atom_coord = rearrange(atom_coord, 'r (a c) -> r a c', a=14)
        

    if protein is None and not backbone_only:
        sequence_token = jnp.argmax(res_logits, axis=-1)
    elif protein is None and backbone_only:
        sequence_token = jnp.full(res_logits.shape[0], all_residues.index('GLY'))
    else:
        sequence_token = protein['residue_token']

    logit_extract = repeat(sequence_token, "r -> r l", l=23) == repeat(
        jnp.arange(0, 23), "l -> () l"
    )

    if protein is None:
        atom_token = (logit_extract[..., None] * all_residues_atom_tokens[None]).sum(-2)
        atom_mask = (logit_extract[..., None] * all_residues_atom_mask[None]).sum(-2)
    else: 
        atom_token = protein['atom_token']
        atom_mask = protein['atom_mask']
        
    atom_coord = atom_coord.at[..., 1, :].set(0.0)
    atom_coord = atom_coord + ca_coord[..., None, :]
    atom_coord = atom_coord * atom_mask[..., None]

    return dict(
        idcode=None,
        resolution=None,
        sequence=None,
        residue_token=sequence_token,
        residue_index=jnp.arange(sequence_token.shape[0]),
        residue_mask=res_mask,
        chain_token=jnp.zeros(sequence_token.shape[0], dtype=jnp.int32),
        atom_token=atom_token,
        atom_coord=atom_coord,
        atom_mask=atom_mask,
        residue_logits=res_logits,
        eos_logits=eos_logits,
    )    

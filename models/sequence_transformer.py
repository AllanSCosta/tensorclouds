from typing import NamedTuple
import haiku as hk

from .transformer import Transformer

import jax.numpy as jnp

class SequenceTransformerModelOutput(NamedTuple):
    res_logits: jnp.ndarray


class SequenceTransformer(hk.Module):

    def __init__(self, dim, num_layers, num_heads, attn_size, dropout_rate, widening_factor):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor


    def __call__(self, datum, is_training=False):
        hidden = hk.Embed(
            vocab_size=22, 
            embed_dim=self.dim
        )(datum.residue_token)

        hidden = Transformer(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            attn_size=self.attn_size,
            dropout_rate=self.dropout_rate,
            widening_factor=self.widening_factor
        )(hidden[None], datum.atom_mask[None, :, 1])[0]

        logits = hk.Linear(
            output_size=23
        )(hidden)
        
        return SequenceTransformerModelOutput(
            res_logits=logits
        ) 
        
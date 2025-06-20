import jax
import flax.linen as nn
import e3nn_jax as e3nn
import jax.numpy as jnp
from ..tensorcloud import TensorCloud


from .attention import EquivariantSelfAttention
from .feed_forward import FeedForward
from .layer_norm import EquivariantLayerNorm


class TransformerBlock(nn.Module):

    irreps: e3nn.Irreps
    # k: int = 0
    k_seq: int = 0
    radial_cut: float = 24.0
    radial_bins: int = 42
    radial_basis: str = "gaussian"
    move: bool = False
    
    @nn.compact
    def __call__(self, x):
        res = x

        x = EquivariantSelfAttention(
            irreps_out=self.irreps,
            # k=self.k,
            k_seq=self.k_seq,
            radial_cut=self.radial_cut,
            radial_bins=self.radial_bins,
            radial_basis=self.radial_basis,
            move=self.move,
        )(x)

        x = x.replace(
            irreps_array=EquivariantLayerNorm()(res.irreps_array + x.irreps_array))
        res = x

        x = FeedForward(self.irreps, 4)(x)
        x = x.replace(
            irreps_array=EquivariantLayerNorm()(res.irreps_array + x.irreps_array))

        return x
    

class Transformer(nn.Module):
    
    irreps: e3nn.Irreps
    depth: int

    k_seq: int = 16
    radial_cut: float = 24.0
    move: bool = False

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = TransformerBlock(
                irreps=self.irreps,
                k_seq=self.k_seq,
                radial_cut=self.radial_cut,
                move=self.move,
            )(x)
        return x
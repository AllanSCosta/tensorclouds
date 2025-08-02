from functools import reduce
from typing import List

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp

from ..tensorcloud import TensorCloud
from .spatial import kNNSpatialConvolution
from .embed import Embed, PairwiseEmbed
from .feed_forward import FeedForward
from .residual import Residual
from .self_interaction import SelfInteraction

from typing import Callable

class TensorFieldBlock(nn.Module):

    irreps: e3nn.Irreps
    ff_factor: int = 4

    radial_dim: int = 32
    radial_cut: float = 32.0
    radial_basis: str = "gaussian"
    k: int = 64
    k_seq: int = 32
    edge_irreps: e3nn.Irreps = e3nn.Irreps("1x0e + 1x1e")
    activation: Callable = jax.nn.silu
    envelope: bool = True
    move: bool = False

    @nn.compact
    def __call__(self, x: TensorCloud) -> TensorCloud:
        x = Residual(
            kNNSpatialConvolution(
                irreps_out=self.irreps,
                k_seq=self.k_seq,
                k=self.k,
                radial_dim=self.radial_dim,
                radial_basis=self.radial_basis,
                radial_cut=self.radial_cut,
                edge_irreps=self.edge_irreps,
                activation=self.activation,
                move=self.move,
            )
        )(x)
        return Residual(SelfInteraction(self.irreps))(x)


class TensorFieldNetwork(nn.Module):


    irreps: e3nn.Irreps
    depth: int

    ff_factor: int = 4
    radial_dim: int = 32
    radial_cut: float = 32.0
    radial_basis: str = "gaussian"
    
    k: int = 64
    k_seq: int = 20

    edge_irreps: e3nn.Irreps = e3nn.Irreps("1x0e + 1x1e")
    activation: Callable = jax.nn.silu
    envelope: bool = True
    move: bool = False


    @nn.compact
    def __call__(self, x: TensorCloud) -> TensorCloud:
        return reduce(
            lambda x, _: TensorFieldBlock(
                irreps=self.irreps,
                radial_dim=self.radial_dim,
                radial_cut=self.radial_cut,
                radial_basis=self.radial_basis,
                k=self.k,
                k_seq=self.k_seq,
                edge_irreps=self.edge_irreps,
                activation=self.activation,
                envelope=self.envelope,
                move=self.move,
                ff_factor=self.ff_factor,
            )(x),
            range(self.depth),
            x,
        )

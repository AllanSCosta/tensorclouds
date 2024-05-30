from typing import List
import haiku as hk

from .decoder import Decoder
from .encoder import Encoder

from ..tensorcloud import TensorCloud 
import e3nn_jax as e3nn

class Autoencoder(hk.Module):

    def __init__(
        self, 
        irreps: e3nn.Irreps,
        layers: List[int],
        rescale: float,
        stride: int,
        kernel_size: int,
    ):
        super().__init__(name='autoencoder')
        self.encoder = Encoder(
            irreps,
            layers,
            rescale,
            stride,
            kernel_size,
        )
        self.decoder = Decoder(
            irreps,
            layers,
            rescale,
            stride,
            kernel_size,
        )

    def __call__(
        self,
        input: TensorCloud = None,
        bottleneck: TensorCloud = None,
    ):
        if input is not None:
            encoder_internals = self.encoder(input)
            skip = encoder_internals[-1]
        else:
            skip = bottleneck
            encoder_internals = None
        tc, decoder_internals = self.decoder(
            skips=skip,
            is_training=True,
        )
        return tc, encoder_internals, encoder_internals, decoder_internals
import chex
from ..base.utils import TensorCloud

@chex.dataclass
class DiffusionStepOutput:
    noise_prediction: TensorCloud
    noise: dict
    reweight: float

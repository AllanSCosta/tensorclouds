import chex
import e3nn_jax as e3nn
from ..tensorcloud import TensorCloud


@chex.dataclass
class ModelPrediction:
    prediction: TensorCloud
    target: dict
    reweight: float = None

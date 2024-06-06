
import chex
from ..tensorcloud import TensorCloud

@chex.dataclass
class ModelPrediction:
    prediction: TensorCloud
    target: dict
    reweight: float = None
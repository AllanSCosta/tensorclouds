import re
import jax.numpy as jnp

from .schedulers import Scheduler

from typing import Tuple, Dict

from moleculib.protein.datum import ProteinDatum


class ModelOutput:
    def __init__(self):
        pass


class LossFunction:
    def __init__(
        self, weight: float = 1.0, start_step: int = 0, scheduler: Scheduler = None
    ):
        self.weight = weight
        self.start_step = start_step
        self.scheduler = scheduler

    def _call(
        self, model_output: ModelOutput, ProteinDatum: Dict
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        raise NotImplementedError

    def __call__(
        self,
        rng_key,
        model_output: ModelOutput,
        batch: ProteinDatum,
        step: int,
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        output, loss, metrics = self._call(rng_key, model_output, batch)
        is_activated = jnp.array(self.start_step <= step).astype(loss.dtype)
        loss = loss * is_activated
        if self.scheduler is not None:
            scheduler_weight = self.scheduler(step)
            loss = loss * scheduler_weight
            loss_name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()
            metrics[loss_name + "_scheduler"] = scheduler_weight
        return output, self.weight * loss, metrics


import re
import jax
import jax.numpy as jnp
from einops import rearrange, repeat

from .schedulers import Scheduler

import e3nn_jax as e3nn
from functools import partial
from typing import Callable, Tuple, Dict, List
from collections import defaultdict

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


class LossPipe:
    def __init__(self, loss_list: List[LossFunction]):
        self.loss_list = loss_list

    def __call__(
        self,
        rng_key,
        model_output: ModelOutput,
        batch: ProteinDatum,
        step: int,
    ):
        loss = 0.0
        metrics = {}
        for loss_fn in self.loss_list:
            model_output, loss_fn_loss, loss_fn_metrics = loss_fn(
                rng_key, model_output, batch, step
            )
            loss += loss_fn_loss
            metrics.update(loss_fn_metrics)
        return model_output, loss, metrics

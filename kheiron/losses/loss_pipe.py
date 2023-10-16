from typing import List
from moleculib.protein.datum import ProteinDatum
from .loss_function import LossFunction, ModelOutput

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

import jax 
import jax.numpy as jnp

from moleculib.protein.datum import ProteinDatum
from .losses import LossFunction, ModelOutput
from typing import Dict, Tuple, Any


class ResidueCrossEntropyLoss(LossFunction):

    def _cross_entropy_loss(self, logits, labels, mask=None):
        if mask is not None:
            logits = jnp.where(mask[..., None], logits, jnp.array([1] + [0] * 22))
        cross_entropy = -(labels * jax.nn.log_softmax(logits)).sum(-1)
        return cross_entropy.mean()

    def _call(
        self, rng_key, model_output: Any, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        assert hasattr(model_output, "res_logits")
        
        res_logits = model_output.res_logits
        total_loss, metrics = 0.0, {}

        res_mask = ground.atom_mask[..., 1]
        res_labels = jax.nn.one_hot(ground.residue_token, 23)
        res_cross_entropy = self._cross_entropy_loss(
            res_logits, res_labels, mask=res_mask
        )
        metrics["res_cross_entropy"] = res_cross_entropy.mean()
        total_loss += res_cross_entropy.mean()

        pred_labels = res_logits.argmax(-1)
        res_accuracy = pred_labels == ground.residue_token
        res_accuracy = (res_accuracy * res_mask).sum() / (res_mask.sum() + 1e-6)
        res_accuracy = res_accuracy * (res_mask.sum() > 0).astype(res_accuracy.dtype)
        metrics["res_accuracy"] = res_accuracy

        return model_output, total_loss, metrics
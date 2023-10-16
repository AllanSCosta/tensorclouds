import jax 
import jax.numpy as jnp

from moleculib.protein.datum import ProteinDatum
from .loss_function import LossFunction, ModelOutput
from typing import Dict, Tuple, Any

from moleculib.protein.alphabet import all_residues


class MaskedLanguageLoss(LossFunction):

    def _call(
        self, rng_key, model_output: Any, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        assert hasattr(model_output, "res_logits")
        metrics = {}
        logits = model_output.res_logits
        labels = jax.nn.one_hot(ground.residue_token, 23)

        res_mask = ground.atom_mask[..., 1]
        masked_tokens_mask = ground.residue_token_masked == all_residues.index("MASK") 
        mask = res_mask * masked_tokens_mask

        cross_entropy = -(labels * jax.nn.log_softmax(logits))
        cross_entropy = (cross_entropy * mask[:, None])
        cross_entropy = cross_entropy.sum((-1, -2)) / (mask.sum(-1) + 1e-6)
        cross_entropy = cross_entropy * (mask.sum(-1) > 0).astype(cross_entropy.dtype)

        metrics["res_cross_entropy"] = cross_entropy

        pred_labels = logits.argmax(-1)
        res_accuracy = pred_labels == ground.residue_token
        res_accuracy = (res_accuracy * mask).sum() / (mask.sum() + 1e-6)
        res_accuracy = res_accuracy * (mask.sum() > 0).astype(res_accuracy.dtype)
        metrics["res_accuracy"] = res_accuracy

        return model_output, cross_entropy, metrics
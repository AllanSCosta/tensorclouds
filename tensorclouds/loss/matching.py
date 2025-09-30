from learnax.loss import LossFunction
import jax
import jax.numpy as jnp
from collections import defaultdict
from typing import Tuple, Dict, Any

import re
import e3nn_jax as e3nn
from moleculib.protein.datum import ProteinDatum
import einops as ein



def safe_norm(vector: jax.Array, axis: int = -1) -> jax.Array:
    """safe_norm(x) = norm(x) if norm(x) != 0 else 1.0"""
    norms_sqr = jnp.sum(vector**2, axis=axis)
    norms = jnp.where(norms_sqr == 0.0, 1.0, norms_sqr) ** 0.5
    return norms

import optax

class TensorCloudMatchingLoss(LossFunction):

    def _call(
        self,
        model_output: Any,
        ground: ProteinDatum,
        reduction="mean",
    ) -> Tuple[Any, jax.Array, Dict[str, float]]:

        if isinstance(model_output, list):
            total_loss = 0.0
            total_metrics = defaultdict(float)
            for model_out in model_output:
                _, loss, metrics = self._call(model_out, ground, reduction=reduction)
                total_loss += loss
                for k, v in metrics.items():
                    total_metrics[k] += v
            for k in total_metrics.keys():
                total_metrics[k] /= len(model_output)
            return model_output, total_loss / len(model_output), dict(total_metrics)

        pred, target = model_output.prediction, model_output.target
        breakpoint()
        
        def vector_map_loss(pred, target, mask):
            vector_map = lambda x: (ein.rearrange(x, "i c -> i () c") 
                                    - ein.rearrange(x, "j c -> () j c"))
            cross_mask = (ein.rearrange(mask, "i -> i ()") 
                          & ein.rearrange(mask, "j -> () j"))
            
            vector_maps = vector_map(pred)
            vector_maps_target = vector_map(target)
            
            cross_mask = cross_mask & (safe_norm(vector_maps_target) < 25.0)

            error = optax.huber_loss(vector_maps, vector_maps_target).sum(-1)
            
            error = (error * cross_mask)
            error = error.sum((-1, -2)) / (cross_mask.sum((-1, -2)) + 1e-6)
            # error = error.mean() * (cross_mask.sum() > 0).astype(error.dtype)
            return error 
        
        vecs = lambda irreps_array: ein.rearrange(irreps_array.array, "i (d e)-> i d e", e=3)

        feat_loss = jax.vmap(vector_map_loss)(pred=vecs(pred.irreps_array), target=vecs(target.irreps_array), mask=target.mask_irreps_array)
        feat_loss = jnp.sum(feat_loss) / (jnp.sum(target.mask_coord))
        
        coord_loss = vector_map_loss(pred.coord, target.coord, target.mask_coord)

        features_pred_norm = jnp.square(pred.irreps_array.array).sum(-1)
        features_pred_norm = jnp.mean(features_pred_norm)

        features_target_norm = jnp.square(target.irreps_array.array).sum(-1)
        features_target_norm = jnp.mean(features_target_norm)

        metrics = dict(
            features_loss=feat_loss,
            coord_loss=coord_loss,
            features_pred_norm=features_pred_norm,
            features_target_norm=features_target_norm,
        )

        return model_output, feat_loss + coord_loss, metrics

class TensorCloudMatchingLoss(LossFunction):

    def _call(
        self,
        rng_key,
        model_output: ModelOutput,
        _: ProteinDatum,
        reduction="sum",
    ) -> Tuple[ModelOutput, jax.Array, Dict[str, float]]:

        if type(model_output) == tuple:
            aggr_loss = 0.0
            metrics = defaultdict(float)
            for output in model_output:
                _, loss_, metrics_ = self._call(rng_key, output, _)
                name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(output).__name__).lower()
                aggr_loss += loss_
                for key, value in metrics_.items():
                    metrics[name + "_" + key] = value
            return model_output, aggr_loss, metrics

        pred, target = model_output.prediction, model_output.target
        if hasattr(model_output, "reweight"):
            reweight = jax.lax.stop_gradient(model_output.reweight)
        else:
            reweight = 1.0

        features_loss = jnp.square(pred.irreps_array.array - target.irreps_array.array)
        features_loss = reweight * features_loss

        features_mask = (
            target.mask_irreps_array
            * e3nn.ones(target.irreps_array.irreps, target.irreps_array.shape[:-1])
        ).array
        features_loss = jnp.sum(features_loss * features_mask)

        if reduction == "mean":
            features_loss = features_loss / (jnp.sum(features_mask) + 1e-6)

        features_pred_norm = jnp.square(pred.irreps_array.array).sum(-1)
        features_pred_norm = jnp.mean(features_pred_norm)

        features_target_norm = jnp.square(target.irreps_array.array).sum(-1)
        features_target_norm = jnp.mean(features_target_norm)

        coord_loss = jnp.square(pred.coord - target.coord)
        coord_loss = reweight * coord_loss
        coord_loss = jnp.sum(coord_loss * target.mask_coord[..., None])

        if reduction == "mean":
            coord_loss = coord_loss / (jnp.sum(target.mask_coord) + 1e-6)

        metrics = dict(
            features_loss=features_loss,
            features_pred_norm=features_pred_norm,
            features_target_norm=features_target_norm,
            coord_loss=coord_loss,
        )

        return model_output, features_loss + coord_loss, metrics

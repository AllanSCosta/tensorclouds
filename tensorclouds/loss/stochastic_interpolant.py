class StochasticInterpolantLoss(LossFunction):

    def _call(self, _, model_output: ModelOutput, __: ProteinDatum):
        if type(model_output) == tuple:
            aggr_loss = 0.0
            metrics = defaultdict(float)
            for output in model_output:
                _, loss_, metrics_ = self._call(_, output, __)
                name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(output).__name__).lower()
                aggr_loss += loss_
                for key, value in metrics_.items():
                    metrics[name + "_" + key] = value
            return model_output, aggr_loss, metrics

        pred = model_output.prediction
        target = model_output.target
        pred = pred.replace(mask_irreps_array=target.mask_irreps_array)

        def stochastic_interpolant_loss(pred, target):
            feature_dot1, coord_dot1 = pred.norm()
            feature_dot2, coord_dot2 = pred.dot(target)

            feature_loss = 0.5 * feature_dot1 + feature_dot2
            coord_loss = 0.5 * coord_dot1 + coord_dot2

            feature_loss = 100 * feature_loss.mean()
            coord_loss = 100 * coord_loss.mean()
            return feature_loss, coord_loss

        features_loss, coord_loss = stochastic_interpolant_loss(pred, -target)

        return (
            model_output,
            features_loss + coord_loss,
            {"features_loss": features_loss, "coord_loss": coord_loss},
        )

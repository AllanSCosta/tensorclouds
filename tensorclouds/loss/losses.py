import re
import jax
import jax.numpy as jnp
from einops import rearrange, repeat

from tensorclouds.train.schedulers import Scheduler

import e3nn_jax as e3nn
import optax
from typing import Callable, Tuple, Dict, List
from collections import defaultdict

from tensorclouds.nn.utils import ModelOutput, safe_norm, safe_normalize

from moleculib.protein.datum import ProteinDatum
from moleculib.assembly.datum import AssemblyDatum


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
        step: int = 0,
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


class ApplyLossToProteins(LossFunction):

    def __init__(
        self,
        weight=1.0,
        start_step=0,
        loss_pipe: LossPipe = [],
        independent: bool = True,
    ):
        self.losses = loss_pipe
        self.independent = independent
        super().__init__(weight=weight, start_step=start_step)

    def __call__(self, rng_key, model_output: ModelOutput, ground: AssemblyDatum, step):
        if self.independent:
            output_protein_data, loss, metrics = jax.vmap(
                self.losses, in_axes=(None, 0, 0, None)
            )(rng_key, model_output.datum.protein_data, ground.protein_data, step)
            model_output = model_output.replace(
                datum=model_output.datum.replace(protein_data=output_protein_data)
            )
        else:
            raise NotImplementedError
        return model_output, loss, metrics


class AtomPermLoss(LossFunction):
    def _call(
        self, rng_key, model_output: ModelOutput, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        loss = model_output.atom_perm_loss
        return model_output, loss.mean(), {"atom_perm_loss": loss.mean()}


class MirrorInterpolantLoss(LossFunction):

    def _call(self, _, model_output: ModelOutput, __: ProteinDatum):
        pred = model_output.noise_prediction
        z = model_output.noise

        feature_dot1, coord_dot1 = pred.norm()
        feature_dot2, coord_dot2 = pred.dot(-z)

        feature_loss = 0.5 * feature_dot1 + feature_dot2
        coord_loss = 0.5 * coord_dot1 + coord_dot2

        feature_loss = 100 * feature_loss.mean()
        coord_loss = 100 * coord_loss.mean()

        return (
            model_output,
            feature_loss + coord_loss,
            {"feature_loss": feature_loss, "coord_loss": coord_loss},
        )


class StochasticInterpolantLoss(LossFunction):

    def _call(self, _, model_output: ModelOutput, __: ProteinDatum):
        noise_prediction, drift_prediction = model_output

        pred_d = drift_prediction.prediction
        d = drift_prediction.target

        pred_z = noise_prediction.prediction
        z = noise_prediction.target

        def stochastic_interpolant_loss(pred, target):
            feature_dot1, coord_dot1 = pred.norm()
            feature_dot2, coord_dot2 = pred.dot(target)

            feature_loss = 0.5 * feature_dot1 + feature_dot2
            coord_loss = 0.5 * coord_dot1 + coord_dot2

            feature_loss = 100 * feature_loss.mean()
            coord_loss = 100 * coord_loss.mean()
            return feature_loss, coord_loss

        feature_loss_d, coord_loss_d = stochastic_interpolant_loss(pred_d, -d)
        feature_loss_z, coord_loss_z = stochastic_interpolant_loss(pred_z, -z)

        return (
            model_output,
            feature_loss_d + coord_loss_d + feature_loss_z + coord_loss_z,
            {
                "feature_loss_d": feature_loss_d,
                "coord_loss_d": coord_loss_d,
                "feature_loss_z": feature_loss_z,
                "coord_loss_z": coord_loss_z,
            },
        )


class TensorCloudMatchingLoss(LossFunction):

    def _call(
        self,
        rng_key,
        model_output: ModelOutput,
        _: ProteinDatum,
        reduction="sum",
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:

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
        features_loss = 100 * jnp.sum(features_loss * features_mask)
        if reduction == "mean":
            features_loss = features_loss / (jnp.sum(features_mask) + 1e-6)

        features_pred_norm = jnp.square(pred.irreps_array.array).sum(-1)
        features_pred_norm = jnp.mean(features_pred_norm)

        features_target_norm = jnp.square(target.irreps_array.array).sum(-1)
        features_target_norm = jnp.mean(features_target_norm)

        coord_loss = jnp.square(pred.coord - target.coord)
        coord_loss = reweight * coord_loss
        coord_loss = 100 * jnp.sum(coord_loss * target.mask_coord[..., None])
        if reduction == "mean":
            coord_loss = coord_loss / (jnp.sum(target.mask_coord) + 1e-6)

        metrics = dict(
            features_loss=features_loss,
            features_pred_norm=features_pred_norm,
            features_target_norm=features_target_norm,
            coord_loss=coord_loss,
        )

        return model_output, features_loss + coord_loss, metrics


class VectorCloudMatchingLoss(LossFunction):

    def _call(
        self, rng_key, model_output: ModelOutput, _: ProteinDatum, ca_index: int = 1
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        pred, target = model_output.prediction, model_output.target

        vec_irreps = "1e"
        pred_vectors = rearrange(
            pred.irreps_array.filter(vec_irreps).array, "... (v e) -> ... v e", e=3
        )
        target_vectors = rearrange(
            target.irreps_array.filter(vec_irreps).array, "... (v e) -> ... v e", e=3
        )
        vec_mask = target_vectors.sum(-1) != 0.0

        pred_vectors = (
            pred_vectors.at[..., ca_index, :].set(0.0) + pred.coord[:, None, :]
        )
        target_vectors = (
            target_vectors.at[..., ca_index, :].set(0.0) + target.coord[:, None, :]
        )

        pred_vectors = rearrange(pred_vectors, "r v ... -> (r v) ...")
        target_vectors = rearrange(target_vectors, "r v ... -> (r v) ...")
        vec_mask = rearrange(vec_mask, "r v -> (r v)")

        def ij_map(x, distance=True):
            x_ij = rearrange(x, "... i c -> ... i () c") - rearrange(
                x, "... j c -> ... () j c"
            )
            return safe_norm(x_ij)[..., None] if distance else x_ij

        pred_dist_map = ij_map(pred_vectors)
        target_dist_map = ij_map(target_vectors)

        cross_mask = rearrange(vec_mask, "i -> i ()") & rearrange(vec_mask, "j -> () j")
        vectors_loss = jnp.square(pred_dist_map - target_dist_map)

        vectors_loss = jnp.sum(vectors_loss * cross_mask[..., None]) / (
            jnp.sum(cross_mask) + 1e-6
        )

        pred_coord_dist_map = ij_map(pred.coord)
        target_coord_dist_map = ij_map(target.coord)

        cross_mask = rearrange(target.mask_coord, "r -> r ()") & rearrange(
            target.mask_coord, "r -> () r"
        )
        coord_loss = jnp.square(pred_coord_dist_map - target_coord_dist_map)

        coord_loss = jnp.sum(coord_loss * cross_mask[..., None]) / (
            jnp.sum(cross_mask) + 1e-6
        )

        return (
            model_output,
            vectors_loss + coord_loss,
            {"vectors_loss": vectors_loss, "coord_loss": coord_loss},
        )


class FrameLoss(LossFunction):

    def _call(
        self, rng_key, model_output: ModelOutput, _: ProteinDatum
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        pred_frame = model_output.frame_prediction
        if pred_frame == None:
            return model_output, 0.0, {}

        vectors = pred_frame.filter("1e").array
        vectors = rearrange(vectors, "... (v e) -> ... v e", e=3)

        vector_norm = safe_norm(vectors)
        vec_ij = rearrange(vectors, "... i c -> ... i () c") * rearrange(
            vectors, "... j c -> ... () j c"
        )
        dot_ij = vec_ij.mean(-1)

        loss_norm = jnp.square(vector_norm - 1.0).sum()
        loss_dot = jnp.square(dot_ij).mean()

        return (
            model_output,
            50 * loss_norm + 100 * loss_dot,
            {"loss_norm": loss_norm.mean(), "loss_dot": loss_dot.mean()},
        )


class InternalVectorLoss(LossFunction):
    def __init__(self, weight=1.0, start_step=0, norm_only=False):
        super().__init__(weight=weight, start_step=start_step)
        self.norm_only = norm_only

    def _call(
        self, rng_key, model_output: ModelOutput, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        def flip_atoms(coord, flips, mask):
            flips_list = jnp.where(mask[..., None], flips, 15)
            p, q = flips_list.T
            p_coords, q_coords = coord[p], coord[q]
            coord = coord.at[p].set(q_coords, mode="drop")
            coord = coord.at[q].set(p_coords, mode="drop")
            return coord

        def _internal_vector_loss(coords, ground_coords, mask):
            cross_mask = rearrange(mask, "... i -> ... i ()") & rearrange(
                mask, "... j -> ... () j"
            )
            cross_mask = cross_mask.astype(coords.dtype)

            cross_vectors = rearrange(coords, "... i c -> ... i () c") - rearrange(
                coords, "... j c -> ... () j c"
            )
            cross_ground_vectors = rearrange(
                ground_coords, "... i c -> ... i () c"
            ) - rearrange(ground_coords, "... j c -> ... () j c")
            if self.norm_only:
                cross_vectors = safe_norm(cross_vectors)[..., None]
                cross_ground_vectors = safe_norm(cross_ground_vectors)[..., None]

            error = optax.huber_loss(
                cross_vectors, cross_ground_vectors, delta=1.0
            ).mean(-1)
            error = (error * cross_mask).sum((-1, -2)) / (
                cross_mask.sum((-1, -2)) + 1e-6
            )
            error = error * (cross_mask.sum() > 0).astype(error.dtype)
            return error

        coords = model_output.datum.atom_coord
        alternative = flip_atoms(coords, ground.flips_list, ground.flips_mask)
        loss = _internal_vector_loss(coords, ground.atom_coord, ground.atom_mask)
        alternate_loss = _internal_vector_loss(
            alternative, ground.atom_coord, ground.atom_mask
        )

        coords = jnp.where(
            (loss < alternate_loss)[..., None, None], coords, alternative
        )

        loss = jnp.where(loss < alternate_loss, loss, alternate_loss)

        new_datum = dict(vars(model_output.datum).items())
        new_datum.update(atom_coord=coords)
        datum = ProteinDatum(**new_datum)

        model_output = model_output.replace(datum=datum)

        return model_output, loss.mean(), {"internal_vector_loss": loss.mean()}


TS_CUTOFFS = [1, 2, 4, 8]
HA_CUTOFFS = [0.5, 1, 2, 4]


def _measure_gdt(x: ProteinDatum, y: ProteinDatum, cutoffs):
    ca_x = x.atom_coord[:, 1, :]
    ca_y = y.atom_coord[:, 1, :]
    mask = x.atom_mask[:, 1] & y.atom_mask[:, 1]

    GDT = jnp.zeros(len(cutoffs))
    dist = jnp.sqrt(jnp.square(ca_x - ca_y).sum(-1))

    # iterate over thresholds
    for i, cutoff in enumerate(cutoffs):
        counts = dist <= cutoff
        means = (counts * mask).sum(-1) / (mask.sum(-1) + 1e-6)
        means = means * (mask.sum(-1) > 0).astype(means.dtype)
        GDT = GDT.at[i].set(means)
    return GDT.mean(-1)


def _measure_msd(x: ProteinDatum, y: ProteinDatum, mode="all_atom"):
    mask = x.atom_mask & y.atom_mask
    x = x.atom_coord
    y = y.atom_coord

    if mode == "CA":
        x = x[:, 1:2, :]
        y = y[:, 1:2, :]
        mask = mask[:, 1:2].astype(jnp.float32)

        x = x - (x * mask[..., None]).sum(0)[None] / (mask.sum(0)[None] + 1e-6)
        y = y - (y * mask[..., None]).sum(0)[None] / (mask.sum(0)[None] + 1e-6)
        x = x * mask[..., None]
        y = y * mask[..., None]

    dists = jnp.square(x - y).sum(-1)
    dists = (dists * mask).sum() / (mask.sum() + 1e-6)
    dists = dists * (mask.sum() > 0).astype(dists.dtype)

    return dists


class BottleneckRegularization(LossFunction):

    def __init__(
        self,
        weight=1.0,
        start_step=0,
        max_norm=1.5,
    ):
        super().__init__(weight=weight, start_step=start_step)
        self.max_norm = max_norm

    def _call(self, rng_key, model_output: ModelOutput, ground: ProteinDatum):
        skip = model_output.encoder_internals[-1]
        mask = skip.mask_irreps_array

        total_loss = 0.0
        metrics = {}

        masked_mean = lambda x: ((x * mask[..., None]).sum((-1, -2)) / mask.sum(-1)) * (
            mask.sum(-1) > 0
        ).astype(x.dtype)

        def _angle_map(x):
            x = safe_normalize(x)
            xij = rearrange(x, "... i c -> ... i () c")
            xji = rearrange(x, "... j c -> ... () j c")
            return (xij * xji).sum(-1)

        for l, zl in enumerate(skip.irreps_array.list):
            norms = jnp.square(zl).sum(-1)

            angle_spread = jnp.square(_angle_map(zl)).mean((-1, -2))

            metrics[f"skip[l={l}]_norm"] = masked_mean(norms.mean(-1))
            metrics[f"skip[l={l}]_max_norm"] = jnp.max(norms)
            metrics[f"skip[l={l}]_angular_spread"] = masked_mean(angle_spread)

            l_loss = jax.nn.relu(norms - self.max_norm).sum(-1)
            l_loss = masked_mean(l_loss) + 4 * masked_mean(angle_spread)

            total_loss += l_loss

        return model_output, total_loss, metrics


class VectorMapLoss(LossFunction):
    def __init__(
        self,
        weight=1.0,
        start_step=0,
        max_radius: float = 32.0,
        max_error: float = 800.0,
        norm_only=False,
    ):
        super().__init__(weight=weight, start_step=start_step)
        self.norm_only = norm_only
        self.max_radius = max_radius
        self.max_error = max_error

    def _call(
        self, rng_key, prediction: ProteinDatum, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:

        all_atom_coords = rearrange(prediction.atom_coord, "... a c -> (... a) c")
        all_atom_coords_ground = rearrange(ground.atom_coord, "... a c -> (... a) c")
        all_atom_mask = rearrange(ground.atom_mask, "... a -> (... a)")

        vector_map = lambda x: rearrange(x, "i c -> i () c") - rearrange(
            x, "j c -> () j c"
        )

        cross_mask = rearrange(all_atom_mask, "i -> i ()") & rearrange(
            all_atom_mask, "j -> () j"
        )

        vector_maps = vector_map(all_atom_coords)
        vector_maps_ground = vector_map(all_atom_coords_ground)
        cross_mask = cross_mask & (safe_norm(vector_maps_ground) < self.max_radius)

        if self.norm_only:
            vector_maps = safe_norm(vector_maps)[..., None]
            vector_maps_ground = safe_norm(vector_maps_ground)[..., None]

        error = optax.huber_loss(vector_maps, vector_maps_ground, delta=1.0).mean(-1)
        if self.max_error > 0.0:
            error = jnp.clip(error, 0.0, self.max_error)

        error = (error * cross_mask.astype(error.dtype)).sum((-1, -2)) / (
            cross_mask.sum((-1, -2)) + 1e-6
        )
        error = error.mean()
        error = error * (cross_mask.sum() > 0).astype(error.dtype)

        metrics = dict(
            cross_vector_loss=error,
        )

        return prediction, error, metrics


class AllAtomRMSD(LossFunction):

    def _call(
        self, rng_key, model_output: ModelOutput, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        all_atom_msd = _measure_msd(model_output.datum, ground, mode="all_atom")
        ca_msd = _measure_msd(model_output.datum, ground, mode="CA")
        metrics = dict(
            all_atom_rmsd=jnp.sqrt(all_atom_msd + 1e-6),
            ca_rmsd=jnp.sqrt(ca_msd + 1e-6),
            gdt_ts=_measure_gdt(model_output.datum, ground, TS_CUTOFFS),
            gdt_ha=_measure_gdt(model_output.datum, ground, HA_CUTOFFS),
        )
        return model_output, ca_msd, metrics


class CrossEntropyLoss(LossFunction):

    def _cross_entropy_loss(self, logits, labels, mask=None):
        if mask is not None:
            logits = jnp.where(mask[..., None], logits, jnp.array([1] + [0] * 22))
        cross_entropy = -(labels * jax.nn.log_softmax(logits)).sum(-1)
        return cross_entropy.mean()

    def _call(
        self, rng_key, model_output: ModelOutput, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        res_logits = model_output.residue_logits

        total_loss, metrics = 0.0, {}

        labels = ground.residue_token
        res_mask = ground.atom_mask[..., 1]

        labels = rearrange(labels, "... -> (...)")
        res_mask = rearrange(res_mask, "... -> (...)")

        res_labels = jax.nn.one_hot(labels, 23)
        res_cross_entropy = self._cross_entropy_loss(
            res_logits, res_labels  # , mask=res_mask
        )
        metrics["res_cross_entropy"] = res_cross_entropy.mean()
        total_loss += res_cross_entropy.mean()

        pred_labels = res_logits.argmax(-1)
        res_accuracy = pred_labels == labels
        res_accuracy = (res_accuracy * res_mask).sum() / (res_mask.sum() + 1e-6)
        res_accuracy = res_accuracy * (res_mask.sum() > 0).astype(res_accuracy.dtype)
        metrics["res_accuracy"] = res_accuracy

        # bound_labels = jax.nn.one_hot(ground.protein_data.boundary_token, 3)
        # sos_labels, eos_labels = bound_labels[..., -2], bound_labels[..., -1]
        # sos_cross_entropy = self._cross_entropy_loss(sos_logits, sos_labels)
        # eos_cross_entropy = self._cross_entropy_loss(eos_logits, eos_labels)
        # boundary_cross_entropy = sos_cross_entropy.mean() + eos_cross_entropy.mean()
        # metrics["boundary_cross_entropy"] = boundary_cross_entropy
        # total_loss += boundary_cross_entropy

        # pred_seq_len = eos_logits.argmax(-1) - sos_logits.argmax(-1)
        # metrics["avg_pred_seq_len"] = pred_seq_len.mean()
        # metrics["avg_gnd_seq_len"] = (labels > 0).sum(-1).mean()

        return model_output, total_loss, metrics


class ChemicalViolationLoss(LossFunction):
    def __init__(
        self,
        weight=1.0,
        start_step=0,
        key=None,
        measure=None,
    ):
        super().__init__(weight=weight, start_step=start_step)
        self.key = key
        self.measure = measure

    def _call(
        self, rng_key, model_output: ModelOutput, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jnp.ndarray, Dict[str, float]]:
        if getattr(self, "key") is None:
            raise ValueError("Must set key before calling ChemicalViolationLoss")
        ground_coords = ground.atom_coord
        coords = model_output.atom_coord

        indices = getattr(ground, f"{self.key}_list")
        mask = getattr(ground, f"{self.key}_mask")

        target = self.measure(ground_coords, indices)
        prediction = self.measure(coords, indices)

        difference = target - prediction
        # if self.key == "dihedrals":
        #     alternative = (2 * jnp.pi - target) - prediction
        #     difference = jnp.where(
        #         jnp.abs(difference) < jnp.abs(alternative), difference, alternative
        #     )

        sqr_error = jnp.square(difference)
        sqr_error = sqr_error * mask.astype(sqr_error.dtype)
        mse = sqr_error.sum((-1, -2)) / (mask.sum((-1, -2)) + 1e-6)
        mse = mse.mean()
        mse = mse * (mask.sum() > 0).astype(mse.dtype)

        return model_output, mse, {f"{self.key}_loss": mse}


class BondLoss(ChemicalViolationLoss):
    def __init__(
        self,
        weight,
        start_step,
    ):
        super().__init__(weight, start_step, "bonds", self.measure_bonds)

    @staticmethod
    def measure_bonds(coords, indices):
        coords = rearrange(coords, "r a c -> (r a) c")
        i, j = rearrange(coords[indices], "... b c -> b ... c")
        norms = safe_norm((i - j))
        return norms


class AngleLoss(ChemicalViolationLoss):
    def __init__(
        self,
        weight,
        start_step,
    ):
        super().__init__(weight, start_step, "angles", self.measure_angles)

    @staticmethod
    def measure_angles(coords, indices):
        coords = rearrange(coords, "r a c -> (r a) c")
        i, j, k = rearrange(indices, "... b -> b ...")

        v1, v2 = coords[i] - coords[j], coords[k] - coords[j]
        v1, v2 = safe_normalize(v1), safe_normalize(v2)
        x, y = safe_norm(v1 + v2), safe_norm(v1 - v2)

        x = jnp.where(x == 0.0, 1e-6, x)
        # NOTE(Allan): this might throw errors still
        # jax._src.source_info_util.JaxStackTraceBeforeTransformation:
        # FloatingPointError: invalid value (nan) encountered in jit(div)
        return 2 * jnp.arctan2(y, x)


class DihedralLoss(ChemicalViolationLoss):
    def __init__(
        self,
        weight,
        start_step,
    ):
        super().__init__(weight, start_step, "dihedrals", self.measure_dihedrals)

    @staticmethod
    def measure_dihedrals(coords, indices):
        # based on Hypnopump's Gist
        coords = rearrange(coords, "r a c -> (r a) c")
        p, q, v, u = rearrange(indices, "... b -> b ...")
        a1 = coords[q] - coords[p]
        a2 = coords[v] - coords[q]
        a3 = coords[u] - coords[v]

        v1 = jnp.cross(a1, a2)
        v1 = safe_normalize(v1)
        v2 = jnp.cross(a2, a3)
        v2 = safe_normalize(v2)

        rad = 2 * jnp.arctan2(safe_norm(v1 - v2), safe_norm(v1 + v2))

        return rad


class ClashLoss(LossFunction):
    def _call(self, rng_key, model_output: ModelOutput, ground: ProteinDatum):
        coords = model_output.atom_coord
        all_atom_coords = rearrange(coords, "r a c -> (r a) c")
        all_atom_radii = rearrange(ground.atom_radius, "r a -> (r a)")
        all_atom_mask = rearrange(ground.atom_mask, "r a -> (r a)")

        vector_map = lambda x: rearrange(x, "i c -> i () c") - rearrange(
            x, "j c -> () j c"
        )

        def mask_bonds(mask, bonds, bonds_mask):
            bonds = jnp.where(bonds_mask[..., None], bonds, coords.shape[-3] * 14)
            i, j = rearrange(bonds, "... b -> b (...)")
            mask = mask.at[i, j].set(False, mode="drop")
            return mask

        distance_maps = safe_norm(vector_map(all_atom_coords))
        cross_radii = rearrange(all_atom_radii, "i -> i ()") + rearrange(
            all_atom_radii, "j -> () j"
        )

        cross_mask = rearrange(all_atom_mask, "i -> i ()") & rearrange(
            all_atom_mask, "j -> () j"
        )
        cross_mask = mask_bonds(cross_mask, ground.bonds_list, ground.bonds_mask)

        cross_radii = jnp.where(cross_radii == 0.0, 1.0, cross_radii)
        clashes = e3nn.soft_envelope(
            distance_maps,
            x_max=cross_radii,
            arg_multiplicator=10.0,
            value_at_origin=1.0,
        )

        mse = (clashes * cross_mask).sum((-1, -2)) / (cross_mask.sum((-1, -2)) + 1e-6)
        mse = mse.mean()
        mse = mse * (cross_mask.sum() > 0).astype(mse.dtype)

        return model_output, mse, {"clash_loss": mse}


def _constraint_satisfaction_term(x, target, lagmul_valid_range=None):
    lag_mul = LagrangeMultiplier(
        shape=x.shape, valid_range=lagmul_valid_range, initializer=1.0
    )()
    constraint_satisfaction = x - target
    return jnp.sum(lag_mul * constraint_satisfaction), lag_mul


class GaussianDivergenceLoss(LossFunction):
    def __init__(
        self,
        weight: float = 1.0,
        start_step: int = 0,
        max_error: float = 1000.0,
        clip: float = 0.0,
        scheduler: Scheduler = None,
        min_rate: float = 0.0,
    ):
        super().__init__(weight=weight, start_step=start_step, scheduler=scheduler)
        self.max_error = max_error
        self.clip = clip
        self.min_rate = min_rate

    def _call(self, rng_key, model_output: ModelOutput, ground: ProteinDatum):
        total_kl = 0.0
        total_lambda = 0.0

        metrics = {}
        counters = defaultdict(int)

        for _, pp in enumerate(model_output.probs):
            mask = pp.mask
            prior, posterior = pp.prior, pp.posterior

            masked_mean = lambda arr: (arr * mask[..., None]).sum((-1, -2)) / (
                mask.sum(-1) + 1e-6
            )

            log_matrix_hat = jnp.einsum(
                "... c i, ... abi-> ... c ab",
                prior.sigma_flat.array,
                prior.sigma_basis.array,
            )
            sigma_hat_inv = jax.vmap(jax.vmap(jax.scipy.linalg.expm))(
                -log_matrix_hat.astype(jnp.float32)
            ).astype(jnp.bfloat16)

            d1 = jnp.trace(posterior.sigma @ sigma_hat_inv, axis1=-1, axis2=-2)
            d1 = masked_mean(d1)

            # NOTE(Allan): the dropping of the vectors there is sus
            diff_mu = e3nn.concatenate(
                [
                    posterior.mu.filter(keep="0e") - prior.mu.filter(keep="0e"),
                    posterior.mu.filter(drop="0e"),
                ]
            ).array
            d2 = (diff_mu * (sigma_hat_inv @ diff_mu[..., None]).squeeze(-1)).sum(-1)
            d2 = masked_mean(d2)

            traces = jnp.einsum("aai->i", posterior.sigma_basis.filter(keep="0e").array)
            traces = rearrange(traces, "... -> () () ...")

            d3 = (
                (
                    prior.sigma_flat.filter(keep="0e").array
                    - posterior.sigma_flat.filter(keep="0e").array
                )
                * traces
            ).sum(-1)
            d3 = masked_mean(d3)

            num_dim = prior.mu.shape[-1] * prior.mu.shape[-2]

            loss_raw = (d1 + d2 + d3 - num_dim) / 2
            loss_clipped = jnp.clip(loss_raw, 0.0, self.max_error)
            kl = jnp.where(jnp.isnan(loss_raw), 0.0, loss_clipped)

            resolution = prior.mu.array.shape[0]
            if self.min_rate > 0.0:
                constraint = pp.lag * (self.min_rate - kl)
                metrics[f"λ [{resolution}] {counters[resolution]}"] = pp.lag
                constraint = jnp.where(jnp.isnan(loss_raw), 0.0, constraint)
                total_lambda += constraint * (kl > 0.0)

            if self.clip > 0.0:
                kl = jnp.where(kl < self.clip, 0.0, kl)

            metrics[f"KL [{resolution}] {counters[resolution]}"] = kl
            counters[resolution] += 1

            total_kl += kl

        total_loss = total_lambda + total_kl

        metrics["ΣKL"] = total_kl
        metrics["Σ(KL + λ)"] = total_loss

        return model_output, total_loss, metrics


class ClassifierLoss(LossFunction):

    def _call(self, rng_key, logits: jnp.ndarray, ground: ProteinDatum):
        num_logits = logits.shape[-1]
        label = jax.nn.one_hot(ground.fold_label, num_logits)
        cross_entropy = -(label * jax.nn.log_softmax(logits)).sum(-1)
        accuracy = (logits.argmax(-1) == ground.fold_label).astype(jnp.float32).mean()
        return (
            logits,
            cross_entropy,
            dict(cross_entropy=cross_entropy.mean(), accuracy=accuracy),
        )


class MultipleBinaryClassifierLoss(LossFunction):

    def _call(self, rng_key, logits: jnp.ndarray, ground: ProteinDatum):
        log_p = jax.nn.log_sigmoid(logits)
        log_not_p = jax.nn.log_sigmoid(-logits)

        labels = ground.fold_label[0]
        bce = -labels * log_p - (1.0 - labels) * log_not_p
        bce = bce.mean()

        # accuracy for binary cross entropy
        accuracy = (logits > 0).astype(jnp.float32) == labels
        labeled_accuracy = ((logits > 0) == labels).astype(jnp.float32) * (
            labels > 1
        ).astype(jnp.float32)
        labeled_accuracy = labeled_accuracy.sum() / (
            (labels > 1).astype(jnp.float32).sum() + 1e-6
        )

        accuracy = accuracy.mean()

        return (
            logits,
            bce,
            dict(bce=bce, accuracy=accuracy, labeled_accuracy=labeled_accuracy),
        )

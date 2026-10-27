"""Cluster bootstrap and paired group randomization, never frame-wise t-tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .artifacts import EvaluationArtifact
from .artifacts import validate_artifact
from .metrics import compute_metrics


def align_candidate(baseline: EvaluationArtifact, candidate: EvaluationArtifact) -> EvaluationArtifact:
    """Align by unique sample IDs; reject changes to the evaluation problem."""
    validate_artifact(baseline)
    validate_artifact(candidate)
    for key in (
        "schema_version",
        "task",
        "task_type",
        "label_names",
        "dataset_fingerprint",
        "split_fingerprint",
        "split",
        "target_key",
        "units",
        "original_class_ids",
    ):
        if baseline.manifest.get(key) != candidate.manifest.get(key):
            raise ValueError(f"Cannot compare different {key}")
    base_protocol, candidate_protocol = baseline.manifest["protocol"], candidate.manifest["protocol"]
    if base_protocol.get("name") != candidate_protocol.get("name") or any(
        set(base_protocol.get(key, [])) != set(candidate_protocol.get(key, []))
        for key in ("train_group_ids", "validation_group_ids", "test_group_ids")
    ):
        raise ValueError("Cannot compare different split protocol")
    if set(baseline.sample_id) != set(candidate.sample_id):
        raise ValueError("Candidates must evaluate identical sample IDs")
    position = {key: index for index, key in enumerate(candidate.sample_id)}
    order = np.array([position[key] for key in baseline.sample_id], dtype=np.int64)
    aligned = replace(
        candidate,
        sample_id=candidate.sample_id[order],
        group_id=candidate.group_id[order],
        y_true=candidate.y_true[order],
        y_pred=candidate.y_pred[order],
    )
    if not np.array_equal(baseline.group_id, aligned.group_id):
        raise ValueError("Episode group IDs differ after sample alignment")
    if not np.array_equal(baseline.y_true, aligned.y_true):
        raise ValueError("Ground-truth labels differ after sample alignment")
    return aligned


class _GroupStatistics:
    def __init__(self, artifact: EvaluationArtifact):
        self.kind = artifact.manifest["task_type"]
        self.groups, inverse = np.unique(artifact.group_id, return_inverse=True)
        size = len(self.groups)
        if self.kind == "regression":
            true, pred = artifact.y_true.astype(np.float64), artifact.y_pred.astype(np.float64)
            centered = true - true.mean(axis=0)
            arrays = (np.ones_like(true), (pred - true) ** 2, np.abs(pred - true), centered, centered**2)
            self.parts = []
            for values in arrays:
                target = np.zeros((size, true.shape[1]), dtype=np.float64)
                np.add.at(target, inverse, values)
                self.parts.append(target)
        else:
            classes = len(artifact.manifest["label_names"])
            pred = artifact.y_pred.argmax(axis=1) if artifact.y_pred.ndim == 2 else artifact.y_pred
            confusion = np.zeros((size, classes * classes), dtype=np.float64)
            np.add.at(confusion, (inverse, artifact.y_true.astype(np.int64) * classes + pred), 1)
            self.parts = [confusion.reshape(size, classes, classes)]

    def aggregate(self, weights: np.ndarray) -> dict[str, np.ndarray]:
        # weights [B,G]; sufficient statistics avoid any [B,frames] allocation.
        parts = [np.tensordot(weights, part, axes=([1], [0])) for part in self.parts]
        if self.kind == "regression":
            count, squared, absolute, centered, centered_sq = parts
            tss = np.maximum(centered_sq - centered**2 / count, 0)
            r2_dims = np.full_like(tss, np.nan)
            np.divide(squared, tss, out=r2_dims, where=tss > 0)
            r2_dims = 1 - r2_dims
            valid = np.isfinite(r2_dims)
            r2 = np.full(len(weights), np.nan)
            np.divide(np.where(valid, r2_dims, 0).sum(axis=1), valid.sum(axis=1), out=r2, where=valid.sum(axis=1) > 0)
            return {
                "rmse": np.sqrt(squared.sum(axis=1) / count.sum(axis=1)),
                "mae": absolute.sum(axis=1) / count.sum(axis=1),
                "r2": r2,
            }
        confusion = parts[0]
        true_count, pred_count = confusion.sum(axis=2), confusion.sum(axis=1)
        correct = np.diagonal(confusion, axis1=1, axis2=2)
        recall = np.divide(correct, true_count, out=np.zeros_like(correct), where=true_count > 0)
        f1 = np.divide(
            2 * correct, true_count + pred_count, out=np.zeros_like(correct), where=true_count + pred_count > 0
        )
        return {
            "accuracy": correct.sum(axis=1) / confusion.sum(axis=(1, 2)),
            "balanced_accuracy": recall.sum(axis=1) / (true_count > 0).sum(axis=1),
            "macro_f1": f1.mean(axis=1),
        }


def _interval(values: np.ndarray, confidence: float) -> list[float] | None:
    # Undefined R² in any resample is reported, rather than silently conditioning
    # on only the resamples with variable ground truth.
    if not np.isfinite(values).all():
        return None
    return np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2]).tolist()


def analyze_run(
    artifact: EvaluationArtifact,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    baseline: EvaluationArtifact | None = None,
    permutation_samples: int = 0,
) -> dict:
    """Frame-weighted metrics with whole episodes resampled as clusters.

    Paired differences are candidate minus baseline (negative improves error
    metrics). Optional two-sided randomization swaps entire episode predictions.
    Confidence intervals describe test-episode uncertainty, not seed uncertainty.
    """
    validate_artifact(artifact)
    if bootstrap_samples < 2 or not 0 < confidence < 1 or permutation_samples < 0:
        raise ValueError("Need bootstrap_samples >= 2, 0 < confidence < 1, permutation_samples >= 0")
    if baseline is not None:
        artifact = align_candidate(baseline, artifact)
    stats = _GroupStatistics(artifact)
    base_stats = _GroupStatistics(baseline) if baseline is not None else None
    point = compute_metrics(
        artifact.manifest["task_type"], artifact.y_true, artifact.y_pred, artifact.manifest["label_names"]
    )
    scalar = [name for name, value in point.items() if not isinstance(value, list)]
    result = {
        "num_examples": len(artifact.sample_id),
        "num_groups": len(stats.groups),
        "metrics": point,
        "confidence": confidence,
        "bootstrap_unit": "source/episode",
        "bootstrap_samples": bootstrap_samples,
        "confidence_intervals": dict.fromkeys(scalar),
    }
    if baseline is not None:
        base_point = compute_metrics(
            baseline.manifest["task_type"], baseline.y_true, baseline.y_pred, baseline.manifest["label_names"]
        )
        result["comparison"] = {
            name: {
                "delta": point[name] - base_point[name]
                if point[name] is not None and base_point[name] is not None
                else None,
                "delta_ci": None,
                "pvalue": None,
            }
            for name in scalar
        }
    if len(stats.groups) < 2:
        result["uncertainty_warning"] = (
            "At least two independent episodes are needed; no confidence interval or p-value reported."
        )
        return result
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(bootstrap_samples) for name in scalar}
    delta_draws = {name: np.empty(bootstrap_samples) for name in scalar} if base_stats is not None else {}
    # Bound the bootstrap weight matrix to about 8 MB even for many episodes.
    batch_size = max(1, min(128, 1_000_000 // len(stats.groups)))
    probabilities = np.full(len(stats.groups), 1 / len(stats.groups))
    for start in range(0, bootstrap_samples, batch_size):
        stop = min(start + batch_size, bootstrap_samples)
        weights = rng.multinomial(len(stats.groups), probabilities, size=stop - start)
        values = stats.aggregate(weights)
        base_values = base_stats.aggregate(weights) if base_stats is not None else None
        for name in scalar:
            draws[name][start:stop] = values[name]
            if base_values is not None:
                delta_draws[name][start:stop] = values[name] - base_values[name]
    result["confidence_intervals"] = {name: _interval(values, confidence) for name, values in draws.items()}
    if base_stats is not None:
        for name in scalar:
            result["comparison"][name]["delta_ci"] = _interval(delta_draws[name], confidence)
        # Group swaps preserve pairing and correlated cameras/frames. Compute
        # metric deltas after swapping aggregate contributions, even for RMSE.
        extremes = dict.fromkeys(scalar, 0)
        invalid = dict.fromkeys(scalar, False)
        total = stats.aggregate(np.ones((1, len(stats.groups))))
        base_total = base_stats.aggregate(np.ones((1, len(stats.groups))))
        for start in range(0, permutation_samples, batch_size):
            count = min(batch_size, permutation_samples - start)
            swap = rng.integers(0, 2, size=(count, len(stats.groups)))
            swapped = _mixed_scores(stats, base_stats, swap)
            inverse = _mixed_scores(base_stats, stats, swap)
            for name in scalar:
                delta = swapped[name] - inverse[name]
                observed = total[name][0] - base_total[name][0]
                invalid[name] |= not np.isfinite(delta).all() or not np.isfinite(observed)
                extremes[name] += int(np.count_nonzero(np.abs(delta) >= abs(observed) - 1e-12))
        if permutation_samples:
            result["permutation_samples"] = permutation_samples
            result["pvalue_note"] = (
                "Unadjusted two-sided group randomization p-values; multiple comparisons require correction."
            )
            for name in scalar:
                result["comparison"][name]["pvalue"] = (
                    None if invalid[name] else (extremes[name] + 1) / (permutation_samples + 1)
                )
    return result


def _mixed_scores(first: _GroupStatistics, second: _GroupStatistics, swap: np.ndarray) -> dict[str, np.ndarray]:
    # Constructing mixed group arrays for every draw is unnecessary: concatenate
    # both sets of group contributions and apply complementary 0/1 weights.
    combined = object.__new__(_GroupStatistics)
    combined.kind = first.kind
    combined.parts = [np.concatenate((a, b), axis=0) for a, b in zip(first.parts, second.parts, strict=True)]
    return combined.aggregate(np.concatenate((1 - swap, swap), axis=1))

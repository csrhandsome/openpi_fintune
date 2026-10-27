"""Validate held-out identities, episode resampling, and independent seed reports."""

from dataclasses import replace
import json

import numpy as np
import pytest

from force_encoder.evaluation.artifacts import load_evaluation_artifact
from force_encoder.evaluation.artifacts import save_evaluation_artifact
from force_encoder.evaluation.metrics import compute_metrics
from force_encoder.evaluation.report import build_report
from force_encoder.evaluation.statistics import align_candidate
from force_encoder.evaluation.statistics import analyze_run


def artifact(tmp_path, name="run", *, pred=None, order=None, task="regression", groups=None, seed=1, extra=None):
    ids = np.array(["s/0/c/0", "s/0/c/1", "s/1/c/0", "s/1/c/1"])
    groups = np.array(groups if groups is not None else ["s/0", "s/0", "s/1", "s/1"])
    labels = ["x", "y"]
    true = (
        np.array([[0.0, 7.0], [1.0, 7.0], [2.0, 7.0], [3.0, 7.0]]) if task == "regression" else np.array([0, 1, 0, 1])
    )
    pred = true.copy() if pred is None else np.asarray(pred)
    if order is not None:
        ids, groups, true, pred = ids[order], groups[order], true[order], pred[order]
    save_evaluation_artifact(
        tmp_path / name,
        task=task,
        sample_id=ids,
        group_id=groups,
        y_true=true,
        y_pred=pred,
        label_names=labels,
        dataset_fingerprint="immutable-dataset-v1",
        seed=seed,
        protocol={
            "name": "held-out-episode-v1",
            "train_group_ids": ["s/100"],
            "validation_group_ids": ["s/99"],
            "test_group_ids": sorted(set(groups)),
        },
        extra_manifest=extra,
    )
    return load_evaluation_artifact(tmp_path / name)


def test_regression_metrics_constant_dimensions_and_json():
    true = np.array([[0.0, 5.0], [2.0, 5.0]])
    result = compute_metrics("regression", true, true + np.array([1.0, 2.0]))
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["mae"] == 1.5
    assert result["r2_per_dim"] == [0.0, None]
    assert result["r2"] == 0.0
    json.dumps(result, allow_nan=False)


def test_classification_metrics_declared_vocabulary():
    result = compute_metrics("classification", np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), ["a", "b"])
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    scores = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert compute_metrics("classification", np.array([0, 1]), scores, ["a", "b"])["accuracy"] == 1.0


def test_atomic_artifact_roundtrip_and_checksum(tmp_path):
    saved = artifact(tmp_path)
    assert saved.manifest["num_groups"] == 2
    assert saved.sample_id.dtype.kind == "U"
    saved.path.write_bytes(saved.path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_evaluation_artifact(tmp_path / "run")


def test_align_reorders_and_rejects_label_or_protocol_mismatch(tmp_path):
    baseline = artifact(tmp_path)
    candidate = artifact(tmp_path, "shuffled", order=np.array([3, 1, 0, 2]))
    aligned = align_candidate(baseline, candidate)
    np.testing.assert_array_equal(aligned.sample_id, baseline.sample_id)
    bad = replace(aligned, y_true=aligned.y_true + 1)
    with pytest.raises(ValueError, match="Ground-truth"):
        align_candidate(baseline, bad)
    for field, value in [("label_names", ["a", "b"]), ("dataset_fingerprint", "different")]:
        bad = replace(candidate, manifest={**candidate.manifest, field: value})
        with pytest.raises(ValueError, match=field):
            align_candidate(baseline, bad)


def test_reject_duplicate_samples_and_leakage(tmp_path):
    saved = artifact(tmp_path)
    from force_encoder.evaluation.artifacts import validate_artifact

    with pytest.raises(ValueError, match="unique"):
        validate_artifact(replace(saved, sample_id=np.array(["x", "x", "y", "z"])))
    protocol = {**saved.manifest["protocol"], "train_group_ids": ["s/0"]}
    with pytest.raises(ValueError, match="overlap"):
        validate_artifact(replace(saved, manifest={**saved.manifest, "protocol": protocol}))
    with pytest.raises(ValueError, match="finite"):
        artifact(tmp_path, "invalid", pred=np.full((4, 2), np.nan))


def test_cluster_bootstrap_keeps_episode_errors_correlated(tmp_path):
    true = np.array([[0.0, 7.0], [1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    pred = true + np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
    run = artifact(tmp_path, pred=pred)
    report = analyze_run(run, bootstrap_samples=1000, seed=3)
    assert report["confidence_intervals"]["rmse"] == [0.0, 10.0]
    assert report == analyze_run(run, bootstrap_samples=1000, seed=3)
    assert report["bootstrap_unit"] == "source/episode"
    json.dumps(report, allow_nan=False)


def test_paired_delta_and_single_group_no_significance(tmp_path):
    true = np.array([[0.0, 7.0], [1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    baseline = artifact(tmp_path, "baseline", pred=true + 2)
    candidate = artifact(tmp_path, "candidate")
    report = analyze_run(candidate, baseline=baseline, bootstrap_samples=100, permutation_samples=100, seed=4)
    assert report["comparison"]["rmse"]["delta"] == -2.0
    assert report["comparison"]["rmse"]["delta_ci"] == [-2.0, -2.0]
    assert 0 < report["comparison"]["rmse"]["pvalue"] <= 1
    one = artifact(tmp_path, "one", groups=["s/0"] * 4)
    report = analyze_run(one, bootstrap_samples=100, permutation_samples=100)
    assert all(value is None for value in report["confidence_intervals"].values())
    assert "uncertainty_warning" in report


def test_report_summarizes_seeds_not_frames_and_omits_unpaired_seed(tmp_path):
    artifact(tmp_path, "seed1", seed=1, extra={"run_family": "jepa_linear"})
    artifact(tmp_path, "seed2", seed=2, extra={"run_family": "jepa_linear"})
    artifact(tmp_path, "baseline", seed=1, extra={"run_family": "random_linear"})
    report = build_report(
        [tmp_path / "seed1", tmp_path / "seed2"], baseline=tmp_path / "baseline", bootstrap_samples=10
    )
    assert report["seed_summaries"][0]["num_runs"] == 2
    assert report["seed_summaries"][0]["metrics"]["rmse"] == {"mean": 0.0, "std_across_seeds": 0.0}
    assert "comparison" in report["runs"][0]
    assert "comparison" not in report["runs"][1]
    json.dumps(report, allow_nan=False)


def test_report_holm_adjustment_and_classification_bootstrap(tmp_path):
    artifact(tmp_path, "candidate", task="classification", extra={"target_key": "object"})
    artifact(tmp_path, "baseline", task="classification", pred=np.array([1, 0, 1, 0]), extra={"target_key": "object"})
    report = build_report(
        [tmp_path / "candidate"], baseline=tmp_path / "baseline", bootstrap_samples=20, permutation_samples=50
    )
    comparisons = report["runs"][0]["comparison"]
    for comparison in comparisons.values():
        assert comparison["pvalue"] <= comparison["pvalue_holm"] <= 1
    assert report["runs"][0]["confidence_intervals"]["accuracy"] == [1.0, 1.0]
    json.dumps(report, allow_nan=False)


def test_reject_different_target_semantics(tmp_path):
    baseline = artifact(tmp_path, "baseline", extra={"target_key": "force", "units": ["N", "N"]})
    candidate = artifact(tmp_path, "candidate", extra={"target_key": "force", "units": ["mN", "mN"]})
    with pytest.raises(ValueError, match="units"):
        align_candidate(baseline, candidate)
    candidate = artifact(tmp_path, "pose", extra={"target_key": "pose", "units": ["N", "N"]})
    with pytest.raises(ValueError, match="target_key"):
        align_candidate(baseline, candidate)

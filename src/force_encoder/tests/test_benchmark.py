"""Real MP4/parquet downstream runs and split/normalization checks."""

import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from force_encoder import benchmark
from force_encoder.benchmark import FrameLabelReader
from force_encoder.benchmark import load_head
from force_encoder.benchmark import run
from force_encoder.benchmark import selected_indices
from force_encoder.benchmark import validate_pretraining_split
from force_encoder.evaluation import load_evaluation_artifact
from force_encoder.model import ImageJEPA
from force_encoder.tests.test_data import v3_root  # noqa: F401


@pytest.fixture
def benchmark_config(v3_root, tmp_path):  # noqa: F811
    path = v3_root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.append_column("frame_index", pa.array(np.tile(np.arange(3), 4)))
    table = table.append_column("observation.gripper_position", pa.array(np.arange(12, dtype=np.float32) / 12))
    table = table.append_column("observation.ee_pose", pa.array([[float(index), -float(index)] for index in range(12)]))
    table = table.append_column("task_index", pa.array(np.tile([10, 20, 10], 4)))
    pq.write_table(table, path, row_group_size=4)
    torch.manual_seed(42)
    model = ImageJEPA(
        image_size=16, patch_size=4, embed_dim=16, depth=1, num_heads=4, predictor_dim=16, predictor_depth=1
    )
    sources = [{"name": "fixture", "root": str(v3_root)}]
    data = {"sources": sources, "val_fraction": 0.25, "test_fraction": 0.25, "frame_stride": 1}
    checkpoint = tmp_path / "jepa.pt"
    torch.save(
        {"model": model.state_dict(), "model_config": model.model_config, "config": {"seed": 42, "data": data}},
        checkpoint,
    )
    return {
        "seed": 42,
        "checkpoint": str(checkpoint),
        "output_dir": str(tmp_path / "probe"),
        "data": data,
        "task": {
            "type": "regression",
            "target_key": "observation.gripper_position",
            "label_names": ["gripper_position"],
            "units": ["ratio"],
        },
        "encoder": {"baseline": "pretrained", "random_seed": 42},
        "head": {"type": "linear"},
        "training": {
            "device": "cpu",
            "batch_size": 4,
            "feature_batch_size": 4,
            "num_workers": 0,
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 0,
            "seeds": [3],
            "max_samples_per_split": None,
        },
    }


def test_real_downstream_run_preserves_units_and_selects_only_on_validation(benchmark_config, monkeypatch):
    predict = benchmark._predict  # noqa: SLF001
    calls = []

    def test_predict(*args, **kwargs):
        path = benchmark._project_path(benchmark_config["output_dir"]) / "seed_3/best_head.pt"  # noqa: SLF001
        assert path.is_file()
        calls.append(path)
        return predict(*args, **kwargs)

    monkeypatch.setattr(benchmark, "_predict", test_predict)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        artifact_paths = run(benchmark_config)
    finally:
        torch.set_num_threads(previous_threads)
    assert len(artifact_paths) == 1
    assert len(calls) == 1
    artifact = load_evaluation_artifact(artifact_paths[0].parent)
    assert artifact.y_true.shape == artifact.y_pred.shape == (6, 1)
    assert artifact.manifest["units"] == ["ratio"]
    protocol = artifact.manifest["protocol"]
    assert set(protocol["train_group_ids"]).isdisjoint(protocol["test_group_ids"])
    assert set(protocol["validation_group_ids"]).isdisjoint(protocol["test_group_ids"])
    output = artifact_paths[0].parents[2]
    head, payload = load_head(output / "seed_3/best_head.pt")
    raw_train = np.load(output / "features/train/targets.npy")
    np.testing.assert_allclose(payload["target_mean"].numpy(), raw_train.mean(axis=0))
    np.testing.assert_allclose(payload["target_scale"].numpy(), raw_train.std(axis=0))
    features = torch.from_numpy(np.load(output / "features/test/features.npy"))
    with torch.no_grad():
        predictions = head(features) * payload["target_scale"] + payload["target_mean"]
    np.testing.assert_allclose(predictions.numpy(), artifact.y_pred, atol=1e-6)
    assert payload["epoch"] in (1, 2)
    assert len((output / "seed_3/training.jsonl").read_text().splitlines()) == 2
    assert "rmse" in json.loads((output / "seed_3/test_metrics.json").read_text())
    assert artifact.manifest["original_class_ids"] == []


def test_classification_remaps_real_noncontiguous_task_indices(benchmark_config):
    config = copy.deepcopy(benchmark_config)
    config["task"] = {"type": "classification", "target_key": "task_index"}
    config["encoder"]["baseline"] = "frozen_random"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        paths = run(config)
    finally:
        torch.set_num_threads(previous_threads)
    artifact = load_evaluation_artifact(paths[0].parent)
    assert set(artifact.y_true) == {0, 1}
    assert set(artifact.y_pred) <= {0, 1}
    assert artifact.manifest["original_class_ids"] == [10, 20]
    assert artifact.manifest["label_names"] == ["10", "20"]
    assert artifact.manifest["run_family"] == "frozen_random_linear"


def test_source_local_task_index_cannot_be_mixed_across_datasets(benchmark_config):
    config = copy.deepcopy(benchmark_config)
    config["task"] = {"type": "classification", "target_key": "task_index"}
    config["data"]["sources"].append({"name": "another", "root": "/unused"})
    with pytest.raises(ValueError, match="source-local vocabularies"):
        run(config)


def test_constant_regression_labels_are_rejected_but_partial_constant_is_allowed(tmp_path):
    np.save(tmp_path / "targets.npy", np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="placeholder labels"):
        benchmark._target_statistics(tmp_path, "regression")  # noqa: SLF001
    np.save(tmp_path / "targets.npy", np.asarray([[0, 1], [0, 2], [0, 3]], dtype=np.float32))
    mean, scale, classes = benchmark._target_statistics(tmp_path, "regression")  # noqa: SLF001
    np.testing.assert_allclose(mean, [0, 2])
    assert scale[0] == pytest.approx(1e-6)
    assert scale[1] > 0
    assert classes == []


def test_real_frame_labels_are_aligned_by_frame_index(benchmark_config, v3_root):  # noqa: F811
    reader = FrameLabelReader("observation.ee_pose", cache_size=1)
    metadata = {"data_path": str(v3_root / "data/chunk-000/file-000.parquet"), "episode_index": 2, "frame_index": 1}
    np.testing.assert_array_equal(reader(metadata), [7, -7])
    metadata["episode_index"], metadata["frame_index"] = 0, 2
    np.testing.assert_array_equal(reader(metadata), [2, -2])
    reader.close()
    missing = FrameLabelReader("observation.force")
    with pytest.raises(ValueError, match="labels are never fabricated"):
        missing(metadata)


def test_probe_rejects_checkpoint_without_unseen_test_episodes(benchmark_config):
    checkpoint = torch.load(benchmark_config["checkpoint"], weights_only=True)
    checkpoint["config"]["data"].pop("test_fraction")
    with pytest.raises(ValueError, match="Old checkpoints"):
        validate_pretraining_split(checkpoint, benchmark_config)
    checkpoint["config"]["data"]["test_fraction"] = 0.25
    checkpoint["config"]["seed"] = 999
    with pytest.raises(ValueError, match="split seed"):
        validate_pretraining_split(checkpoint, benchmark_config)
    checkpoint["config"]["seed"] = 42
    checkpoint["config"]["data"]["val_fraction"] = 0.1
    with pytest.raises(ValueError, match="val_fraction"):
        validate_pretraining_split(checkpoint, benchmark_config)


def test_debug_subset_covers_index_and_records_identities(benchmark_config):
    np.testing.assert_array_equal(selected_indices(20, 3), [0, 9, 19])
    with pytest.raises(ValueError, match="positive"):
        selected_indices(20, 0)
    config = copy.deepcopy(benchmark_config)
    config["training"]["max_samples_per_split"] = 3
    config["training"]["epochs"] = 1
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        paths = run(config)
    finally:
        torch.set_num_threads(previous_threads)
    root = paths[0].parents[2]
    manifest = json.loads((root / "features/test/manifest.json").read_text())
    assert manifest["samples"] == manifest["sample_cap"] == 3
    assert manifest["available_samples"] == 6
    assert len((root / "features/test/samples.jsonl").read_text().splitlines()) == 3

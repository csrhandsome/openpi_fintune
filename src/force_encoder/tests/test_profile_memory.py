"""CPU checks for sweep isolation, mask stress selection and recommendation."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from force_encoder import profile_memory as profile
from force_encoder.model import ImageJEPA


def test_profile_override_keeps_training_and_mask_recipe():
    base = {
        "seed": 42,
        "model": {"embed_dim": 192, "depth": 12, "local_scale": [0.05, 0.1]},
        "training": {"batch_size": 64, "precision": "bf16", "learning_rate": 0.0003, "max_steps": 100000},
        "data": {"sources": [{"root": "data/force"}]},
    }
    original = copy.deepcopy(base)
    config = profile.profile_config(base, {"batch_size": 64}, {"model": {"embed_dim": 768, "depth": 12}})
    assert base == original
    assert config["model"]["embed_dim"] == 768
    assert config["model"]["local_scale"] == original["model"]["local_scale"]
    assert config["training"] == original["training"]
    assert config["data"] == original["data"]


def test_recommendation_excludes_oom_and_preserves_headroom(tmp_path):
    def result(name, parameters, memory, status="ok"):
        return {
            "profile": name,
            "parameters": {"encoder": parameters},
            "status": status,
            "device_total_gib": 48,
            "estimated_peak_device_used_gib": memory,
            "peak_allocated_gib": memory - 1,
            "peak_reserved_gib": memory,
        }

    results = [
        result("small", 22_000_000, 8),
        result("large", 300_000_000, 40),
        result("huge", 600_000_000, 45),
        result("giant", 1_000_000_000, 47, "oom"),
    ]
    assert profile.recommend(results, 0.9) == "large"
    assert profile.recommend(results, 0.99) == "huge"
    assert profile.recommend(results, 0.1) is None
    sweep = {"batch_size": 64, "steps": 8, "target_utilization": 0.9}
    profile.write_summary(results, sweep, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["recommended_profile"] == "large"
    assert summary["results"][-1]["status"] == "oom"
    assert "| giant | oom |" in (tmp_path / "summary.md").read_text()


def test_virtual_overcommit_is_not_a_healthy_fit_even_when_steps_complete(tmp_path):
    result = {
        "profile": "giant48",
        "status": "ok",
        "completed_steps": 8,
        "parameters": {"encoder": 1_300_000_000},
        "device_total_gib": 47.99,
        "peak_allocated_gib": 45.8,
        "peak_reserved_gib": 53.13,
        "estimated_peak_device_used_gib": 54.69,
        "baseline": {"sampled_device_used_gib": 1.56, "reserved_gib": 0},
        "stage_peaks": [{"sampled_device_used_gib": 47.99}],
        "steps": [{"warmed": False, "seconds": 30}, {"warmed": True, "seconds": 11}, {"warmed": True, "seconds": 17}],
    }
    annotated = profile.annotate_result(result)
    assert annotated["status"] == "ok"
    assert annotated["memory_suitability"] == "overcommit_or_paging_suspected"
    assert annotated["virtual_allocator_exceeds_physical"]
    assert annotated["sampled_peak_device_used_gib"] == 47.99
    assert annotated["virtual_reserved_plus_baseline_gib"] == pytest.approx(54.69)
    assert annotated["warmed_mean_step_seconds"] == 14
    assert "memory_suitability" not in result
    assert profile.recommend([result], 0.9) is None
    profile.write_summary([result], {"batch_size": 64, "steps": 8, "target_utilization": 0.9}, tmp_path)
    saved = json.loads((tmp_path / "giant48.json").read_text())
    assert saved["memory_suitability"] == "overcommit_or_paging_suspected"
    assert saved["peak_reserved_gib"] == result["peak_reserved_gib"]
    assert "Sampled device GiB" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize(
    ("allocated", "reserved", "physical", "estimate", "expected"),
    [
        (49.5, 40, 47.9, 47.9, "overcommit_or_paging_suspected"),
        (39, 44.77, 46.4, 46.4, "high_pressure"),
        (35, 40, 42, 42, "within_budget"),
        (38, 43, 44, 44, "above_budget"),
    ],
)
def test_memory_pressure_uses_physical_capacity_and_device_observations(
    allocated, reserved, physical, estimate, expected
):
    result = {
        "status": "ok",
        "device_total_gib": 48,
        "peak_allocated_gib": allocated,
        "peak_reserved_gib": reserved,
        "stage_peaks": [{"sampled_device_used_gib": physical}],
        "estimated_peak_device_used_gib": estimate,
    }
    assert profile.annotate_result(result)["memory_suitability"] == expected


def test_mask_extrema_reproduce_actual_forward_masks():
    torch.set_num_threads(1)
    model = ImageJEPA(
        image_size=32,
        patch_size=8,
        embed_dim=16,
        depth=1,
        num_heads=2,
        predictor_dim=16,
        predictor_depth=1,
        local_scale=(0.05, 0.1),
        global_scale=(0.15, 0.2),
    )
    extremes = profile.mask_extremes(model.model_config, seed=7, candidates=16)
    counts = [len(model.sample_masks(torch.Generator().manual_seed(seed)).context) for seed in range(7, 23)]
    assert extremes["min_context_tokens"] == min(counts)
    assert extremes["max_context_tokens"] == max(counts)
    images = torch.randn(2, 3, 32, 32)
    for bound in ("min", "max"):
        metrics = model(images, generator=torch.Generator().manual_seed(extremes[f"{bound}_context_seed"]))
        assert int(metrics["context_tokens"]) == extremes[f"{bound}_context_tokens"]
        assert torch.isfinite(metrics["loss"])


def test_prepare_batches_decodes_exactly_one_full_batch_per_split(tmp_path, monkeypatch):
    datasets = []

    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.source_ranges = [("one", 0, 32), ("two", 32, 64)]
            self.summary = {"frames": 64}
            self.decoded = []
            self.closed = False

        def __len__(self):
            return 64

        def __getitem__(self, index):
            self.decoded.append(index)
            return torch.full((3, 4, 4), float(index))

        def sample_metadata(self, index):
            return {
                "source": "one" if index < 32 else "two",
                "episode_index": index,
                "video_key": "gripper",
                "frame_index": 0,
                "video_path": "private/video.mp4",
            }

        def close(self):
            self.closed = True

    def make_dataset(*_args):
        dataset = Dataset()
        datasets.append(dataset)
        return dataset

    monkeypatch.setattr(profile, "make_dataset", make_dataset)
    base = {"seed": 42, "training": {"num_workers": 0}, "model": {"image_size": 4}}
    path = profile.prepare_batches(base, {"batch_size": 64}, tmp_path)
    cached = torch.load(path, weights_only=True)
    assert set(cached) == {"train", "val"}
    assert all(tuple(batch.shape) == (64, 3, 4, 4) for batch in cached.values())
    assert all(len(dataset.decoded) == 64 and dataset.closed for dataset in datasets)
    metadata = json.loads((tmp_path / "batch_metadata.json").read_text())
    assert len(metadata["train"]["samples"]) == 64
    assert all("video_path" not in record for record in metadata["train"]["samples"])
    assert {record["source"] for record in metadata["train"]["samples"]} == {"one", "two"}
    assert profile.validate_cached_batches(base, {"batch_size": 64}, tmp_path) == path
    assert all(not dataset.decoded and dataset.closed for dataset in datasets[2:])
    with pytest.raises(ValueError, match="data IDs/configuration changed"):
        profile.validate_cached_batches({**base, "seed": 43}, {"batch_size": 64}, tmp_path)
    with pytest.raises(ValueError, match="batch shape"):
        profile.validate_cached_batches(base, {"batch_size": 32}, tmp_path)


def test_append_keeps_previous_profiles_and_replaces_rerun(tmp_path, monkeypatch):
    old_result = {
        "profile": "tiny",
        "status": "ok",
        "parameters": {"encoder": 5},
        "device_total_gib": 48,
        "estimated_peak_device_used_gib": 2,
    }
    (tmp_path / "summary.json").write_text(json.dumps({"batch_size": 64, "steps": 8, "results": [old_result]}))
    sweep = {
        "batch_size": 64,
        "steps": 8,
        "warmup_steps": 2,
        "target_utilization": 0.9,
        "base_config": "unused.yaml",
        "output_dir": str(tmp_path),
        "profiles": [{"name": "large", "model": {}}],
    }
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(yaml.safe_dump(sweep))
    cache = tmp_path / "batches.pt"
    cache.write_bytes(b"existing real batches")
    monkeypatch.setattr(profile, "read_config", lambda _path: {})
    monkeypatch.setattr(profile, "validate_cached_batches", lambda *_args: cache)

    def no_new_batches(*_args):
        pytest.fail("Append must not decode replacement batches")

    monkeypatch.setattr(profile, "prepare_batches", no_new_batches)

    def run_worker(command, **_kwargs):
        assert command[command.index("--batch-cache") + 1] == str(cache)
        new_result = {
            **old_result,
            "profile": "large",
            "parameters": {"encoder": 300},
            "estimated_peak_device_used_gib": 35,
        }
        (tmp_path / "large.json").write_text(json.dumps(new_result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(profile.subprocess, "run", run_worker)
    monkeypatch.setattr(sys, "argv", ["profile", "--config", str(config_path), "--profiles", "large", "--append"])
    profile.main()
    profile.main()
    results = json.loads((tmp_path / "summary.json").read_text())["results"]
    assert [result["profile"] for result in results] == ["tiny", "large"]
    assert cache.read_bytes() == b"existing real batches"


@pytest.mark.parametrize(
    "override",
    [
        {"batch_size": 0},
        {"warmup_steps": -1},
        {"steps": 2},
        {"target_utilization": 1.1},
        {"profiles": [{"name": "../unsafe"}]},
        {"profiles": [{"name": "same"}, {"name": "same"}]},
    ],
)
def test_invalid_sweep_rejected_before_loading_data(tmp_path, override):
    sweep = {
        "batch_size": 64,
        "steps": 8,
        "warmup_steps": 2,
        "target_utilization": 0.9,
        "profiles": [{"name": "tiny", "model": {}}],
        **override,
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(sweep))
    with pytest.raises(ValueError, match="Require|target_utilization|Profile names"):
        profile.read_sweep(path)

"""Training checks using real JEPA optimization and small deterministic images."""

import copy
import json
import math
from pathlib import Path
import random

import pytest
import torch
from torch.utils.data import Dataset

from force_encoder import train
from force_encoder.model import ImageJEPA
from force_encoder.model import load_encoder


class SyntheticImages(Dataset):
    def __init__(self):
        self.source_ranges = [("small", 0, 4), ("large", 4, 12)]
        self.summary = {"synthetic": True, "images": 12}

    def __len__(self):
        return 12

    def __getitem__(self, index):
        return torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(int(index)))


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "make_dataset", lambda _config, _split: SyntheticImages())
    return {
        "seed": 7,
        "output_dir": str(tmp_path / "training"),
        "data": {"sources": [{"name": "synthetic", "root": "unused"}]},
        "model": {
            "image_size": 32,
            "patch_size": 8,
            "embed_dim": 24,
            "depth": 1,
            "num_heads": 3,
            "predictor_depth": 1,
            "predictor_dim": 24,
        },
        "training": {
            "device": "cpu",
            "precision": "fp32",
            "batch_size": 2,
            "num_workers": 0,
            "steps_per_epoch": 3,
            "max_steps": 4,
            "learning_rate": 0.001,
            "min_learning_rate": 0.00001,
            "warmup_steps": 1,
            "weight_decay": 0.04,
            "final_weight_decay": 0.1,
            "ema_start": 0.9,
            "ema_end": 0.99,
            "grad_clip": 1.0,
            "log_every": 1,
            "validate_every": 2,
            "val_batches": 1,
            "save_every": 1,
        },
    }


def test_schedule_warmup_and_cosine_endpoints(config):
    cfg = {**config["training"], "max_steps": 6, "warmup_steps": 2}
    first = train.schedule(0, cfg)
    warmup_end = train.schedule(1, cfg)
    decay_start = train.schedule(2, cfg)
    final = train.schedule(5, cfg)
    assert first == pytest.approx((cfg["learning_rate"] / 2, cfg["weight_decay"], cfg["ema_start"]))
    assert warmup_end[0] == pytest.approx(cfg["learning_rate"])
    assert decay_start[0] == pytest.approx(cfg["learning_rate"])
    assert final == pytest.approx((cfg["min_learning_rate"], cfg["final_weight_decay"], cfg["ema_end"]))
    rates = [train.schedule(step, cfg)[0] for step in range(2, 6)]
    assert rates == sorted(rates, reverse=True)


def test_checkpoint_loads_weights_only_with_optimizer_and_python_rng(config, tmp_path):
    model = ImageJEPA(**config["model"])
    optimizer = torch.optim.AdamW(parameter for parameter in model.parameters() if parameter.requires_grad)
    model(SyntheticImages()[0].unsqueeze(0))["loss"].backward()
    optimizer.step()
    expected_python_rng = random.getstate()
    checkpoint_path = tmp_path / "checkpoint.pt"
    train.save_checkpoint(checkpoint_path, model, optimizer, 1, config, 0.5)
    payload = torch.load(checkpoint_path, weights_only=True)
    assert payload["step"] == 1
    assert payload["best_loss"] == 0.5
    assert payload["python_rng"] == expected_python_rng
    assert payload["optimizer"]["state"]
    restored_optimizer = torch.optim.AdamW(parameter for parameter in model.parameters() if parameter.requires_grad)
    restored_optimizer.load_state_dict(payload["optimizer"])
    assert all(state["step"].item() == 1 for state in restored_optimizer.state.values())
    assert torch.equal(
        load_encoder(checkpoint_path)(SyntheticImages()[0].unsqueeze(0)),
        model.encode(SyntheticImages()[0].unsqueeze(0)),
    )
    assert not checkpoint_path.with_suffix(".tmp").exists()


def test_cpu_training_and_interrupted_resume_match(config, tmp_path, monkeypatch):
    reference = copy.deepcopy(config)
    reference["output_dir"] = str(tmp_path / "reference")
    train.run(reference)
    reference_checkpoint = torch.load(tmp_path / "reference/latest.pt", weights_only=True)

    original_save = train.save_checkpoint

    def interrupt_after_second_step(path, model, optimizer, step, config, best_loss):
        original_save(path, model, optimizer, step, config, best_loss)
        if path.name == "latest.pt" and step == 2:
            raise InterruptedError("Simulate a process interruption after the checkpoint is durable")

    monkeypatch.setattr(train, "save_checkpoint", interrupt_after_second_step)
    with pytest.raises(InterruptedError, match="Simulate"):
        train.run(config)
    monkeypatch.setattr(train, "save_checkpoint", original_save)
    checkpoint_path = tmp_path / "training/latest.pt"
    assert torch.load(checkpoint_path, weights_only=True)["step"] == 2
    train.run(config, resume=str(checkpoint_path))
    resumed = torch.load(checkpoint_path, weights_only=True)
    assert resumed["step"] == 4
    for name, tensor in reference_checkpoint["model"].items():
        assert torch.equal(tensor, resumed["model"][name]), name
    for parameter, state in reference_checkpoint["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, resumed["optimizer"]["state"][parameter][name])
    assert (tmp_path / "training/best.pt").is_file()
    records = [json.loads(line) for line in (tmp_path / "training/metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records if "loss" in record] == [1, 2, 3, 4]
    assert all(math.isfinite(record.get("loss", record.get("val/loss"))) for record in records)


def test_progress_bar_and_tracker_receive_the_same_records(config, monkeypatch):
    class Tracker:
        def __init__(self):
            self.logs = []
            self.finished = 0

        def log(self, record, step):
            self.logs.append((record, step))

        def finish(self):
            self.finished += 1

    tracker = Tracker()
    opened = []
    bars = []

    class Bar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = 0
            self.postfix = None
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, count=1):
            self.updates += count

        def set_postfix(self, ordered_dict=None, **kwargs):
            self.postfix = ordered_dict if ordered_dict is not None else kwargs

        def write(self, _text):
            return None

    def fake_init(cfg, directory, *, resuming):
        opened.append((cfg["tracking_backend"], directory, resuming))
        return tracker

    monkeypatch.setattr(train, "init_tracking", fake_init)
    monkeypatch.setattr(train.tqdm, "tqdm", lambda **kwargs: Bar(**kwargs))
    config.update(tracking_backend="swanlab", name="force_encoder", exp_name="jepa", project_name="openpi")
    train.run(config)
    output = config["output_dir"]
    records = [json.loads(line) for line in (Path(output) / "metrics.jsonl").read_text().splitlines()]
    assert opened == [("swanlab", Path(output), False)]
    assert bars[0].kwargs == {"total": 4, "initial": 0, "desc": "Training"}
    assert bars[0].updates == 4
    assert set(bars[0].postfix) == {"loss", "lr", "step"}
    assert bars[0].postfix["step"] == 4
    assert [(record, record["step"]) for record in records] == tracker.logs
    assert [record["step"] for record in records if "loss" in record] == [1, 2, 3, 4]
    assert [record["step"] for record in records if "val/loss" in record] == [2, 4]
    assert tracker.finished == 1

    train.run(config, resume=str(Path(output) / "latest.pt"))
    assert opened[1] == ("swanlab", Path(output), True)
    assert bars[1].kwargs["initial"] == 4
    assert bars[1].updates == 0
    assert tracker.finished == 2


def test_unknown_tracking_backend_is_rejected(config):
    config["tracking_backend"] = "other"
    with pytest.raises(ValueError, match="tracking_backend must be"):
        train.run(config)


def test_validation_reuses_masks_and_restores_training_mode(config):
    model = ImageJEPA(**config["model"])
    loader = torch.utils.data.DataLoader(SyntheticImages(), batch_size=2, generator=torch.Generator().manual_seed(0))
    model.train()
    global_rng = torch.get_rng_state().clone()
    first = train.validate(model, loader, torch.device("cpu"), config["training"])
    second = train.validate(model, loader, torch.device("cpu"), config["training"])
    assert first == second
    assert model.training
    assert not model.target_encoder.training
    assert torch.equal(global_rng, torch.get_rng_state())

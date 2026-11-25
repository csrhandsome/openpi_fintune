"""Tests for DualCameraForceJEPA, PairedTactileFemDataset, and FEM norm stats."""

import copy
import json
from pathlib import Path
import sys

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import Dataset

from force_encoder import train
from force_encoder.data import PairedTactileFemDataset
from force_encoder.model import DualCameraForceJEPA
from force_encoder.model import load_encoder


def _write_mp4(path: Path, num_frames: int, fps: int = 30, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for idx in range(num_frames):
            frame_arr = np.full((size, size, 3), fill_value=(idx * 17) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _make_synthetic_lerobot_v3(root: Path, *, with_fem: bool, num_episodes: int = 3, ep_len: int = 10) -> None:
    (root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data/chunk-000").mkdir(parents=True, exist_ok=True)

    if with_fem:
        cam_keys = [
            "observation.images.left_hand_left_rgb",
            "observation.images.left_hand_right_rgb",
            "observation.images.right_hand_left_rgb",
            "observation.images.right_hand_right_rgb",
        ]
    else:
        cam_keys = ["observation.gripper_image_left", "observation.gripper_image_right"]

    features = {k: {"dtype": "video", "shape": [32, 32, 3]} for k in cam_keys}
    if with_fem:
        features["observation.fem_label"] = {"dtype": "float32", "shape": [2, 2, 11]}

    total_frames = num_episodes * ep_len
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "total_episodes": num_episodes,
                "total_frames": total_frames,
                "features": features,
            }
        )
    )
    (root / "verified.json").write_text(json.dumps({"episodes": num_episodes, "frames": total_frames}))

    for key in cam_keys:
        _write_mp4(root / f"videos/{key}/chunk-000/file-000.mp4", num_frames=total_frames)

    # Write data parquet
    ep_indices = np.repeat(np.arange(num_episodes, dtype=np.int64), ep_len)
    frame_indices = np.tile(np.arange(ep_len, dtype=np.int64), num_episodes)
    timestamps = frame_indices.astype(np.float32) / 30.0
    cols = {
        "episode_index": pa.array(ep_indices),
        "frame_index": pa.array(frame_indices),
        "timestamp": pa.array(timestamps),
    }
    if with_fem:
        fem_vals = np.arange(total_frames * 2 * 2 * 11, dtype=np.float32).reshape(total_frames, 2, 2, 11)
        cols["observation.fem_label"] = pa.array(fem_vals.tolist())
    pq.write_table(pa.table(cols), root / "data/chunk-000/file-000.parquet")

    # Write episodes metadata parquet
    ep_rows = []
    for ep in range(num_episodes):
        row = {
            "episode_index": ep,
            "length": ep_len,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for key in cam_keys:
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = 0
            row[f"videos/{key}/from_timestamp"] = ep * ep_len / 30.0
            row[f"videos/{key}/to_timestamp"] = (ep + 1) * ep_len / 30.0
        ep_rows.append(row)
    pq.write_table(pa.Table.from_pylist(ep_rows), root / "meta/episodes/chunk-000/file-000.parquet")


@pytest.mark.parametrize("input_mode", ["single_frame", "spatiotemporal"])
@pytest.mark.parametrize("target_mode", ["ema", "sigreg"])
def test_dual_camera_force_jepa_forward_backward_and_export(tmp_path, input_mode, target_mode):
    model = DualCameraForceJEPA(
        image_size=32,
        patch_size=8,
        embed_dim=24,
        depth=1,
        num_heads=3,
        predictor_depth=1,
        predictor_dim=24,
        input_mode=input_mode,
        num_frames=4 if input_mode == "spatiotemporal" else 1,
        tubelet_size=2 if input_mode == "spatiotemporal" else 1,
        target_mode=target_mode,
        sigreg_num_proj=16,
    )
    model.set_fem_stats(torch.zeros(11), torch.full((11,), 2.0))
    shape = (2, 3, 32, 32) if input_mode == "single_frame" else (2, 3, 4, 32, 32)
    left = torch.randn(shape)
    right = torch.randn(shape)
    fem_label = torch.randn(2, 2, 11)
    has_fem = torch.tensor([True, False])

    out = model(left, right, fem_label=fem_label, has_fem=has_fem, generator=torch.Generator().manual_seed(4))
    assert torch.isfinite(out["loss"])
    assert out["fem_loss"].item() > 0
    assert out["fem_valid_fraction"].item() == pytest.approx(0.5)
    out["loss"].backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fem_head.parameters())
    if model.target_encoder is not None:
        assert all(p.grad is None for p in model.target_encoder.parameters())

    # Pure real batch (no FEM) should produce zero fem_loss and still backprop JEPA.
    model.zero_grad()
    out_real = model(left, right, fem_label=fem_label, has_fem=torch.tensor([False, False]))
    assert out_real["fem_loss"].item() == 0.0
    assert torch.isfinite(out_real["loss"])
    out_real["loss"].backward()

    # Checkpoint export via load_encoder
    ckpt = tmp_path / "dual_fem.pt"
    torch.save({"model_config": model.model_config, "model": model.state_dict()}, ckpt)
    encoder = load_encoder(ckpt)
    assert encoder(left).shape == (2, 24)
    assert model.encode(left, right).shape == (2, 2, 24)


def test_paired_dataset_and_compute_norm_stats_fem(tmp_path):
    sim_root = tmp_path / "sim_task"
    real_root = tmp_path / "real_task"
    _make_synthetic_lerobot_v3(sim_root, with_fem=True, num_episodes=3, ep_len=10)
    _make_synthetic_lerobot_v3(real_root, with_fem=False, num_episodes=3, ep_len=10)

    ds = PairedTactileFemDataset(
        sources=[
            {"name": "sim", "root": str(sim_root), "has_fem": True},
            {"name": "real", "root": str(real_root), "has_fem": False},
        ],
        image_size=32,
        split="all",
        val_fraction=0.0,
        frame_stride=2,
        num_frames=4,
        clip_frame_step=2,
    )
    # sim has 2 camera pairs x 3 episodes x 2 clips = 12; real has 1 pair x 3 x 2 = 6
    assert len(ds) == 18
    sim_sample = ds[0]
    assert sim_sample["left"].shape == (3, 4, 32, 32)
    assert sim_sample["right"].shape == (3, 4, 32, 32)
    assert sim_sample["fem_label"].shape == (2, 11)
    assert bool(sim_sample["has_fem"]) is True
    # Clip starts at frame 0 with 4 frames, step 2 -> last frame is 6; arm 0 of frame 6
    expected = np.arange(30 * 2 * 2 * 11, dtype=np.float32).reshape(30, 2, 2, 11)[6, 0]
    assert torch.allclose(sim_sample["fem_label"], torch.from_numpy(expected))

    real_sample = ds[-1]
    assert bool(real_sample["has_fem"]) is False
    assert torch.equal(real_sample["fem_label"], torch.zeros(2, 11))

    # Test compute_norm_stats_fem.py
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import compute_norm_stats_fem as fem_stats

        stats_out = tmp_path / "fem_stats.json"
        rc = fem_stats.main(["--roots", str(sim_root), "--output", str(stats_out)])
        assert rc == 0
        payload = json.loads(stats_out.read_text())
        assert payload["count"] == 30 * 2 * 2
        assert len(payload["mean"]) == 11
        assert len(payload["std"]) == 11
    finally:
        sys.path.remove(str(scripts_dir))


class _SyntheticPairedDataset(Dataset):
    def __init__(self, num_frames: int = 4):
        self.num_frames = num_frames
        self.source_ranges = [("sim", 0, 4), ("real", 4, 8)]
        self.summary = {"synthetic_paired": True, "samples": 8}

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(int(index))
        shape = (3, 32, 32) if self.num_frames == 1 else (3, self.num_frames, 32, 32)
        is_sim = index < 4
        return {
            "left": torch.randn(shape, generator=gen),
            "right": torch.randn(shape, generator=gen),
            "fem_label": torch.randn((2, 11), generator=gen) if is_sim else torch.zeros(2, 11),
            "has_fem": torch.tensor(is_sim, dtype=torch.bool),
        }


def test_dual_fem_train_run_and_resume(tmp_path, monkeypatch):
    stats_path = tmp_path / "fem_stats.json"
    stats_path.write_text(json.dumps({"mean": [0.0] * 11, "std": [1.0] * 11}))

    monkeypatch.setattr(train, "make_dataset", lambda cfg, _split: _SyntheticPairedDataset(num_frames=4))
    cfg = {
        "seed": 11,
        "output_dir": str(tmp_path / "out"),
        "data": {"sources": [{"name": "sim", "root": "unused"}], "fem_stats_path": str(stats_path)},
        "model": {
            "model_type": "dual_fem_jepa",
            "input_mode": "spatiotemporal",
            "num_frames": 4,
            "tubelet_size": 2,
            "target_mode": "ema",
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
            "steps_per_epoch": 2,
            "max_steps": 3,
            "learning_rate": 0.001,
            "min_learning_rate": 0.0001,
            "warmup_steps": 1,
            "weight_decay": 0.04,
            "final_weight_decay": 0.1,
            "ema_start": 0.9,
            "ema_end": 0.99,
            "grad_clip": 1.0,
            "log_every": 1,
            "validate_every": 2,
            "val_batches": 1,
            "save_every": 2,
        },
    }
    train.run(copy.deepcopy(cfg))
    latest = tmp_path / "out/latest.pt"
    assert latest.is_file()
    encoder = load_encoder(latest)
    assert encoder(torch.randn(1, 3, 4, 32, 32)).shape == (1, 24)

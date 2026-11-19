"""Temporal ordering, episode boundaries, masked-pixel isolation and no-EMA gradients."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from force_encoder import train
from force_encoder.model import ImageJEPA
from force_encoder.model import load_encoder
from force_encoder.sigreg import SIGReg
from force_encoder.tests.test_data import TACTILE_KEYS
from force_encoder.tests.test_data import _dataset
from force_encoder.tests.test_data import _red_channel
from force_encoder.tests.test_data import v3_root  # noqa: F401


def video_model(**kwargs):
    return ImageJEPA(
        image_size=32,
        patch_size=8,
        embed_dim=24,
        depth=1,
        num_heads=3,
        predictor_dim=24,
        predictor_depth=1,
        input_mode="spatiotemporal",
        num_frames=4,
        tubelet_size=2,
        **kwargs,
    )


def test_video_tokens_tube_masks_and_no_hidden_pixel_leakage():
    model = video_model()
    masks = model.sample_masks(torch.Generator().manual_seed(7))
    context = masks.context.reshape(2, -1)
    assert torch.equal(context[1] - 16, context[0])
    for target in masks.targets:
        assert torch.equal(target.reshape(2, -1)[1] - 16, target.reshape(2, -1)[0])
    assert model.context_encoder.pos_embed.shape == (1, 32, 24)
    assert not torch.equal(model.context_encoder.pos_embed[:, :16], model.context_encoder.pos_embed[:, 16:])
    clips = torch.randn(2, 3, 4, 32, 32, requires_grad=True)
    model(clips, generator=torch.Generator().manual_seed(7))["loss"].backward()
    # Conv3d uses disjoint 2x8x8 blocks. Hidden tube pixels have zero context gradient.
    patch_grad = clips.grad.reshape(2, 3, 2, 2, 4, 8, 4, 8)
    patch_grad = patch_grad.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(2, 32, -1)
    targets = torch.unique(torch.cat(masks.targets))
    assert torch.count_nonzero(patch_grad[:, targets]) == 0
    assert patch_grad[:, masks.context].abs().sum() > 0
    assert model.context_encoder.patch_embed.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("target_mode", ["ema", "sigreg"])
def test_video_checkpoint_and_frame_order(target_mode, tmp_path):
    model = video_model(target_mode=target_mode, sigreg_num_proj=16).eval()
    clips = torch.randn(2, 3, 4, 32, 32)
    tokens = model.encode(clips, pool=False)
    assert tokens.shape == (2, 32, 24)
    assert not torch.allclose(tokens, model.encode(clips.flip(2), pool=False))
    checkpoint = tmp_path / "video.pt"
    torch.save({"model": model.state_dict(), "model_config": model.model_config}, checkpoint)
    encoder = load_encoder(checkpoint)
    torch.testing.assert_close(encoder.forward_features(clips), tokens, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(encoder(clips), model.encode(clips), atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="Expected RGB input"):
        encoder(clips[:, :, 0])


def test_sigreg_has_no_teacher_and_both_branches_receive_prediction_gradients():
    model = video_model(target_mode="sigreg", sigreg_num_proj=16)
    assert model.target_encoder is None
    assert not any(key.startswith("target_encoder.") for key in model.state_dict())
    branches = []

    def keep_grad(_module, _inputs, output):
        output.retain_grad()
        branches.append(output)

    handle = model.context_encoder.blocks.register_forward_hook(keep_grad)
    output = model(torch.randn(3, 3, 4, 32, 32))
    # Isolate prediction MSE so SIGReg cannot hide a mistakenly detached target.
    output["prediction_loss"].backward()
    handle.remove()
    assert len(branches) == 2
    assert all(branch.grad is not None and branch.grad.abs().sum() > 0 for branch in branches)
    assert model.target_projector[0].weight.grad.abs().sum() > 0
    assert output["sigreg_loss"] > 0
    assert output["loss"].item() == pytest.approx(
        output["prediction_loss"].item() + model.sigreg_weight * output["sigreg_loss"].item()
    )


def test_sigreg_detects_collapse_even_with_temporal_variation():
    regularizer = SIGReg(num_proj=64)
    healthy = torch.randn(4, 1024, 12, generator=torch.Generator().manual_seed(3), requires_grad=True)
    collapsed = torch.randn(4, 1, 12, generator=torch.Generator().manual_seed(4)).expand(4, 1024, 12)
    good = regularizer(healthy, torch.Generator().manual_seed(8))
    bad = regularizer(collapsed, torch.Generator().manual_seed(8))
    assert bad > good * 10
    good.backward()
    assert healthy.grad.isfinite().all()
    assert healthy.grad.abs().sum() > 0


def test_clips_preserve_frame_order_camera_and_shared_episode_offsets(v3_root):  # noqa: F811
    dataset = _dataset(v3_root, num_frames=2, clip_frame_step=2)
    assert len(dataset) == 8  # One complete clip per episode/camera; no end padding.
    for index in [7, 0, 4, 3, 6, 1]:
        metadata = dataset.sample_metadata(index)
        assert metadata["frame_indices"] == [0, 2]
        clip = dataset[index]
        assert clip.shape == (3, 2, 16, 16)
        for time, frame_index in enumerate(metadata["frame_indices"]):
            absolute = metadata["episode_index"] * 3 + frame_index
            assert abs(_red_channel(clip[:, time]) - (20 + absolute * 17)) < 5
    with pytest.raises(ValueError, match="No tactile video samples"):
        _dataset(v3_root, num_frames=4)


def test_recorded_clip_timestamps_and_short_episode_exclusion(v3_root):  # noqa: F811
    path = v3_root / "data/chunk-000/file-000.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": np.repeat(np.arange(4), 3),
                "timestamp": np.tile(np.arange(3, dtype=np.float32) / 10, 4),
            }
        ),
        path,
        row_group_size=4,
    )
    dataset = _dataset(v3_root, num_frames=3, timestamp_mode="recorded")
    assert abs(_red_channel(dataset[7][:, -1]) - (20 + 11 * 17)) < 5
    # Shorten one episode while retaining a valid metadata total.
    episode_path = v3_root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(episode_path).to_pylist()
    rows[0]["length"] = 2
    for key in TACTILE_KEYS:
        rows[0][f"videos/{key}/to_timestamp"] = 0.2
    pq.write_table(pa.Table.from_pylist(rows), episode_path)
    info_path = v3_root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 11
    info_path.write_text(json.dumps(info))
    (v3_root / "verified.json").write_text(json.dumps({"episodes": 4, "frames": 11}))
    dataset = _dataset(v3_root, num_frames=3)
    assert len(dataset) == 6
    assert dataset.summary["sources"][0]["excluded_short_episodes"] == 1


def test_yaml_mode_controls_dataset_and_single_frame_compatibility(v3_root):  # noqa: F811
    cfg = {
        "seed": 42,
        "data": {"sources": [{"name": "fixture", "root": v3_root}], "val_fraction": 0},
        "model": {"image_size": 16, "input_mode": "single_frame", "num_frames": 2},
    }
    assert train.make_dataset(cfg, "train")[0].shape == (3, 16, 16)
    cfg["model"]["input_mode"] = "spatiotemporal"
    assert train.make_dataset(cfg, "train")[0].shape == (3, 2, 16, 16)
    model = ImageJEPA(
        image_size=32,
        patch_size=8,
        embed_dim=24,
        num_heads=3,
        depth=1,
        predictor_depth=1,
        predictor_dim=24,
        input_mode="single_frame",
        num_frames=16,
        tubelet_size=2,
    )
    assert model.context_encoder.num_frames == 1
    assert model.context_encoder(torch.randn(2, 3, 32, 32)).shape == (2, 24)

"""Exercise real MP4 seeking, v3 shared-file offsets, and episode splitting."""

# Cache assertions exercise the bounded worker resources deliberately.
# ruff: noqa: SLF001

import json
import pickle
from typing import ClassVar

import av
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from force_encoder.data import IMAGENET_MEAN
from force_encoder.data import IMAGENET_STD
from force_encoder.data import BalancedSourceSampler
from force_encoder.data import LeRobotTactileDataset
from force_encoder.data import preprocess_image

TACTILE_KEYS = ["observation.gripper_image_left", "observation.gripper_image_right"]
EXTERNAL_KEY = "observation.images.front"


def _write_video(path, frames=12, rate=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=rate)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 4
        for index in range(frames):
            # Distinct solid red levels expose seeking into the wrong episode.
            array = np.full((32, 32, 3), (20 + index * 17, 50, 80), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


@pytest.fixture
def v3_root(tmp_path):
    root = tmp_path / "lerobot_v3"
    episode_dir = root / "meta/episodes/chunk-000"
    episode_dir.mkdir(parents=True)
    keys = [*TACTILE_KEYS, EXTERNAL_KEY]
    info = {
        "codebase_version": "v3.0",
        "total_episodes": 4,
        "total_frames": 12,
        # Intentionally wrong, as in the converted tactile NAS datasets.
        "fps": 1,
        "features": {key: {"dtype": "video"} for key in keys},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "verified.json").write_text(json.dumps({"episodes": 4, "frames": 12}))
    episodes = []
    for index in range(4):
        row = {"episode_index": index, "length": 3, "data/chunk_index": 0, "data/file_index": 0}
        for key in keys:
            row.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": 0,
                    f"videos/{key}/from_timestamp": index * 0.3,
                    f"videos/{key}/to_timestamp": (index + 1) * 0.3,
                }
            )
        episodes.append(row)
    pq.write_table(pa.Table.from_pylist(episodes), episode_dir / "file-000.parquet")
    data_path = root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    # Also intentionally wrong timestamps: the default must use actual video rate.
    pq.write_table(
        pa.table(
            {
                "episode_index": np.repeat(np.arange(4), 3),
                "timestamp": np.tile(np.arange(3, dtype=np.float32), 4),
            }
        ),
        data_path,
        row_group_size=4,
    )
    for key in keys:
        _write_video(root / f"videos/{key}/chunk-000/file-000.mp4")
    return root


def _dataset(root, **kwargs):
    return LeRobotTactileDataset(
        [{"name": "fixture", "root": root}], image_size=16, frame_stride=1, split="all", **kwargs
    )


def _red_channel(image):
    return float(image[0].mean() * IMAGENET_STD[0] + IMAGENET_MEAN[0]) * 255


def test_frame_index_uses_actual_video_rate_and_nonzero_offsets(v3_root):
    dataset = _dataset(v3_root)
    assert len(dataset) == 24  # Four episodes, two tactile cameras, three frames.
    assert dataset.summary["sources"][0]["video_keys"] == TACTILE_KEYS
    # Random order exercises backwards seeking as well as shared-file offsets.
    for index in [23, 0, 14, 7, 17, 5, 20, 2]:
        metadata = dataset.sample_metadata(index)
        absolute_frame = metadata["episode_index"] * 3 + metadata["frame_index"]
        image = dataset[index]
        assert image.shape == (3, 16, 16)
        assert image.dtype == torch.float32
        assert abs(_red_channel(image) - (20 + absolute_frame * 17)) < 5
    assert len(dataset._videos) <= 4
    dataset.close()


def test_recorded_timestamps_reads_correct_episode_across_row_groups(v3_root):
    data_path = v3_root / "data/chunk-000/file-000.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": np.repeat(np.arange(4), 3),
                "timestamp": np.tile(np.arange(3, dtype=np.float32) / 10, 4),
            }
        ),
        data_path,
        row_group_size=4,
    )
    dataset = _dataset(v3_root, timestamp_mode="recorded")
    assert abs(_red_channel(dataset[23]) - (20 + 11 * 17)) < 5
    assert len(dataset._timestamps) == 1


def test_recorded_mode_fails_on_incorrect_converted_timestamps(v3_root):
    dataset = _dataset(v3_root, timestamp_mode="recorded")
    with pytest.raises(ValueError, match="exceeds episode video interval"):
        dataset[2]


def test_episode_split_keeps_all_cameras_together(v3_root):
    kwargs = {"sources": [{"name": "fixture", "root": v3_root}], "frame_stride": 2, "val_fraction": 0.25, "seed": 123}
    train = LeRobotTactileDataset(split="train", **kwargs)
    val = LeRobotTactileDataset(split="val", **kwargs)
    repeated = LeRobotTactileDataset(split="val", **kwargs)
    train_episodes = {train.sample_metadata(i)["episode_index"] for i in range(len(train))}
    val_episodes = {val.sample_metadata(i)["episode_index"] for i in range(len(val))}
    assert train_episodes.isdisjoint(val_episodes)
    assert train_episodes | val_episodes == set(range(4))
    assert val_episodes == {repeated.sample_metadata(i)["episode_index"] for i in range(len(repeated))}
    assert len(train) == 12
    assert len(val) == 4
    assert all(train.sample_metadata(i)["frame_index"] in {0, 2} for i in range(len(train)))


def test_train_validation_test_episodes_are_disjoint(v3_root):
    options = {
        "sources": [{"name": "fixture", "root": v3_root}],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "frame_stride": 1,
        "seed": 42,
    }
    datasets = {split: LeRobotTactileDataset(split=split, **options) for split in ("train", "val", "test")}
    groups = {split: set(dataset.episode_groups) for split, dataset in datasets.items()}
    assert len(groups["train"]) == 2
    assert len(groups["val"]) == len(groups["test"]) == 1
    assert groups["train"].isdisjoint(groups["val"] | groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert len(set.union(*groups.values())) == 4
    for dataset in datasets.values():
        assert "data_path" in dataset.sample_metadata(0)
        assert all(dataset.sample_metadata(i)["source"] == "fixture" for i in range(len(dataset)))


def test_invalid_or_empty_three_way_splits_fail(v3_root):
    source = [{"name": "fixture", "root": v3_root}]
    with pytest.raises(ValueError, match="val_fraction .*test_fraction"):
        LeRobotTactileDataset(source, val_fraction=0.6, test_fraction=0.5)
    with pytest.raises(ValueError, match="at least 3 episodes"):
        LeRobotTactileDataset(source, val_fraction=0.1, test_fraction=0.1, max_episodes_per_source=2)


def test_partial_and_mismatched_sources_are_rejected(v3_root):
    marker = v3_root / "verified.json"
    marker.unlink()
    with pytest.raises(ValueError, match="missing .*verified.json"):
        _dataset(v3_root)
    marker.write_text(json.dumps({"episodes": 4, "frames": 13}))
    with pytest.raises(ValueError, match="verified frames"):
        _dataset(v3_root)
    marker.write_text(json.dumps({"episodes": 4, "frames": 12}))
    episode_path = v3_root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(episode_path)
    pq.write_table(table.slice(0, 3), episode_path)
    with pytest.raises(ValueError, match="episode table count"):
        _dataset(v3_root)


def test_auto_key_selection_excludes_external_rgb_camera(v3_root):
    dataset = _dataset(v3_root)
    assert {dataset.sample_metadata(i)["video_key"] for i in range(len(dataset))} == set(TACTILE_KEYS)
    selected = LeRobotTactileDataset(
        [
            {
                "name": "fixture",
                "root": v3_root,
                "video_keys": [TACTILE_KEYS[1]],
            }
        ],
        frame_stride=1,
        split="all",
    )
    assert len(selected) == 12
    with pytest.raises(ValueError, match="not a video feature"):
        LeRobotTactileDataset([{"name": "fixture", "root": v3_root, "video_keys": ["action"]}])


def test_frame_count_duration_mismatch_is_rejected(v3_root):
    episode_path = v3_root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(episode_path).to_pylist()
    for key in TACTILE_KEYS:
        rows[0][f"videos/{key}/to_timestamp"] = 1.3
    pq.write_table(pa.Table.from_pylist(rows), episode_path)
    with pytest.raises(ValueError, match="frame_index mode requires"):
        _dataset(v3_root)[0]


def test_worker_pickle_drops_decoder_handles_and_caches(v3_root):
    dataset = _dataset(v3_root, video_cache_size=1, frame_cache_size=2)
    first = dataset[0]
    for index in [3, 6, 9, 12]:
        dataset[index]
    assert len(dataset._videos) <= 1
    assert len(dataset._frames) <= 2
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored._videos
    assert torch.equal(first, restored[0])


def test_balanced_sampler_is_compact_and_deterministic():
    class FakeDataset:
        source_ranges: ClassVar[list[tuple[str, int, int]]] = [("tiny", 0, 3), ("large", 3, 10003)]

        def __len__(self):
            return 10003

    sampler = BalancedSourceSampler(FakeDataset(), num_samples=1000, seed=7)
    first = list(sampler)
    assert first == list(sampler)
    assert 400 < sum(index < 3 for index in first) < 600
    sampler.set_epoch(1)
    assert first != list(sampler)
    other_rank = BalancedSourceSampler(FakeDataset(), num_samples=1000, seed=7, rank=1, num_replicas=2)
    assert first != list(other_rank)


def test_preprocess_matches_training_and_is_rgb(v3_root):
    image = Image.new("L", (9, 15), color=128)
    actual = preprocess_image(image, image_size=16)
    assert actual.shape == (3, 16, 16)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    assert torch.allclose(actual * std + mean, torch.full_like(actual, 128 / 255))
    dataset = _dataset(v3_root)
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    assert dataset.sample_metadata(-1) == dataset.sample_metadata(len(dataset) - 1)

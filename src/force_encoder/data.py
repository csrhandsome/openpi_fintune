"""Read local, verified LeRobot v3 tactile videos without the LeRobot runtime.

Only ``observation.gripper_image*`` video features are selected automatically.
The converted NAS datasets contain incorrect ``info.fps`` *and* parquet
timestamps, so their default ``frame_index`` mode uses the actual video rate.
Use source option ``timestamp_mode: recorded`` for datasets with correct episode
timestamps. All image preprocessing is shared between training and inference.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import operator
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Resize the complete RGB sensor image and return ImageNet-normalized CHW."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std


@dataclass(frozen=True)
class _CameraEpisode:
    source: str
    episode_index: int
    video_key: str
    length: int
    data_path: Path
    video_path: Path
    from_timestamp: float
    to_timestamp: float
    timestamp_mode: str


@dataclass
class _Video:
    container: Any
    stream: Any
    rate: float


class LeRobotTactileDataset(Dataset):
    """A compact episode/camera index over local LeRobot v3 datasets.

    ``sources`` contains dictionaries with ``name``, ``root`` (the lerobot_v3
    directory), optional ``video_keys``, and optional ``timestamp_mode``.
    ``verified.json`` is required by default and its episode/frame counts must
    match both info.json and the complete episode table. Splits happen before
    expanding cameras, preventing leakage of another sensor from one episode.
    Video and timestamp caches are bounded and reset after a worker fork.
    """

    def __init__(
        self,
        sources: list[dict[str, Any]],
        image_size: int = 224,
        split: str = "train",
        val_fraction: float = 0.05,
        seed: int = 42,
        frame_stride: int = 5,
        max_episodes_per_source: int | None = None,
        *,
        test_fraction: float = 0.0,
        require_verified: bool = True,
        timestamp_mode: str = "frame_index",
        video_cache_size: int = 4,
        timestamp_cache_size: int = 8,
        frame_cache_size: int = 16,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be train, val, test, or all")
        if not 0 <= val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
            raise ValueError("test_fraction must be nonnegative and val_fraction + test_fraction must be < 1")
        if frame_stride < 1 or image_size < 1:
            raise ValueError("frame_stride and image_size must be positive")
        if max_episodes_per_source is not None and max_episodes_per_source < 1:
            raise ValueError("max_episodes_per_source must be positive")
        if min(video_cache_size, timestamp_cache_size, frame_cache_size) < 1:
            raise ValueError("cache sizes must be positive")
        self.image_size = image_size
        self.frame_stride = frame_stride
        self.split = split
        self._video_cache_size = video_cache_size
        self._timestamp_cache_size = timestamp_cache_size
        self._frame_cache_size = frame_cache_size
        self._streams: list[_CameraEpisode] = []
        self._ends: list[int] = []
        self.source_ranges: list[tuple[str, int, int]] = []
        self._summary: dict[str, Any] = {
            "split": split,
            "sources": [],
            "frame_stride": frame_stride,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "split_seed": seed,
        }
        names: set[str] = set()
        for source in sources:
            name = str(source["name"])
            if name in names:
                raise ValueError(f"Duplicate source name: {name}")
            names.add(name)
            mode = source.get("timestamp_mode", timestamp_mode)
            if mode not in {"frame_index", "recorded"}:
                raise ValueError(f"{name}: timestamp_mode must be frame_index or recorded")
            self._add_source(source, mode, require_verified, val_fraction, test_fraction, seed, max_episodes_per_source)
        if not self._streams:
            raise ValueError(f"No tactile video samples in {split} split")
        self._cumulative = np.asarray(self._ends, dtype=np.int64)
        self._summary["samples"] = len(self)
        self._summary["camera_streams"] = len(self._streams)
        self._pid: int | None = None
        self._videos: OrderedDict[str, _Video] = OrderedDict()
        self._timestamps: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._parquet_files: OrderedDict[str, Any] = OrderedDict()
        self._frames: OrderedDict[tuple[str, float], torch.Tensor] = OrderedDict()
        self._validated_streams: set[tuple[str, int, str]] = set()

    def _add_source(self, source, mode, require_verified, val_fraction, test_fraction, seed, episode_limit):
        import pyarrow.parquet as pq

        root = Path(source["root"]).expanduser().resolve()
        name = str(source["name"])
        info = json.loads((root / "meta/info.json").read_text())
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"{name}: expected LeRobot v3.0, found {info.get('codebase_version')!r}")
        marker = root / "verified.json"
        if require_verified and not marker.is_file():
            raise ValueError(f"{name}: missing {marker}; unfinished datasets are excluded")
        if marker.is_file():
            verified = json.loads(marker.read_text())
            for marker_key, info_key in (("episodes", "total_episodes"), ("frames", "total_frames")):
                if int(verified.get(marker_key, -1)) != int(info.get(info_key, -2)):
                    raise ValueError(f"{name}: verified {marker_key} does not match info.{info_key}")
        features = info.get("features", {})
        keys = source.get("video_keys")
        if keys is None:
            keys = sorted(
                k
                for k, v in features.items()
                if k.startswith("observation.gripper_image") and v.get("dtype") == "video"
            )
        else:
            if isinstance(keys, str):
                raise ValueError(f"{name}: video_keys must be a list, not a string")
            keys = list(keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError(f"{name}: select at least one unique tactile video key")
        for key in keys:
            if features.get(key, {}).get("dtype") != "video":
                raise ValueError(f"{name}: {key!r} is not a video feature")
        columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
        for key in keys:
            columns.extend(
                f"videos/{key}/{field}" for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
            )
        files = sorted((root / "meta/episodes").rglob("*.parquet"))
        if not files:
            raise ValueError(f"{name}: no LeRobot v3 episode parquet files")
        episodes = []
        for path in files:
            episodes.extend(pq.read_table(path, columns=columns).to_pylist())
        episodes.sort(key=lambda row: int(row["episode_index"]))
        indices = [int(row["episode_index"]) for row in episodes]
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name}: duplicate episode indices")
        if len(episodes) != int(info["total_episodes"]):
            raise ValueError(f"{name}: episode table count does not match info.total_episodes")
        if any(int(row["length"]) < 1 for row in episodes):
            raise ValueError(f"{name}: empty episode in metadata")
        if sum(int(row["length"]) for row in episodes) != int(info["total_frames"]):
            raise ValueError(f"{name}: episode lengths do not match info.total_frames")
        # Stable across Python processes, source order, and independently constructed splits.
        name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        rng = np.random.default_rng((seed + name_seed) % (2**64))
        order = rng.permutation(len(episodes)).tolist()
        if episode_limit is not None:
            order = order[:episode_limit]
        minimum = 1 + int(val_fraction > 0) + int(test_fraction > 0)
        if len(order) < minimum:
            raise ValueError(f"{name}: need at least {minimum} episodes for nonempty train/val/test splits")
        n_val = (
            min(len(order) - 1 - int(test_fraction > 0), max(1, round(len(order) * val_fraction)))
            if val_fraction
            else 0
        )
        n_test = min(len(order) - n_val - 1, max(1, round(len(order) * test_fraction))) if test_fraction else 0
        partitions = {
            "all": order,
            "val": order[:n_val],
            "test": order[n_val : n_val + n_test],
            "train": order[n_val + n_test :],
        }
        chosen = partitions[self.split]
        selected = [episodes[i] for i in sorted(chosen)]
        start = self._ends[-1] if self._ends else 0
        for row in selected:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            data_path = root / info.get("data_path", _DATA_PATH).format(
                episode_index=episode_index,
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            for key in keys:
                prefix = f"videos/{key}/"
                video_path = root / info.get("video_path", _VIDEO_PATH).format(
                    video_key=key,
                    episode_index=episode_index,
                    chunk_index=int(row[prefix + "chunk_index"]),
                    file_index=int(row[prefix + "file_index"]),
                )
                offset, end = float(row[prefix + "from_timestamp"]), float(row[prefix + "to_timestamp"])
                if not math.isfinite(offset) or not math.isfinite(end) or offset < 0 or end <= offset:
                    raise ValueError(f"{name} episode {episode_index}: invalid video timestamp interval")
                self._streams.append(
                    _CameraEpisode(name, episode_index, key, length, data_path, video_path, offset, end, mode)
                )
                self._ends.append((self._ends[-1] if self._ends else 0) + (length - 1) // self.frame_stride + 1)
        end = self._ends[-1] if self._ends else 0
        if end > start:
            self.source_ranges.append((name, start, end))
        self._summary["sources"].append(
            {
                "name": name,
                "root": str(root),
                "verified": marker.is_file(),
                "video_keys": keys,
                "timestamp_mode": mode,
                "episodes": len(selected),
                "frames": sum(int(row["length"]) for row in selected),
                "samples": end - start,
                "available_episodes": len(episodes),
                "available_frames": int(info["total_frames"]),
            }
        )

    @property
    def summary(self) -> dict[str, Any]:
        return self._summary

    @property
    def episode_groups(self) -> list[str]:
        """Stable episode group identities without expanding individual frames."""
        return sorted({f"{record.source}/{record.episode_index}" for record in self._streams})

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def _locate(self, index: int) -> tuple[_CameraEpisode, int]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        stream_index = int(np.searchsorted(self._cumulative, index, side="right"))
        start = int(self._cumulative[stream_index - 1]) if stream_index else 0
        return self._streams[stream_index], (index - start) * self.frame_stride

    def sample_metadata(self, index: int) -> dict[str, Any]:
        record, frame_index = self._locate(index)
        return {
            "source": record.source,
            "episode_index": record.episode_index,
            "video_key": record.video_key,
            "frame_index": frame_index,
            "video_path": str(record.video_path),
            "data_path": str(record.data_path),
        }

    def _ensure_process(self) -> None:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()

    def _video(self, path: Path) -> _Video:
        import av

        key = str(path)
        if key not in self._videos:
            container = av.open(key, mode="r")
            try:
                stream = container.streams.video[0]
                stream.thread_count = 1
                rate = float(stream.average_rate or stream.guessed_rate or 0)
                if not math.isfinite(rate) or rate <= 0:
                    raise ValueError(f"Cannot determine actual video rate for {path}")
            except Exception:
                container.close()
                raise
            self._videos[key] = _Video(container, stream, rate)
            if len(self._videos) > self._video_cache_size:
                self._videos.popitem(last=False)[1].container.close()
        self._videos.move_to_end(key)
        return self._videos[key]

    def _episode_timestamps(self, record: _CameraEpisode) -> np.ndarray:
        import pyarrow.parquet as pq

        cache_key = (str(record.data_path), record.episode_index)
        if cache_key not in self._timestamps:
            path = str(record.data_path)
            if path not in self._parquet_files:
                self._parquet_files[path] = pq.ParquetFile(path)
                if len(self._parquet_files) > self._video_cache_size:
                    self._parquet_files.popitem(last=False)[1].close()
            self._parquet_files.move_to_end(path)
            parquet = self._parquet_files[path]
            episode_column = parquet.schema_arrow.get_field_index("episode_index")
            if episode_column < 0:
                raise ValueError(f"Missing episode_index column in {path}")
            pieces = []
            for group in range(parquet.num_row_groups):
                stats = parquet.metadata.row_group(group).column(episode_column).statistics
                if stats is not None and stats.has_min_max and not stats.min <= record.episode_index <= stats.max:
                    continue
                table = parquet.read_row_group(group, columns=["episode_index", "timestamp"])
                mask = table["episode_index"].to_numpy() == record.episode_index
                pieces.append(table["timestamp"].to_numpy()[mask])
            times = np.concatenate(pieces).astype(np.float64) if pieces else np.empty(0)
            if len(times) != record.length or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                raise ValueError(f"{path}: invalid or incomplete timestamps for episode {record.episode_index}")
            if times[0] < -1e-5:
                raise ValueError(f"{path}: negative episode timestamp")
            self._timestamps[cache_key] = times
            if len(self._timestamps) > self._timestamp_cache_size:
                self._timestamps.popitem(last=False)
        self._timestamps.move_to_end(cache_key)
        return self._timestamps[cache_key]

    def __getitem__(self, index: int) -> torch.Tensor:
        self._ensure_process()
        record, frame_index = self._locate(index)
        video = self._video(record.video_path)
        if record.timestamp_mode == "frame_index":
            validation_key = (record.source, record.episode_index, record.video_key)
            if validation_key not in self._validated_streams:
                expected = (record.to_timestamp - record.from_timestamp) * video.rate
                if abs(expected - record.length) > max(2.0, record.length * 0.001):
                    raise ValueError(
                        f"{record.source} episode {record.episode_index}: video span at actual rate "
                        f"contains {expected:.2f} frames, metadata length is {record.length}; "
                        "frame_index mode requires all video frames in order"
                    )
                self._validated_streams.add(validation_key)
            target = record.from_timestamp + frame_index / video.rate
        else:
            target = record.from_timestamp + float(self._episode_timestamps(record)[frame_index])
        if target >= record.to_timestamp + 0.6 / video.rate:
            raise ValueError(f"{record.source}: timestamp {target:.6f} exceeds episode video interval")
        cache_key = (str(record.video_path), target)
        if cache_key not in self._frames:
            self._frames[cache_key] = preprocess_image(self._decode(video, target, record.video_path), self.image_size)
            if len(self._frames) > self._frame_cache_size:
                self._frames.popitem(last=False)
        self._frames.move_to_end(cache_key)
        return self._frames[cache_key].clone()

    @staticmethod
    def _decode(video: _Video, target: float, path: Path) -> Image.Image:
        # FFmpeg seeks to a preceding keyframe, then only decodes the short GOP
        # containing the requested frame, including shared-file episode offsets.
        video.container.seek(
            int(target / float(video.stream.time_base)), stream=video.stream, backward=True, any_frame=False
        )
        best = None
        best_distance = math.inf
        tolerance = max(1e-4, 0.6 / video.rate)
        for frame in video.container.decode(video.stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            distance = abs(timestamp - target)
            if distance < best_distance:
                best, best_distance = frame, distance
            if timestamp >= target - 1e-7:
                break
        if best is None or best_distance > tolerance:
            raise ValueError(f"{path}: no video frame near {target:.6f}s (distance={best_distance:.6f}s)")
        return best.to_image().convert("RGB")

    def close(self) -> None:
        for video in getattr(self, "_videos", {}).values():
            video.container.close()
        for parquet in getattr(self, "_parquet_files", {}).values():
            parquet.close()
        self._videos = OrderedDict()
        self._timestamps = OrderedDict()
        self._parquet_files = OrderedDict()
        self._frames = OrderedDict()
        self._validated_streams = set()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(
            _pid=None,
            _videos=OrderedDict(),
            _timestamps=OrderedDict(),
            _parquet_files=OrderedDict(),
            _frames=OrderedDict(),
            _validated_streams=set(),
        )
        return state

    def __del__(self):
        self.close()


class BalancedSourceSampler(Sampler[int]):
    """Sample sources equally, then sample a frame uniformly within that source.

    Sampling is with replacement and holds only O(number of sources) indices.
    ``num_samples`` is the number yielded *per rank*. Different DDP ranks use
    independent deterministic random streams. Call ``set_epoch`` each epoch.
    """

    def __init__(self, dataset, num_samples=None, seed=42, rank=0, num_replicas=1):
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid rank or num_replicas")
        self.ranges = [(start, end) for _, start, end in dataset.source_ranges if end > start]
        if not self.ranges:
            raise ValueError("no nonempty sources to sample")
        self.num_samples = math.ceil(len(dataset) / num_replicas) if num_samples is None else int(num_samples)
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.seed, self.rank, self.epoch = seed, rank, 0

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.rank]))
        for _ in range(self.num_samples):
            start, end = self.ranges[int(rng.integers(len(self.ranges)))]
            yield int(rng.integers(start, end))

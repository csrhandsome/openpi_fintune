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


@dataclass(frozen=True)
class _PairedCameraEpisode:
    source: str
    episode_index: int
    left: _CameraEpisode
    right: _CameraEpisode
    has_fem: bool
    fem_feature: str | None
    fem_arm_index: int | None


@dataclass
class _Video:
    container: Any
    stream: Any
    rate: float


class LeRobotTactileDataset(Dataset):
    """A compact episode/camera index over local LeRobot v3 datasets.

    ``sources`` contains dictionaries with ``name``, ``root`` (the lerobot_v3
    directory), optional ``video_keys``, and optional ``timestamp_mode``.
    Dataset-level ``video_keys`` may be provided as a default when sources omit
    ``video_keys``; otherwise ``observation.gripper_image*`` keys are selected.
    Native v3 exports may select another ``info_file`` (relative to root) and
    explicitly override ``require_verified`` per source or at dataset level.
    Setting ``skip_incomplete_sources=True`` skips sources whose metadata is not
    yet present or that lack enough episodes for the requested splits, recording
    them in ``summary['skipped_sources']`` while still validating any present
    dataset metadata. Complete episode/frame metadata checks apply even when a
    native export has no conversion marker. ``verified.json`` is required by
    default and its episode/frame counts must match both info.json and the
    complete episode table. Splits happen before expanding cameras, preventing
    leakage of another sensor from one episode. Video and timestamp caches are
    bounded and reset after a worker fork.
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
        video_keys: list[str] | None = None,
        skip_incomplete_sources: bool = False,
        video_cache_size: int = 4,
        timestamp_cache_size: int = 8,
        frame_cache_size: int = 16,
        num_frames: int = 1,
        clip_frame_step: int = 1,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be train, val, test, or all")
        if not 0 <= val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
            raise ValueError("test_fraction must be nonnegative and val_fraction + test_fraction must be < 1")
        if frame_stride < 1 or image_size < 1:
            raise ValueError("frame_stride and image_size must be positive")
        if num_frames < 1 or clip_frame_step < 1:
            raise ValueError("num_frames and clip_frame_step must be positive")
        if not isinstance(skip_incomplete_sources, bool):
            raise ValueError("skip_incomplete_sources must be a boolean")
        default_video_keys: list[str] | None = None
        if video_keys is not None:
            if isinstance(video_keys, str):
                raise ValueError("video_keys must be a list, not a string")
            default_video_keys = list(video_keys)
            if not default_video_keys or len(set(default_video_keys)) != len(default_video_keys):
                raise ValueError("video_keys must select at least one unique tactile video key")
        self.num_frames = num_frames
        self.clip_frame_step = clip_frame_step
        self.clip_span = 1 + (num_frames - 1) * clip_frame_step
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
            "skipped_sources": [],
            "frame_stride": frame_stride,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "split_seed": seed,
            "num_frames": num_frames,
            "clip_frame_step": clip_frame_step,
            "clip_span_frames": self.clip_span,
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
            self._add_source(
                source,
                mode,
                require_verified,
                val_fraction,
                test_fraction,
                seed,
                max_episodes_per_source,
                default_video_keys,
                skip_incomplete_sources,
            )
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

    def _add_source(
        self,
        source,
        mode,
        require_verified,
        val_fraction,
        test_fraction,
        seed,
        episode_limit,
        default_video_keys,
        skip_incomplete,
    ):
        import pyarrow.parquet as pq

        root = Path(source["root"]).expanduser().resolve()
        name = str(source["name"])
        source_skip_incomplete = source.get("skip_incomplete", skip_incomplete)
        if not isinstance(source_skip_incomplete, bool):
            raise ValueError(f"{name}: skip_incomplete must be a boolean")
        info_file = root / source.get("info_file", "meta/info.json")
        if source_skip_incomplete and not info_file.is_file():
            self._summary["skipped_sources"].append({"name": name, "root": str(root), "reason": "missing_info_file"})
            return
        info = json.loads(info_file.read_text())
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"{name}: expected LeRobot v3.0, found {info.get('codebase_version')!r}")
        marker = root / "verified.json"
        source_require_verified = source.get("require_verified", require_verified)
        if not isinstance(source_require_verified, bool):
            raise ValueError(f"{name}: require_verified must be a boolean")
        if source_require_verified and not marker.is_file():
            raise ValueError(f"{name}: missing {marker}; unfinished datasets are excluded")
        if marker.is_file():
            verified = json.loads(marker.read_text())
            for marker_key, info_key in (("episodes", "total_episodes"), ("frames", "total_frames")):
                if int(verified.get(marker_key, -1)) != int(info.get(info_key, -2)):
                    raise ValueError(f"{name}: verified {marker_key} does not match info.{info_key}")
        features = info.get("features", {})
        keys = source.get("video_keys")
        if keys is None and default_video_keys is not None:
            keys = list(default_video_keys)
        elif keys is None:
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
            if source_skip_incomplete and int(info.get("total_episodes", 0)) == 0:
                self._summary["skipped_sources"].append({"name": name, "root": str(root), "reason": "no_episodes"})
                return
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
            if source_skip_incomplete:
                self._summary["skipped_sources"].append(
                    {
                        "name": name,
                        "root": str(root),
                        "reason": f"insufficient_episodes:{len(order)}<{minimum}",
                    }
                )
                return
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
        eligible = [row for row in selected if int(row["length"]) >= self.clip_span]
        start = self._ends[-1] if self._ends else 0
        for row in eligible:
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
                self._ends.append(
                    (self._ends[-1] if self._ends else 0) + (length - self.clip_span) // self.frame_stride + 1
                )
        end = self._ends[-1] if self._ends else 0
        if end > start:
            self.source_ranges.append((name, start, end))
        self._summary["sources"].append(
            {
                "name": name,
                "root": str(root),
                "info_file": str(info_file),
                "require_verified": source_require_verified,
                "verified": marker.is_file(),
                "video_keys": keys,
                "timestamp_mode": mode,
                "episodes": len(eligible),
                "frames": sum(int(row["length"]) for row in eligible),
                "excluded_short_episodes": len(selected) - len(eligible),
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
        metadata = {
            "source": record.source,
            "episode_index": record.episode_index,
            "video_key": record.video_key,
            "frame_index": frame_index,
            "video_path": str(record.video_path),
            "data_path": str(record.data_path),
        }
        if self.num_frames > 1:
            metadata.update(
                frame_indices=[frame_index + step * self.clip_frame_step for step in range(self.num_frames)],
                clip_frame_step=self.clip_frame_step,
            )
        return metadata

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
        return self._decode_camera(record, frame_index)

    def _decode_camera(self, record: _CameraEpisode, frame_index: int) -> torch.Tensor:
        video = self._video(record.video_path)
        if self.num_frames > 1:
            indices = [frame_index + step * self.clip_frame_step for step in range(self.num_frames)]
            targets = [self._frame_timestamp(record, video, frame) for frame in indices]
            # One seek per clip, followed by sequential decoding, even on NAS.
            frames = self._decode_clip(video, targets, record.video_path)
            return torch.stack([preprocess_image(frame, self.image_size) for frame in frames], dim=1)
        target = self._frame_timestamp(record, video, frame_index)
        cache_key = (str(record.video_path), target)
        if cache_key not in self._frames:
            self._frames[cache_key] = preprocess_image(self._decode(video, target, record.video_path), self.image_size)
            if len(self._frames) > self._frame_cache_size:
                self._frames.popitem(last=False)
        self._frames.move_to_end(cache_key)
        return self._frames[cache_key].clone()

    def _frame_timestamp(self, record: _CameraEpisode, video: _Video, frame_index: int) -> float:
        if record.timestamp_mode == "frame_index":
            validation_key = (record.source, record.episode_index, record.video_key)
            if validation_key not in self._validated_streams:
                expected = (record.to_timestamp - record.from_timestamp) * video.rate
                # The NAS converter copies complete camera videos but derives
                # length from trajectory rows. Extra trailing video frames are
                # valid: keep the trajectory-aligned prefix, without rescaling.
                if record.length - expected > max(2.0, record.length * 0.001):
                    raise ValueError(
                        f"{record.source} episode {record.episode_index}: video span at actual rate "
                        f"contains {expected:.2f} frames, metadata length is {record.length}; "
                        "frame_index mode requires enough video frames in order"
                    )
                self._validated_streams.add(validation_key)
            target = record.from_timestamp + frame_index / video.rate
        else:
            target = record.from_timestamp + float(self._episode_timestamps(record)[frame_index])
        if target >= record.to_timestamp + 0.6 / video.rate:
            raise ValueError(f"{record.source}: timestamp {target:.6f} exceeds episode video interval")
        return target

    @staticmethod
    def _decode_clip(video: _Video, targets: list[float], path: Path) -> list[Image.Image]:
        video.container.seek(
            int(targets[0] / float(video.stream.time_base)), stream=video.stream, backward=True, any_frame=False
        )
        output = []
        previous = None
        tolerance = max(1e-4, 0.6 / video.rate)
        for frame in video.container.decode(video.stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            while len(output) < len(targets) and timestamp >= targets[len(output)] - 1e-7:
                target = targets[len(output)]
                best, distance = frame, abs(timestamp - target)
                if previous is not None and abs(previous[0] - target) < distance:
                    best, distance = previous[1], abs(previous[0] - target)
                if distance > tolerance:
                    raise ValueError(f"{path}: no video frame near {target:.6f}s (distance={distance:.6f}s)")
                output.append(best.to_image().convert("RGB"))
            if len(output) == len(targets):
                return output
            previous = (timestamp, frame)
        # A rounded target can be just after the last frame's timestamp.
        while previous is not None and len(output) < len(targets):
            if abs(previous[0] - targets[len(output)]) > tolerance:
                break
            output.append(previous[1].to_image().convert("RGB"))
        if len(output) != len(targets):
            raise ValueError(f"{path}: incomplete clip; decoded {len(output)}/{len(targets)} frames")
        return output

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


class PairedTactileFemDataset(LeRobotTactileDataset):
    """Dual-camera (left/right finger) tactile dataset with optional FEM labels.

    Each sample corresponds to one gripper's paired left/right internal RGB
    clips and, when ``has_fem: true`` on the source, the corresponding ``[2, 11]``
    FEM physical state label at the final frame of the clip.
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
        video_keys: list[str] | None = None,
        skip_incomplete_sources: bool = False,
        video_cache_size: int = 4,
        timestamp_cache_size: int = 8,
        frame_cache_size: int = 16,
        num_frames: int = 1,
        clip_frame_step: int = 1,
        fem_dim: int = 11,
    ) -> None:
        self.fem_dim = fem_dim
        self._fem_labels: OrderedDict[tuple[str, int, str], np.ndarray] = OrderedDict()
        super().__init__(
            sources=sources,
            image_size=image_size,
            split=split,
            val_fraction=val_fraction,
            seed=seed,
            frame_stride=frame_stride,
            max_episodes_per_source=max_episodes_per_source,
            test_fraction=test_fraction,
            require_verified=require_verified,
            timestamp_mode=timestamp_mode,
            video_keys=video_keys,
            skip_incomplete_sources=skip_incomplete_sources,
            video_cache_size=video_cache_size,
            timestamp_cache_size=timestamp_cache_size,
            frame_cache_size=frame_cache_size,
            num_frames=num_frames,
            clip_frame_step=clip_frame_step,
        )

    @staticmethod
    def _resolve_camera_pairs(name: str, source: dict[str, Any], features: dict[str, Any]) -> list[tuple[str, str]]:
        raw_pairs = source.get("camera_pairs")
        if raw_pairs is not None:
            pairs = [tuple(pair) for pair in raw_pairs]
        elif (
            features.get("observation.gripper_image_left", {}).get("dtype") == "video"
            and features.get("observation.gripper_image_right", {}).get("dtype") == "video"
        ):
            pairs = [("observation.gripper_image_left", "observation.gripper_image_right")]
        elif all(
            features.get(f"observation.images.{cam}", {}).get("dtype") == "video"
            for cam in ("left_hand_left_rgb", "left_hand_right_rgb", "right_hand_left_rgb", "right_hand_right_rgb")
        ):
            pairs = [
                ("observation.images.left_hand_left_rgb", "observation.images.left_hand_right_rgb"),
                ("observation.images.right_hand_left_rgb", "observation.images.right_hand_right_rgb"),
            ]
        else:
            raise ValueError(f"{name}: specify camera_pairs of [left_key, right_key]")
        if not pairs or any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
            raise ValueError(f"{name}: camera_pairs must be nonempty [left, right] pairs")
        for left_key, right_key in pairs:
            for key in (left_key, right_key):
                if features.get(key, {}).get("dtype") != "video":
                    raise ValueError(f"{name}: {key!r} is not a video feature")
        return pairs

    def _add_source(
        self,
        source,
        mode,
        require_verified,
        val_fraction,
        test_fraction,
        seed,
        episode_limit,
        default_video_keys,
        skip_incomplete,
    ):
        import pyarrow.parquet as pq

        root = Path(source["root"]).expanduser().resolve()
        name = str(source["name"])
        source_skip_incomplete = source.get("skip_incomplete", skip_incomplete)
        if not isinstance(source_skip_incomplete, bool):
            raise ValueError(f"{name}: skip_incomplete must be a boolean")
        info_file = root / source.get("info_file", "meta/info.json")
        if source_skip_incomplete and not info_file.is_file():
            self._summary["skipped_sources"].append({"name": name, "root": str(root), "reason": "missing_info_file"})
            return
        info = json.loads(info_file.read_text())
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"{name}: expected LeRobot v3.0, found {info.get('codebase_version')!r}")
        marker = root / "verified.json"
        source_require_verified = source.get("require_verified", require_verified)
        if not isinstance(source_require_verified, bool):
            raise ValueError(f"{name}: require_verified must be a boolean")
        if source_require_verified and not marker.is_file():
            raise ValueError(f"{name}: missing {marker}; unfinished datasets are excluded")
        if marker.is_file():
            verified = json.loads(marker.read_text())
            for marker_key, info_key in (("episodes", "total_episodes"), ("frames", "total_frames")):
                if int(verified.get(marker_key, -1)) != int(info.get(info_key, -2)):
                    raise ValueError(f"{name}: verified {marker_key} does not match info.{info_key}")

        features = info.get("features", {})
        pairs = self._resolve_camera_pairs(name, source, features)
        keys = list(dict.fromkeys(key for pair in pairs for key in pair))

        has_fem = bool(source.get("has_fem", False))
        fem_feature = str(source.get("fem_feature", "observation.fem_label")) if has_fem else None
        fem_arm_indices: list[int | None]
        if has_fem:
            if fem_feature not in features:
                raise ValueError(f"{name}: has_fem=True but {fem_feature!r} missing from info features")
            fem_shape = list(features[fem_feature].get("shape", []))
            if fem_shape == [2, 2, self.fem_dim]:
                raw_arms = source.get("fem_arm_indices")
                fem_arm_indices = list(range(len(pairs))) if raw_arms is None else list(raw_arms)
            elif fem_shape == [2, self.fem_dim]:
                fem_arm_indices = [None] * len(pairs)
            else:
                raise ValueError(
                    f"{name}: expected {fem_feature} shape [2, 2, {self.fem_dim}] or [2, {self.fem_dim}], got {fem_shape}"
                )
            if len(fem_arm_indices) != len(pairs):
                raise ValueError(f"{name}: fem_arm_indices length must match camera_pairs")
        else:
            fem_arm_indices = [None] * len(pairs)

        columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
        for key in keys:
            columns.extend(
                f"videos/{key}/{field}" for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
            )
        files = sorted((root / "meta/episodes").rglob("*.parquet"))
        if not files:
            if source_skip_incomplete and int(info.get("total_episodes", 0)) == 0:
                self._summary["skipped_sources"].append({"name": name, "root": str(root), "reason": "no_episodes"})
                return
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

        name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        rng = np.random.default_rng((seed + name_seed) % (2**64))
        order = rng.permutation(len(episodes)).tolist()
        if episode_limit is not None:
            order = order[:episode_limit]
        minimum = 1 + int(val_fraction > 0) + int(test_fraction > 0)
        if len(order) < minimum:
            if source_skip_incomplete:
                self._summary["skipped_sources"].append(
                    {"name": name, "root": str(root), "reason": f"insufficient_episodes:{len(order)}<{minimum}"}
                )
                return
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
        eligible = [row for row in selected if int(row["length"]) >= self.clip_span]
        start = self._ends[-1] if self._ends else 0

        def _make_cam(row: dict[str, Any], key: str, data_path: Path, episode_index: int, length: int) -> _CameraEpisode:
            prefix = f"videos/{key}/"
            video_path = root / info.get("video_path", _VIDEO_PATH).format(
                video_key=key,
                episode_index=episode_index,
                chunk_index=int(row[prefix + "chunk_index"]),
                file_index=int(row[prefix + "file_index"]),
            )
            offset, end_ts = float(row[prefix + "from_timestamp"]), float(row[prefix + "to_timestamp"])
            if not math.isfinite(offset) or not math.isfinite(end_ts) or offset < 0 or end_ts <= offset:
                raise ValueError(f"{name} episode {episode_index}: invalid video timestamp interval")
            return _CameraEpisode(name, episode_index, key, length, data_path, video_path, offset, end_ts, mode)

        for row in eligible:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            data_path = root / info.get("data_path", _DATA_PATH).format(
                episode_index=episode_index,
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            for (left_key, right_key), arm_index in zip(pairs, fem_arm_indices, strict=True):
                left_ep = _make_cam(row, left_key, data_path, episode_index, length)
                right_ep = _make_cam(row, right_key, data_path, episode_index, length)
                self._streams.append(
                    _PairedCameraEpisode(name, episode_index, left_ep, right_ep, has_fem, fem_feature, arm_index)
                )
                self._ends.append(
                    (self._ends[-1] if self._ends else 0) + (length - self.clip_span) // self.frame_stride + 1
                )

        end = self._ends[-1] if self._ends else 0
        if end > start:
            self.source_ranges.append((name, start, end))
        self._summary["sources"].append(
            {
                "name": name,
                "root": str(root),
                "info_file": str(info_file),
                "require_verified": source_require_verified,
                "verified": marker.is_file(),
                "camera_pairs": [list(pair) for pair in pairs],
                "has_fem": has_fem,
                "fem_feature": fem_feature,
                "timestamp_mode": mode,
                "episodes": len(eligible),
                "frames": sum(int(row["length"]) for row in eligible),
                "excluded_short_episodes": len(selected) - len(eligible),
                "samples": end - start,
                "available_episodes": len(episodes),
                "available_frames": int(info["total_frames"]),
            }
        )

    def _episode_fem_labels(self, record: _PairedCameraEpisode) -> np.ndarray:
        import pyarrow.parquet as pq

        assert record.fem_feature is not None
        cache_key = (str(record.left.data_path), record.episode_index, record.fem_feature)
        if cache_key not in self._fem_labels:
            path = str(record.left.data_path)
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
                table = parquet.read_row_group(group, columns=["episode_index", record.fem_feature])
                mask = table["episode_index"].to_numpy() == record.episode_index
                if mask.any():
                    arr = np.asarray(table[record.fem_feature].to_pylist(), dtype=np.float32)
                    pieces.append(arr[mask])
            labels = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 2, self.fem_dim), dtype=np.float32)
            if len(labels) != record.left.length:
                raise ValueError(
                    f"{path}: FEM label count {len(labels)} != episode length {record.left.length} "
                    f"for episode {record.episode_index}"
                )
            self._fem_labels[cache_key] = labels
            if len(self._fem_labels) > self._timestamp_cache_size:
                self._fem_labels.popitem(last=False)
        self._fem_labels.move_to_end(cache_key)
        return self._fem_labels[cache_key]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._ensure_process()
        record, frame_index = self._locate(index)
        left = self._decode_camera(record.left, frame_index)
        right = self._decode_camera(record.right, frame_index)
        if record.has_fem:
            label_frame = frame_index + (self.num_frames - 1) * self.clip_frame_step
            ep_labels = self._episode_fem_labels(record)
            frame_label = ep_labels[label_frame]
            if record.fem_arm_index is not None:
                frame_label = frame_label[record.fem_arm_index]
            fem_label = torch.from_numpy(np.asarray(frame_label, dtype=np.float32).copy())
            has_fem = torch.tensor(True, dtype=torch.bool)
        else:
            fem_label = torch.zeros((2, self.fem_dim), dtype=torch.float32)
            has_fem = torch.tensor(False, dtype=torch.bool)
        return {"left": left, "right": right, "fem_label": fem_label, "has_fem": has_fem}

    def close(self) -> None:
        super().close()
        self._fem_labels = OrderedDict()

    def __getstate__(self):
        state = super().__getstate__()
        state["_fem_labels"] = OrderedDict()
        return state


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

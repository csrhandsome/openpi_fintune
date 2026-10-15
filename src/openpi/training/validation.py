"""Validate physical action dimensions before loading weights or creating outputs."""

import json
import math

from openpi.training import config
from openpi.training.collector_frames import is_legacy_collector
from openpi.training.lerobot_utils import resolve_dataset_root


def validate_dataset(settings: config.TrainConfig) -> None:
    data = settings.data
    if not isinstance(data, config.LeRobotFrankaDataConfig):
        return
    root = resolve_dataset_root(data.repo_id)
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info["features"]
    layout = data.layout
    if is_legacy_collector(features):
        _validate_legacy_mp4(data, features, info_path)
    else:
        _validate_split_actions(data, features, info_path, layout)
    expected_fps = (settings.policy_metadata or {}).get("action_fps", 30.0)
    if abs(float(info["fps"]) - expected_fps) > 0.01:
        raise ValueError(f"Dataset fps differs from configured action_fps={expected_fps}: {info_path}")
    episodes = int(info["total_episodes"])
    if episodes < 1:
        raise ValueError(f"Dataset has no episodes: {info_path}")
    for episode in range(episodes):
        sync_path = root / f"episode_{episode:06d}.sync.json"
        if not sync_path.is_file():
            sync_path = root / "audio" / f"episode_{episode:06d}.sync.json"
        if not sync_path.is_file():
            raise FileNotFoundError(f"Missing action timing metadata: {sync_path}")
        sync = json.loads(sync_path.read_text(encoding="utf-8"))
        if not _valid_action_timing(sync):
            raise ValueError(f"Action timing mismatch in {sync_path}")


def _validate_split_actions(data, features, info_path, layout) -> None:
    action_keys = (f"action.{layout.target}", "action.gripper_position")
    for key in action_keys:
        if key not in features:
            raise ValueError(f"Missing action field {key} for action_space={data.action_space}: {info_path}")
    shapes = [features[key]["shape"] for key in action_keys]
    actual_dim = sum(math.prod(shape) for shape in shapes)
    if actual_dim != data.action_dim:
        raise ValueError(
            f"Action dimension mismatch: YAML data.action_dim={data.action_dim}, dataset={actual_dim} "
            f"({', '.join(f'{key}{shape}' for key, shape in zip(action_keys, shapes, strict=True))}); "
            f"action_space={data.action_space}: {info_path}"
        )
    expected_shapes = ([layout.motion_dim * layout.num_arms], [layout.gripper_dim * layout.num_arms])
    required = {
        f"{prefix}.{target}": shape
        for prefix in ("observation", "action")
        for target, shape in zip((layout.target, "gripper_position"), expected_shapes, strict=True)
    }
    for key, shape in required.items():
        if features.get(key, {}).get("shape") != shape:
            raise ValueError(f"Dataset field {key} must have shape {shape}: {info_path}")


def _validate_legacy_mp4(data, features, info_path) -> None:
    if data.action_space != "joint":
        raise ValueError(
            "This MP4 dataset stores joint targets in actions, not EE pose. "
            f"Set data.action_space to joint and data.action_dim to {features['actions']['shape'][-1]}: {info_path}"
        )
    expected = {
        "actions": [data.action_dim],
        "joint_position": [data.layout.motion_dim],
        "gripper_position": [data.layout.gripper_dim],
    }
    actual = {key: features[key]["shape"] for key in expected}
    if actual != expected:
        raise ValueError(f"Legacy MP4 field shapes must be {expected}, got {actual}: {info_path}")
    for key in ("exterior_image_1_left", "wrist_image_left"):
        if features[key].get("dtype") not in ("video", "image"):
            raise ValueError(f"Dataset field {key} must be an image or MP4 video: {info_path}")


def _valid_action_timing(sync: dict) -> bool:
    offset = sync.get("action_target_offset_frames")
    if offset is not None:
        return offset == 1
    if sync.get("action_semantics") == "next_camera_observation_measured_state":
        return True
    frames = sync.get("frame_records")
    return bool(frames) and isinstance(frames[0], dict) and "action_joint_position" in frames[0]

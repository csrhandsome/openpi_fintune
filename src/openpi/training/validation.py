"""Validate physical action dimensions before loading weights or creating outputs."""

import json
import math

from openpi.training import config
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
        offset = sync.get("action_target_offset_frames")
        valid_timing = (
            offset == 1
            if offset is not None
            else (sync.get("action_semantics") == "next_camera_observation_measured_state")
        )
        if not valid_timing:
            raise ValueError(f"Action timing mismatch in {sync_path}")

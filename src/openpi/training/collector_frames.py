"""Read collector datasets whose cameras are MP4 videos.

LeRobot decodes ``dtype: video`` frames. Retained ``*_v3_mp4`` datasets still use the
pre-unification column names and one 8D ``actions`` vector. Newer recordings already
use ``observation.*`` / ``action.*`` and need no renaming.
"""

from collections.abc import Sequence

import numpy as np


def is_legacy_collector(features: dict) -> bool:
    return (
        "exterior_image_1_left" in features
        and "wrist_image_left" in features
        and "joint_position" in features
        and "gripper_position" in features
        and "actions" in features
        and "observation.exterior_image" not in features
        and "action.joint_position" not in features
    )


def sequence_keys(features: dict, requested: Sequence[str]) -> tuple[tuple[str, ...], bool]:
    """Choose LeRobot delta-timestamp keys, splitting legacy ``actions`` after decode."""
    requested = tuple(requested)
    if all(key in features for key in requested):
        return requested, False
    if is_legacy_collector(features):
        if set(requested) - {"action.joint_position", "action.gripper_position"}:
            raise ValueError(
                f"Collector actions contain joint targets and gripper positions; cannot read {list(requested)}. "
                "Use data.action_space: joint and data.action_dim: 8 for the retained MP4 datasets."
            )
        return ("actions",), True
    missing = [key for key in requested if key not in features]
    raise ValueError(
        f"Dataset is missing {missing}. MP4 cameras are decoded from videos/; "
        f"expected action fields {list(requested)} or legacy actions/joint_position columns."
    )


class LegacyCollectorFrames:
    """Expose retained MP4 rows with the training field names."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.meta = dataset.meta
        self.root = dataset.root
        self.repo_id = dataset.repo_id
        joint_shape = dataset.meta.features["joint_position"]["shape"]
        if len(joint_shape) != 1:
            raise ValueError(f"joint_position must be a vector, got shape {joint_shape}")
        self._motion_dim = int(joint_shape[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["observation.exterior_image"] = item["exterior_image_1_left"]
        item["observation.wrist_image_left"] = item["wrist_image_left"]
        item["observation.joint_position"] = np.asarray(item["joint_position"], np.float32)
        item["observation.gripper_position"] = np.asarray(item["gripper_position"], np.float32).reshape(-1)
        actions = np.asarray(item["actions"], np.float32)
        if actions.shape[-1] <= self._motion_dim:
            raise ValueError(f"Legacy actions must include a gripper column, got shape {actions.shape}")
        item["action.joint_position"] = actions[..., : self._motion_dim]
        item["action.gripper_position"] = actions[..., self._motion_dim :]
        if "actions_is_pad" in item:
            item["action.joint_position_is_pad"] = item["actions_is_pad"]
            item["action.gripper_position_is_pad"] = item["actions_is_pad"]
        return item

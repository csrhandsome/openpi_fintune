"""Franka Cartesian observations and 6D pose plus gripper actions."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.droid_policy import _parse_image
from openpi.policies.panda_policy import with_panda_actions


def _wrap_radians(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


@dataclasses.dataclass(frozen=True)
class FrankaEEInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        data = with_panda_actions(data, "ee_pose")
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported Franka EE model type: {self.model_type}")
        ee_pose = np.asarray(data["observation.ee_pose"], dtype=np.float32).reshape(6)
        gripper = np.asarray(data["observation.gripper_position"], dtype=np.float32).reshape(1)
        base_image = _parse_image(data["observation.exterior_image"])
        wrist_image = _parse_image(data["observation.wrist_image_left"])
        result = {
            "state": np.concatenate([ee_pose, gripper]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 7:
                raise ValueError(f"Franka EE actions must be 7D, got {actions.shape}")
            result["actions"] = actions.copy()
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return result


@dataclasses.dataclass(frozen=True)
class EEDeltaActions(transforms.DataTransformFn):
    """Train position and wrapped RPY offsets; leave the gripper absolute."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"]).copy()
        state = np.asarray(data["state"])
        actions[..., :6] -= state[:6]
        actions[..., 3:6] = _wrap_radians(actions[..., 3:6])
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class EEAbsoluteActions(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"]).copy()
        state = np.asarray(data["state"])
        actions[..., :6] += state[:6]
        actions[..., 3:6] = _wrap_radians(actions[..., 3:6])
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class FrankaEEOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}

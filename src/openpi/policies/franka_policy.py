"""Configurable Franka action layouts, with one contiguous block per arm."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model
from openpi.policies.droid_policy import _parse_image


@dataclasses.dataclass(frozen=True)
class ActionLayout:
    action_space: str = "ee_delta"
    action_dim: int = 7
    motion_dim: int = 6
    gripper_dim: int = 1

    def __post_init__(self):
        if self.action_space not in ("joint", "ee", "ee_delta"):
            raise ValueError("data.action_space must be joint, ee or ee_delta")
        for key in ("action_dim", "motion_dim", "gripper_dim"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"data.{key} must be a positive integer")
        if self.action_space != "joint" and self.motion_dim != 6:
            raise ValueError("EE motion_dim must be 6 (XYZ + RPY)")
        if self.action_dim % self.arm_dim:
            raise ValueError("data.action_dim must equal (motion_dim + gripper_dim) * number of arms")

    @property
    def arm_dim(self):
        return self.motion_dim + self.gripper_dim

    @property
    def num_arms(self):
        return self.action_dim // self.arm_dim

    @property
    def target(self):
        return "joint_position" if self.action_space == "joint" else "ee_pose"

    def combine(self, motion, gripper):
        motion, gripper = np.asarray(motion, np.float32), np.asarray(gripper, np.float32)
        leading = motion.shape[:-1]
        if motion.shape[-1:] != (self.motion_dim * self.num_arms,):
            raise ValueError(f"{self.target} has invalid shape {motion.shape} for action_dim={self.action_dim}")
        if gripper.shape == leading and self.num_arms == self.gripper_dim == 1:
            gripper = gripper[..., None]
        if gripper.shape != (*leading, self.gripper_dim * self.num_arms):
            raise ValueError(f"gripper shape {gripper.shape} disagrees with action_dim={self.action_dim}")
        return np.concatenate(
            (
                motion.reshape(*leading, self.num_arms, self.motion_dim),
                gripper.reshape(*leading, self.num_arms, self.gripper_dim),
            ),
            axis=-1,
        ).reshape(*leading, self.action_dim)


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    model_type: model.ModelType
    layout: ActionLayout

    def __call__(self, data):
        if self.model_type not in (model.ModelType.PI0, model.ModelType.PI05):
            raise ValueError("Franka inputs require PI0/PI05")
        state = self.layout.combine(data[f"observation.{self.layout.target}"], data["observation.gripper_position"])
        base = _parse_image(data["observation.exterior_image"])
        wrist = _parse_image(data["observation.wrist_image_left"])
        result = {
            "state": state,
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        }
        key = f"action.{self.layout.target}"
        if key in data:
            result["actions"] = self.layout.combine(data[key], data["action.gripper_position"])
        elif "actions" in data:
            actions = np.asarray(data["actions"], np.float32)
            if actions.shape[-1] != self.layout.action_dim:
                raise ValueError(
                    f"Action dimension {actions.shape[-1]} differs from configured {self.layout.action_dim}"
                )
            result["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return result


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data):
        return {**data, "actions": np.asarray(data["actions"])[..., : self.action_dim].copy()}


@dataclasses.dataclass(frozen=True)
class EEPoseOffsets(transforms.DataTransformFn):
    layout: ActionLayout
    inverse: bool = False

    def __call__(self, data):
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"]).copy()
        blocks = actions.reshape(*actions.shape[:-1], self.layout.num_arms, self.layout.arm_dim)
        state = np.asarray(data["state"])[..., : self.layout.action_dim].reshape(
            self.layout.num_arms, self.layout.arm_dim
        )
        blocks[..., :6] += state[..., :6] * (1 if self.inverse else -1)
        blocks[..., 3:6] = (blocks[..., 3:6] + np.pi) % (2 * np.pi) - np.pi
        return {**data, "actions": actions}

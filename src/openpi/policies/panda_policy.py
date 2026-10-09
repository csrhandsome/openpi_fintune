"""Panda dataset and wire keys; combine split targets only at model input."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.droid_policy import DroidInputs


def with_panda_actions(data: dict, target: str) -> dict:
    pose_key = f"action.{target}"
    grip_key = "action.gripper_position"
    if pose_key not in data and grip_key not in data:
        return data
    pose = np.asarray(data[pose_key], dtype=np.float32)
    grip = np.asarray(data[grip_key], dtype=np.float32)
    if grip.shape == pose.shape[:-1]:
        grip = grip[..., None]
    dimension = 7 if target == "joint_position" else 6
    if pose.shape[-1:] != (dimension,) or grip.shape != (*pose.shape[:-1], 1):
        raise ValueError("Panda action targets must have matching chunk dimensions")
    return {**data, "actions": np.concatenate((pose, grip), axis=-1)}


@dataclasses.dataclass(frozen=True)
class PandaJointInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        data = with_panda_actions(data, "joint_position")
        # Reuse DROID's internal camera slots and joint/gripper representation.
        return DroidInputs(self.model_type)(
            {
                **data,
                "observation/exterior_image_1_left": data["observation.exterior_image"],
                "observation/wrist_image_left": data["observation.wrist_image_left"],
                "observation/joint_position": data["observation.joint_position"],
                "observation/gripper_position": data["observation.gripper_position"],
            }
        )

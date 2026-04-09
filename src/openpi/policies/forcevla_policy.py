import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_forcevla_example() -> dict:
    """Creates a random input example for the forcevla policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/gripper_image_left": np.random.randint(
            256, size=(4, 224, 224, 3), dtype=np.uint8
        ),
        "observation/gripper_image_right": np.random.randint(
            256, size=(4, 224, 224, 3), dtype=np.uint8
        ),
        "observation/joint_position": np.ones((7,), dtype=np.float32),
        "observation/gripper_position": np.ones((1,), dtype=np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if np.min(image) < 0.0:
            image = ((image + 1.0) / 2.0 * 255.0).astype(np.uint8)
        else:
            image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.ndim == 4 and image.shape[1] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


def _get_first_present(data: dict, keys: tuple[str, ...], *, required: bool = True):
    for key in keys:
        if key in data:
            return data[key]
    if required:
        raise KeyError(
            f"Missing required keys {keys}. Available keys: {tuple(data.keys())}"
        )
    return None


def _as_gripper_position(value) -> np.ndarray:
    gripper_position = np.asarray(value, dtype=np.float32)
    if gripper_position.ndim == 0:
        gripper_position = gripper_position[np.newaxis]
    return gripper_position


@dataclasses.dataclass(frozen=True)
class Forcevla_inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        joint_position = _get_first_present(
            data,
            ("observation/joint_position", "joint_position"),
            required=False,
        )
        gripper_position = _get_first_present(
            data,
            ("observation/gripper_position", "gripper_position"),
            required=False,
        )
        if joint_position is not None and gripper_position is not None:
            state_raw = np.concatenate(
                [
                    np.asarray(joint_position, dtype=np.float32),
                    _as_gripper_position(gripper_position),
                ],
                axis=-1,
            )
        else:
            state_raw = _get_first_present(data, ("observation/state", "state"))
        state = transforms.pad_to_dim(
            np.asarray(state_raw, dtype=np.float32), self.action_dim
        )

        base_image = _parse_image(
            _get_first_present(
                data,
                ("observation/exterior_image_1_left", "image", "observation.image"),
            )
        )
        left_wrist_image = _parse_image(
            _get_first_present(
                data,
                ("observation/wrist_image_left", "wrist_image_left", "wrist_image"),
            )
        )

        right_wrist_value = _get_first_present(
            data,
            ("observation/wrist_image_right", "wrist_image_right"),
            required=False,
        )
        has_right_wrist = right_wrist_value is not None
        right_wrist_image = (
            _parse_image(right_wrist_value)
            if has_right_wrist
            else np.zeros_like(base_image)
        )

        gripper_left_image = _parse_image(
            _get_first_present(
                data,
                ("observation/gripper_image_left", "gripper_image_left"),
            )
        )
        gripper_right_image = _parse_image(
            _get_first_present(
                data,
                ("observation/gripper_image_right", "gripper_image_right"),
            )
        )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
                "gripper_left_0_rgb": gripper_left_image,
                "gripper_right_0_rgb": gripper_right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.asarray(has_right_wrist or not mask_padding),
                "gripper_left_0_rgb": np.True_,
                "gripper_right_0_rgb": np.True_,
            },
        }

        actions_value = _get_first_present(data, ("actions", "action"), required=False)
        if actions_value is not None:
            actions = transforms.pad_to_dim(
                np.asarray(actions_value, dtype=np.float32), self.action_dim
            )
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class Forcevla_outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}

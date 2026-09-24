import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array-like structures used in model IO. This includes `jax.ShapeDtypeStruct`
# because `inputs_spec` returns shape-only placeholders with this type.
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray | jax.ShapeDtypeStruct)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_AUDIO = "pi0_audio"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "audio": float32[*b, ...],  # Optional legacy audio alias for ambient_audio
#     "audio_mask": bool[*b],  # Optional legacy audio validity mask
#     "ambient_audio": float32[*b, ...],  # Optional ambient audio representation
#     "ambient_audio_mask": bool[*b],  # Optional ambient audio validity mask
#     "instruction_audio": float32[*b, ...],  # Optional VAD-selected instruction audio representation
#     "instruction_audio_mask": bool[*b],  # Optional instruction audio validity mask
#     "route_label": int32[*b],  # Optional route label id for audio supervision
#     "route_label_mask": bool[*b],  # Optional route-label validity mask
#     "action_loss_mask": bool[*b],  # Optional action suffix loss validity mask
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]
    # Optional legacy audio representation. This aliases `ambient_audio` for backward
    # compatibility with single-slot audio code paths.
    audio: at.Float[ArrayT, "..."] | None = None
    # Optional legacy audio validity mask.
    audio_mask: at.Bool[ArrayT, "*b"] | None = None
    # Optional ambient audio representation (waveform window or time-sequence features).
    ambient_audio: at.Float[ArrayT, "..."] | None = None
    # Optional ambient audio validity mask.
    ambient_audio_mask: at.Bool[ArrayT, "*b"] | None = None
    # Optional instruction audio representation (for example VAD-selected speech).
    instruction_audio: at.Float[ArrayT, "..."] | None = None
    # Optional instruction audio validity mask.
    instruction_audio_mask: at.Bool[ArrayT, "*b"] | None = None
    # Optional route label id, e.g. straight / detour / none after transform mapping.
    route_label: at.Int[ArrayT, "*b"] | None = None
    # Optional route-label validity mask. Invalid labels are ignored by auxiliary losses.
    route_label_mask: at.Bool[ArrayT, "*b"] | None = None
    # Optional action suffix loss validity mask. False samples keep auxiliary losses but skip action BC.
    action_loss_mask: at.Bool[ArrayT, "*b"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    # Dialogue answer tokens use a causal prefix-LM mask. These fields are
    # consumed by Pi0Audio's optional Talker text-loss branch.
    dialogue_token_ids: at.Int[ArrayT, "*b l"] | None = None
    dialogue_token_mask: at.Bool[ArrayT, "*b l"] | None = None
    dialogue_token_ar_mask: at.Bool[ArrayT, "*b l"] | None = None
    dialogue_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        dialogue_fields = (
            "dialogue_token_ids",
            "dialogue_token_mask",
            "dialogue_token_ar_mask",
            "dialogue_token_loss_mask",
        )
        present_dialogue_fields = [field in data for field in dialogue_fields]
        if any(present_dialogue_fields) and not all(present_dialogue_fields):
            raise ValueError(f"Dialogue token fields must be provided together: {dialogue_fields}")
        if "audio_mask" in data and "audio" not in data and "ambient_audio" not in data:
            raise ValueError("audio_mask cannot be provided without audio or ambient_audio.")
        if "ambient_audio_mask" in data and "ambient_audio" not in data and "audio" not in data:
            raise ValueError("ambient_audio_mask cannot be provided without ambient_audio or audio.")
        if "instruction_audio_mask" in data and "instruction_audio" not in data:
            raise ValueError("instruction_audio_mask cannot be provided without instruction_audio.")
        if "route_label_mask" in data and "route_label" not in data:
            raise ValueError("route_label_mask cannot be provided without route_label.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0

        def _default_audio_mask(audio_value, audio_mask_value):
            if audio_value is None or audio_mask_value is not None:
                return audio_mask_value
            audio_batch_shape = audio_value.shape[:-2] if audio_value.ndim >= 3 else audio_value.shape[:-1]
            if isinstance(audio_value, torch.Tensor):
                return torch.ones(audio_batch_shape, dtype=torch.bool, device=audio_value.device)
            return np.ones(audio_batch_shape, dtype=np.bool_)

        ambient_audio = data.get("ambient_audio", data.get("audio"))
        ambient_audio_mask = data.get("ambient_audio_mask", data.get("audio_mask"))
        ambient_audio_mask = _default_audio_mask(ambient_audio, ambient_audio_mask)

        instruction_audio = data.get("instruction_audio")
        instruction_audio_mask = _default_audio_mask(
            instruction_audio,
            data.get("instruction_audio_mask"),
        )

        route_label = data.get("route_label")
        route_label_mask = data.get("route_label_mask")
        if route_label is not None and route_label_mask is None:
            if isinstance(route_label, torch.Tensor):
                route_label_mask = torch.ones(route_label.shape, dtype=torch.bool, device=route_label.device)
            else:
                route_label_mask = np.ones(np.asarray(route_label).shape, dtype=np.bool_)

        audio = ambient_audio if ambient_audio is not None else data.get("audio")
        audio_mask = ambient_audio_mask if ambient_audio_mask is not None else data.get("audio_mask")
        audio_mask = _default_audio_mask(audio, audio_mask)

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            audio=audio,
            audio_mask=audio_mask,
            ambient_audio=ambient_audio,
            ambient_audio_mask=ambient_audio_mask,
            instruction_audio=instruction_audio,
            instruction_audio_mask=instruction_audio_mask,
            route_label=route_label,
            route_label_mask=route_label_mask,
            action_loss_mask=data.get("action_loss_mask"),
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            dialogue_token_ids=data.get("dialogue_token_ids"),
            dialogue_token_mask=data.get("dialogue_token_mask"),
            dialogue_token_ar_mask=data.get("dialogue_token_ar_mask"),
            dialogue_token_loss_mask=data.get("dialogue_token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        leading_shape = image.shape[:-3]
        flat_image = jnp.reshape(image, (-1, *image.shape[-3:]))
        if flat_image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {flat_image.shape[1:3]} to {image_resolution}")
            flat_image = image_tools.resize_with_pad(flat_image, *image_resolution)

        if train:
            flat_image = flat_image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key and "gripper" not in key:
                height, width = flat_image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, flat_image.shape[0])
            flat_image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, flat_image)

            flat_image = flat_image * 2.0 - 1.0

        out_images[key] = jnp.reshape(
            flat_image,
            (*leading_shape, image_resolution[0], image_resolution[1], flat_image.shape[-1]),
        )

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        audio=observation.audio if observation.audio is not None else observation.ambient_audio,
        audio_mask=(observation.audio_mask if observation.audio_mask is not None else observation.ambient_audio_mask),
        ambient_audio=observation.ambient_audio,
        ambient_audio_mask=observation.ambient_audio_mask,
        instruction_audio=observation.instruction_audio,
        instruction_audio_mask=observation.instruction_audio_mask,
        route_label=observation.route_label,
        route_label_mask=observation.route_label_mask,
        action_loss_mask=observation.action_loss_mask,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        dialogue_token_ids=observation.dialogue_token_ids,
        dialogue_token_mask=observation.dialogue_token_mask,
        dialogue_token_ar_mask=observation.dialogue_token_ar_mask,
        dialogue_token_loss_mask=observation.dialogue_token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)

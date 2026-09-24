from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

_RTC_ACTION_PREFIX_KEYS = ("action_prefix", "rtc/action_prefix", "rtc_action_prefix")
_RTC_DELAY_KEYS = ("rtc_delay", "rtc/delay", "action_prefix_delay", "delay")


def _first_present_flat(data: dict, keys: tuple[str, ...]):
    flat = flax.traverse_util.flatten_dict(data, sep="/")
    for key in keys:
        if key in flat:
            return flat[key]
    return None


def _contains_flat(data: dict, keys: tuple[str, ...]) -> bool:
    flat = flax.traverse_util.flatten_dict(data, sep="/")
    return any(key in flat for key in keys)


def _infer_delay_from_prefix(prefix) -> np.ndarray:
    prefix = np.asarray(prefix)
    if prefix.ndim < 2:
        raise ValueError(f"RTC action_prefix must have shape [H, D] or [B, H, D], got {prefix.shape}")
    return np.asarray(prefix.shape[-2], dtype=np.int32)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._predict_route_logits = None
        self._generate_text_tokens = None
        self._dialogue_tokenizer = None
        route_label_names = getattr(model, "route_label_names", None)
        self._route_label_names = tuple(route_label_names) if route_label_names is not None else None

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            if getattr(model, "use_route_head", False):
                self._predict_route_logits = nnx_utils.module_jit(model.predict_route_logits)
            if getattr(model, "enable_text_loss", False) and hasattr(model, "generate_text_tokens"):
                # JIT the Talker decode loop. The eager Python loop otherwise
                # recompiles a new KV-cache shape on every token (48 compiles
                # on first request) and crashes XLA/glibc on WSL.
                self._generate_text_tokens = nnx_utils.module_jit(model.generate_text_tokens)
                self._dialogue_tokenizer = _tokenizer.PaligemmaTokenizer(model.max_token_len)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        dialogue_generate_value = inputs.get("dialogue_generate", False)
        dialogue_generate = bool(np.asarray(dialogue_generate_value).reshape(()).item())
        rtc_action_prefix = _first_present_flat(inputs, _RTC_ACTION_PREFIX_KEYS)
        rtc_delay = _first_present_flat(inputs, _RTC_DELAY_KEYS)
        if rtc_action_prefix is not None:
            if not _contains_flat(inputs, ("actions", "action")):
                inputs["actions"] = rtc_action_prefix
            if rtc_delay is None:
                rtc_delay = _infer_delay_from_prefix(rtc_action_prefix)

        inputs = self._input_transform(inputs)
        transformed_action_prefix = inputs.get("actions") if rtc_action_prefix is not None else None
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if transformed_action_prefix is not None:
            if self._is_pytorch_model:
                raise NotImplementedError("RTC action_prefix inference is only wired for JAX models.")
            sample_kwargs["action_prefix"] = jnp.asarray(transformed_action_prefix)[None, ...]
            sample_kwargs["delay"] = jnp.asarray(rtc_delay, dtype=jnp.int32)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        route_outputs = {}
        if dialogue_generate:
            if self._is_pytorch_model or self._generate_text_tokens is None or self._dialogue_tokenizer is None:
                raise ValueError(
                    "This policy does not expose a Pi0Audio Talker decoder. "
                    "Load a checkpoint configured with enable_text_loss=True."
                )
            text_tokens = np.asarray(self._generate_text_tokens(observation)[0])
            model_time = time.monotonic() - start_time
            return {
                "text_tokens": text_tokens,
                "text": self._dialogue_tokenizer.decode(text_tokens),
                "policy_timing": {"infer_ms": model_time * 1000},
            }

        if self._predict_route_logits is not None:
            route_logits = self._predict_route_logits(observation)
            route_probabilities = jax.nn.softmax(route_logits, axis=-1)
            route_outputs = {
                "route_logits": route_logits,
                "route_probabilities": route_probabilities,
                "route_label": jnp.argmax(route_probabilities, axis=-1),
            }
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            route_outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), route_outputs)

        outputs = self._output_transform(outputs)
        outputs.update(route_outputs)
        if self._route_label_names is not None and "route_label" in outputs:
            route_index = int(np.asarray(outputs["route_label"]).reshape(()).item())
            if 0 <= route_index < len(self._route_label_names):
                outputs["route_label_name"] = self._route_label_names[route_index]
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

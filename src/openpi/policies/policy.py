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
        self._rl_readout = None

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if self._rl_readout is not None and ("monotonic_ts" not in obs or "O_T_TCP" not in obs):
            raise ValueError("RL-token requests require client monotonic_ts and measured O_T_TCP")
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
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
        if self._is_pytorch_model:
            with torch.no_grad():
                if self._rl_readout is None:
                    actions = self._sample_actions(observation, **sample_kwargs)
                else:
                    actions, prefix, prefix_mask = self._model.sample_actions_with_prefix(
                        observation, **sample_kwargs
                    )
                    if list(prefix.shape[1:]) != self._metadata["prefix_shape"]:
                        raise ValueError("Readout prefix shape differs from training protocol")
                    rl_tokens = self._rl_readout.encode(prefix, prefix_mask)
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if self._rl_readout is not None:
            physical_actions = np.asarray(outputs["actions"])
            action_space = self._metadata.get("action_space", "joint")
            if action_space not in ("ee", "joint"):
                raise ValueError(f"Unsupported RL-token action space: {action_space}")
            physical_action_dim = 7 if action_space == "ee" else 8
            if physical_actions.ndim != 2 or physical_actions.shape[1] != physical_action_dim:
                raise ValueError(f"Physical VLA {action_space} actions must have shape [H,{physical_action_dim}]")
            outputs["z_rlt"] = rl_tokens[0].detach().float().cpu().numpy().reshape(-1)
            if action_space == "joint":
                outputs["ref_q"] = physical_actions[:, :7].copy()
            outputs["ref_chunk"] = physical_actions.copy()
            outputs["ref_obs_ts"] = obs.get("monotonic_ts")
            # Wall-clock timestamp is for logging. Age checks use the client's
            # echoed monotonic observation timestamp, never a remote clock.
            outputs["ref_generation_ts"] = time.time()
            outputs["ref_anchor_TCP"] = obs.get("O_T_TCP")
            for key in ("episode_id", "step_id", "external_frame_id"):
                if key in obs:
                    outputs[key] = obs[key]
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def attach_rl_readout(self, readout, *, metadata: dict[str, Any]) -> None:
        """Enable shared-prefix RL-token inference for a frozen PyTorch Pi0."""
        if not self._is_pytorch_model:
            raise TypeError("RL-token readout requires a PyTorch Pi0 policy")
        if readout.config.input_dim != self._model.llm.configs[0].width:
            raise ValueError("Readout input dimension does not match VLA prefix width")
        self._rl_readout = readout.to(self._pytorch_device).eval()
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad = False
        for parameter in self._rl_readout.parameters():
            parameter.requires_grad = False
        self._metadata.update(metadata)

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

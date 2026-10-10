"""Validate raw episodes and build windows without crossing an episode boundary."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from openpi.rlt.rewards import EpisodeRewards
from openpi.rlt.rewards import TerminalReward

SOURCES = {"BASE", "RL", "HUMAN"}


@dataclass(frozen=True)
class ReplaySpec:
    z_dim: int
    proprio_dim: int
    horizon: int
    chunk_len: int
    action_dim: int
    action_space: str = "ee"
    fps: float = 30.0
    grip_dim: int = 0

    def __post_init__(self):
        if any(
            type(x) is not int or x <= 0
            for x in (self.z_dim, self.proprio_dim, self.horizon, self.chunk_len, self.action_dim)
        ):
            raise ValueError("Replay dimensions must be positive integers")
        if self.chunk_len > self.horizon or self.action_space not in {"ee", "joint"}:
            raise ValueError("Invalid execution horizon or action space")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Replay fps must be positive")
        if type(self.grip_dim) is not int or self.grip_dim < 0:
            raise ValueError("grip_dim must be a nonnegative integer")


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


@dataclass(frozen=True)
class RLTTransition:
    episode_id: str
    step_id: int
    z_rlt: np.ndarray
    proprio: np.ndarray
    ref_chunk: np.ndarray
    action_chunk: np.ndarray
    rewards: np.ndarray
    executed_mask: np.ndarray
    done: bool
    next_z_rlt: np.ndarray
    next_proprio: np.ndarray
    next_ref_chunk: np.ndarray
    source_chunk: np.ndarray
    success: bool
    z_grip: np.ndarray | None = None
    next_z_grip: np.ndarray | None = None

    @property
    def source(self):
        sources = set(self.source_chunk[self.executed_mask].tolist())
        return next(iter(sources)) if len(sources) == 1 else "MIXED"

    def validate(self, spec: ReplaySpec):
        for name in ("z_grip", "next_z_grip"):
            value = getattr(self, name)
            if spec.grip_dim:
                finite_array(value, (spec.grip_dim,), name)
            elif value is not None and np.asarray(value).size:
                raise ValueError("Force-disabled replay cannot contain grip features")
        for name, shape in (
            ("z_rlt", (spec.z_dim,)),
            ("next_z_rlt", (spec.z_dim,)),
            ("proprio", (spec.proprio_dim,)),
            ("next_proprio", (spec.proprio_dim,)),
            ("ref_chunk", (spec.horizon, spec.action_dim)),
            ("next_ref_chunk", (spec.horizon, spec.action_dim)),
            ("action_chunk", (spec.chunk_len, spec.action_dim)),
            ("rewards", (spec.chunk_len,)),
        ):
            finite_array(getattr(self, name), shape, name)
        mask = np.asarray(self.executed_mask)
        if mask.dtype != np.bool_ or mask.shape != (spec.chunk_len,):
            raise ValueError("executed_mask must be a boolean execution prefix")
        count = int(mask.sum())
        if not count or not mask[:count].all() or mask[count:].any():
            raise ValueError("executed_mask must be a nonempty contiguous prefix")
        if count < spec.chunk_len and not self.done:
            raise ValueError("A partial window must terminate")
        if np.any(self.rewards[~mask] != 0):
            raise ValueError("Padding cannot carry rewards")
        if self.source_chunk.shape != mask.shape or not set(self.source_chunk[mask].tolist()) <= SOURCES:
            raise ValueError("Unknown action source")
        if not self.episode_id or type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("Invalid episode/step identity")
        if type(self.done) is not bool or type(self.success) is not bool:
            raise ValueError("done and success must be explicit booleans")


def window_starts(count: int, *, chunk_len: int, stride: int) -> list[int]:
    if type(stride) is not int or stride <= 0 or count <= 0:
        raise ValueError("Window construction requires actual steps and a positive stride")
    return sorted(set(range(0, max(count - chunk_len + 1, 1), stride)) | {max(0, count - chunk_len)})


def build_episode_transitions(
    episode_id: str,
    features: list[dict[str, Any] | None],
    actions: np.ndarray,
    sources: list[str],
    *,
    spec: ReplaySpec,
    success: bool,
    stride: int = 2,
    reward: EpisodeRewards | None = None,
) -> list[RLTTransition]:
    """N actual actions require N+1 states. A final aligned window preserves terminal reward."""
    if type(stride) is not int or stride <= 0:
        raise ValueError("stride must be a positive integer")
    count = len(actions)
    finite_array(actions, (count, spec.action_dim), "actions")
    if len(features) != count + 1 or len(sources) != count or not set(sources) <= SOURCES:
        raise ValueError("An episode needs one more state than actions, with explicit sources")
    rewards = (reward or TerminalReward()).generate(count, success=success)
    finite_array(rewards, (count,), "episode rewards")
    starts = window_starts(count, chunk_len=spec.chunk_len, stride=stride)
    required = set(starts) | {min(start + spec.chunk_len, count) for start in starts}
    for index in required:
        feature = features[index]
        if feature is None:
            raise ValueError("Replay window lacks a current or next-state feature")
        for key, shape in (
            ("z_rlt", (spec.z_dim,)),
            ("proprio", (spec.proprio_dim,)),
            ("ref_chunk", (spec.horizon, spec.action_dim)),
        ):
            finite_array(feature[key], shape, key)
        if spec.grip_dim:
            finite_array(feature.get("z_grip"), (spec.grip_dim,), "z_grip")
    records = []
    for start in starts:
        stop = min(start + spec.chunk_len, count)
        valid = stop - start
        mask = np.arange(spec.chunk_len) < valid
        executed = np.repeat(np.asarray(actions[stop - 1])[None], spec.chunk_len, axis=0)
        executed[:valid] = actions[start:stop]
        window_rewards = np.zeros(spec.chunk_len, dtype=np.float32)
        window_rewards[:valid] = rewards[start:stop]
        window_sources = np.full(spec.chunk_len, "PAD", dtype="U5")
        window_sources[:valid] = sources[start:stop]
        current, next_state = features[start], features[stop]
        record = RLTTransition(
            episode_id,
            start,
            current["z_rlt"],
            current["proprio"],
            current["ref_chunk"],
            executed,
            window_rewards,
            mask,
            stop == count,
            next_state["z_rlt"],
            next_state["proprio"],
            next_state["ref_chunk"],
            window_sources,
            success,
            current.get("z_grip") if spec.grip_dim else None,
            next_state.get("z_grip") if spec.grip_dim else None,
        )
        record.validate(spec)
        records.append(record)
    return records

"""Validated executed-transition journal with episode and intervention boundaries."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from openpi.force_rlt.networks import NetworkConfig


@dataclass(frozen=True)
class State:
    z_rlt: np.ndarray
    z_grip: np.ndarray
    q: np.ndarray
    ref_q: np.ndarray


@dataclass(frozen=True)
class Transition:
    episode_id: str
    step_id: int
    state: State
    proposed_q: np.ndarray
    executed_q: np.ndarray
    executed_mask: np.ndarray
    rewards: np.ndarray
    done: bool
    next_state: State
    source: str
    actor_version: str
    safety_flags: tuple[str, ...]
    independent_success: bool
    intervention_flag: bool
    observation_meta: dict

    def validate(self, cfg: NetworkConfig):
        for state in (self.state, self.next_state):
            expected = (
                (state.z_rlt, (cfg.z_rlt_dim,)),
                (state.z_grip, (cfg.z_grip_dim,)),
                (state.q, (7,)),
                (state.ref_q, (cfg.horizon, cfg.action_dim)),
            )
            for value, shape in expected:
                if np.asarray(value).shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"State field must be finite with shape {shape}")
        for value, shape in (
            (self.proposed_q, (cfg.horizon, cfg.action_dim)),
            (self.executed_q, (cfg.executed_steps, cfg.action_dim)),
            (self.rewards, (cfg.executed_steps,)),
            (self.executed_mask, (cfg.executed_steps,)),
        ):
            if np.asarray(value).shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Transition field must be finite with shape {shape}")
        valid = np.asarray(self.executed_mask, dtype=bool)
        n = int(valid.sum())
        if n < 1 or not valid[:n].all() or valid[n:].any():
            raise ValueError("Executed steps must form a nonempty contiguous prefix")
        if n < cfg.executed_steps and not self.done:
            raise ValueError("A short execution must terminate the transition")
        if n < cfg.executed_steps:
            if np.any(np.asarray(self.rewards)[n:] != 0):
                raise ValueError("Unexecuted steps cannot carry reward")
            if not np.allclose(np.asarray(self.executed_q)[n:], np.asarray(self.executed_q)[n - 1]):
                raise ValueError("Unexecuted targets must repeat the last executed target")
        if self.source not in {"BASE", "RL", "HUMAN"}:
            raise ValueError("Invalid action source")
        if self.intervention_flag != (self.source == "HUMAN"):
            raise ValueError("Human intervention must be isolated by source")
        if not self.episode_id or self.step_id < 0:
            raise ValueError("Episode/step ID missing")


class ReplayJournal:
    def __init__(self, path: str | Path, cfg: NetworkConfig):
        self.path = Path(path)
        self.cfg = cfg
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last: Transition | None = None

    def append(self, transition: Transition):
        transition.validate(self.cfg)
        previous = self._last
        if (
            previous
            and previous.episode_id == transition.episode_id
            and (previous.done or previous.intervention_flag or transition.step_id != previous.step_id + 1)
        ):
            raise ValueError("Episode crossed a terminal or intervention boundary")
        payload = json.dumps(asdict(transition), default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload + "\n")
        self._last = transition

    def read_all(self) -> list[Transition]:
        records = []
        if not self.path.exists():
            return records
        for line in self.path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            value.pop("artifact_hashes", None)
            for key in ("state", "next_state"):
                value[key] = State(**{name: np.asarray(array, dtype=np.float32) for name, array in value[key].items()})
            for key in ("proposed_q", "executed_q", "rewards"):
                value[key] = np.asarray(value[key], dtype=np.float32)
            value["executed_mask"] = np.asarray(value["executed_mask"], dtype=bool)
            value["safety_flags"] = tuple(value["safety_flags"])
            record = Transition(**value)
            record.validate(self.cfg)
            records.append(record)
        return records

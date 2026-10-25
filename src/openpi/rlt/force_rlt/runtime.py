"""Contact handoff, actor selection and transition construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from openpi.rlt.force_rlt.action_adapter import FrankaSafetyProjector
from openpi.rlt.force_rlt.action_adapter import Projection
from openpi.rlt.force_rlt.networks import ChunkActor
from openpi.rlt.force_rlt.observation import GripFeatures
from openpi.rlt.force_rlt.replay import State
from openpi.rlt.force_rlt.replay import Transition


@dataclass(frozen=True)
class HandoffConfig:
    max_visual_error_m: float
    max_measured_tcp_error_m: float
    confirm_frames: int

    def __post_init__(self):
        if min(self.max_visual_error_m, self.max_measured_tcp_error_m) <= 0 or self.confirm_frames <= 0:
            raise ValueError("Handoff thresholds and confirmation count must be configured")


class HandoffGate:
    def __init__(self, config: HandoffConfig):
        self.config = config
        self._count = 0
        self._active = False

    def update(self, *, visual_error_m: float, measured_tcp_error_m: float, grasp_closed: bool) -> bool:
        if self._active:
            return True
        valid = (
            grasp_closed
            and np.isfinite(visual_error_m)
            and np.isfinite(measured_tcp_error_m)
            and visual_error_m <= self.config.max_visual_error_m
            and measured_tcp_error_m <= self.config.max_measured_tcp_error_m
        )
        self._count = self._count + 1 if valid else 0
        self._active = self._count >= self.config.confirm_frames
        return self._active


class ContactPolicy:
    """A0 uses VLA reference; B/C use separately trained actors."""

    def __init__(self, group: str, projector: FrankaSafetyProjector, actor: ChunkActor | None = None):
        if group not in {"A0", "B", "C"} or (group == "A0") != (actor is None):
            raise ValueError("A0 has no actor; B/C require separately trained actors")
        self.group = group
        self.projector = projector
        self.actor = actor.eval() if actor is not None else None

    def propose(self, state: State, grip: GripFeatures | None) -> np.ndarray:
        if self.group == "A0":
            return np.asarray(state.ref_q, dtype=np.float64)
        if self.group == "C" and (grip is None or not grip.valid_mask.all()):
            raise ValueError("C requires a complete synchronized four-frame history")
        with torch.no_grad():
            device = next(self.actor.parameters()).device
            z_grip = np.zeros_like(state.z_grip) if self.group == "B" else state.z_grip
            inputs = [
                torch.as_tensor(value, dtype=torch.float32, device=device)[None]
                for value in (state.z_rlt, z_grip, state.q, state.ref_q)
            ]
            return self.actor.sample(*inputs, deterministic=True)[0].cpu().numpy()

    def decide(
        self,
        state: State,
        grip: GripFeatures | None,
        *,
        current_tcp: np.ndarray,
        prior_velocity: np.ndarray,
        prior_tcp_velocity_xyz: np.ndarray,
        prior_tcp_angular_speed: float,
        ref_anchor_tcp: np.ndarray,
        ref_obs_ts: float,
        ref_generation_ts: float,
        now: float,
    ) -> tuple[np.ndarray, Projection]:
        hold = np.repeat(np.asarray(state.q)[None], self.projector.execute, axis=0)
        try:
            proposed = self.propose(state, grip)
        except ValueError as error:
            return np.asarray(state.ref_q), Projection(executed_q=hold, accepted=False, flags=(str(error),))
        projected = self.projector.project(
            proposed,
            state.ref_q,
            current_q=state.q,
            current_tcp=current_tcp,
            prior_velocity=prior_velocity,
            prior_tcp_velocity_xyz=prior_tcp_velocity_xyz,
            prior_tcp_angular_speed=prior_tcp_angular_speed,
            ref_anchor_tcp=ref_anchor_tcp,
            ref_obs_ts=ref_obs_ts,
            ref_generation_ts=ref_generation_ts,
            now=now,
        )
        return proposed, projected


def make_transition(
    *,
    episode_id: str,
    step_id: int,
    state: State,
    proposal: np.ndarray,
    commanded: np.ndarray,
    confirmed_steps: int,
    rewards: np.ndarray,
    done: bool,
    next_state: State,
    source: str,
    actor_version: str,
    safety_flags: tuple[str, ...],
    independent_success: bool,
    intervention_flag: bool,
    observation_meta: dict,
    source_chunk: np.ndarray | None = None,
) -> Transition:
    """Journal only acknowledged targets; pad the unused tail with the last target."""
    commanded = np.asarray(commanded, dtype=np.float32)
    if not 0 < confirmed_steps <= len(commanded):
        raise ValueError("At least one executed target must be confirmed")
    executed = commanded.copy()
    executed[confirmed_steps:] = executed[confirmed_steps - 1]
    reward_array = np.asarray(rewards, dtype=np.float32).copy()
    reward_array[confirmed_steps:] = 0
    mask = np.arange(len(commanded)) < confirmed_steps
    return Transition(
        episode_id,
        step_id,
        state,
        np.asarray(proposal, dtype=np.float32),
        executed,
        mask,
        reward_array,
        done,
        next_state,
        source,
        actor_version,
        safety_flags,
        independent_success,
        intervention_flag,
        observation_meta,
        source_chunk,
    )

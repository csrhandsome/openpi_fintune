"""Franka joint-block adaptation with independent fail-closed TCP checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Kinematics(Protocol):
    def fk(self, q: np.ndarray) -> np.ndarray: ...
    def ik(self, pose: np.ndarray, seed: np.ndarray) -> np.ndarray | None: ...


@dataclass(frozen=True)
class SafetyLimits:
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    max_joint_step: np.ndarray
    max_joint_velocity: np.ndarray
    max_joint_acceleration: np.ndarray
    max_tcp_step_xyz: np.ndarray
    max_tcp_velocity_xyz: np.ndarray
    max_tcp_acceleration_xyz: np.ndarray
    max_tcp_angle_step_rad: float
    max_tcp_angular_velocity_rad_s: float
    max_tcp_angular_acceleration_rad_s2: float
    workspace_lower_xyz: np.ndarray
    workspace_upper_xyz: np.ndarray
    reference_radius: np.ndarray
    control_hz: float
    max_ref_age_s: float

    def __post_init__(self):
        vector7 = (
            "joint_lower",
            "joint_upper",
            "max_joint_step",
            "max_joint_velocity",
            "max_joint_acceleration",
            "reference_radius",
        )
        vector3 = (
            "max_tcp_step_xyz",
            "max_tcp_velocity_xyz",
            "max_tcp_acceleration_xyz",
            "workspace_lower_xyz",
            "workspace_upper_xyz",
        )
        for name in vector7:
            value = np.asarray(getattr(self, name))
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must have seven values")
        for name in vector3:
            value = np.asarray(getattr(self, name))
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must have three values")
        for name in (
            "max_joint_step",
            "max_joint_velocity",
            "max_joint_acceleration",
            "reference_radius",
            "max_tcp_step_xyz",
            "max_tcp_velocity_xyz",
            "max_tcp_acceleration_xyz",
        ):
            if np.any(np.asarray(getattr(self, name)) <= 0):
                raise ValueError(f"{name} must be physically configured and positive")
        scalars = (
            self.control_hz,
            self.max_ref_age_s,
            self.max_tcp_angle_step_rad,
            self.max_tcp_angular_velocity_rad_s,
            self.max_tcp_angular_acceleration_rad_s2,
        )
        if not np.isfinite(scalars).all() or min(scalars) <= 0:
            raise ValueError("Frequency, reference age and TCP angle limit must be positive")
        if np.any(self.joint_lower >= self.joint_upper) or np.any(self.workspace_lower_xyz >= self.workspace_upper_xyz):
            raise ValueError("Invalid joint or workspace bounds")


@dataclass(frozen=True)
class Projection:
    executed_q: np.ndarray
    accepted: bool
    flags: tuple[str, ...]


def _reject(hold: np.ndarray, flags: list[str], reason: str) -> Projection:
    return Projection(executed_q=hold, accepted=False, flags=(*flags, reason))


def _pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("TCP pose must be a finite 4x4 homogeneous matrix")
    return pose


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return float(np.arccos(np.clip(cosine, -1, 1)))


def reanchor_reference(
    ref_q: np.ndarray,
    ref_anchor_tcp: np.ndarray,
    current_tcp: np.ndarray,
    current_q: np.ndarray,
    kinematics: Kinematics,
) -> np.ndarray:
    """Retarget the recorded TCP-relative VLA trajectory from measured TCP."""
    ref_q = np.asarray(ref_q, dtype=np.float64)
    if ref_q.ndim != 2 or ref_q.shape[1] != 7:
        raise ValueError("ref_q must be [H,7]")
    old_anchor = _pose(ref_anchor_tcp)
    new_anchor = _pose(current_tcp)
    seed = np.asarray(current_q, dtype=np.float64)
    result = []
    old_inverse = np.linalg.inv(old_anchor)
    for target in ref_q:
        relative = old_inverse @ _pose(kinematics.fk(target))
        solution = kinematics.ik(new_anchor @ relative, seed)
        if solution is None:
            raise ValueError("Reference reanchoring IK failed")
        seed = np.asarray(solution, dtype=np.float64)
        if seed.shape != (7,) or not np.isfinite(seed).all():
            raise ValueError("Reference reanchoring IK returned invalid joints")
        result.append(seed)
    return np.stack(result)


class FrankaSafetyProjector:
    def __init__(
        self, limits: SafetyLimits, kinematics: Kinematics, *, prediction_horizon: int, execution_horizon: int
    ):
        if prediction_horizon <= 0 or not 0 < execution_horizon <= prediction_horizon:
            raise ValueError("Require 0 < E <= H")
        self.limits = limits
        self.kinematics = kinematics
        self.horizon = prediction_horizon
        self.execute = execution_horizon

    def project(
        self,
        proposed_q: np.ndarray,
        ref_q: np.ndarray,
        *,
        current_q: np.ndarray,
        current_tcp: np.ndarray,
        prior_velocity: np.ndarray,
        ref_anchor_tcp: np.ndarray,
        prior_tcp_velocity_xyz: np.ndarray,
        prior_tcp_angular_speed: float,
        ref_obs_ts: float,
        ref_generation_ts: float,
        now: float,
    ) -> Projection:
        q = np.asarray(current_q, dtype=np.float64)
        hold = np.repeat(q[None], self.execute, axis=0)
        proposed = np.asarray(proposed_q, dtype=np.float64)
        ref = np.asarray(ref_q, dtype=np.float64)
        if q.shape != (7,) or proposed.shape != (self.horizon, 7) or ref.shape != proposed.shape:
            raise ValueError("Action/reference/q dimensions disagree with configured H and Franka 7D")
        if not np.isfinite(proposed).all() or not np.isfinite(ref).all() or not np.isfinite(q).all():
            return _reject(hold, [], "nonfinite")
        # ref_generation_ts is server wall time; now/ref_obs_ts share the
        # client monotonic clock. Never compare timestamps across those clocks.
        if not np.isfinite(ref_generation_ts) or ref_obs_ts > now or now - ref_obs_ts > self.limits.max_ref_age_s:
            return _reject(hold, [], "stale_reference")
        try:
            anchored = reanchor_reference(ref, ref_anchor_tcp, current_tcp, q, self.kinematics)
        except (ValueError, np.linalg.LinAlgError):
            return _reject(hold, [], "reanchor_failed")
        radius = self.limits.reference_radius
        # Actor proposals are absolute joint targets; preserve their offset from
        # the original VLA chunk when the reference is reanchored.
        shifted = anchored + np.clip(proposed - ref, -radius, radius)
        clipped = not np.allclose(shifted, anchored + proposed - ref)
        flags = ["reference_clipped"] if clipped else []
        previous_q = q
        previous_pose = _pose(current_tcp)
        previous_velocity = np.asarray(prior_velocity, dtype=np.float64)
        previous_tcp_velocity = np.asarray(prior_tcp_velocity_xyz, dtype=np.float64)
        previous_tcp_angular_speed = float(prior_tcp_angular_speed)
        if previous_velocity.shape != (7,) or not np.isfinite(previous_velocity).all():
            return _reject(hold, flags, "invalid_prior_velocity")
        if (
            previous_tcp_velocity.shape != (3,)
            or not np.isfinite(previous_tcp_velocity).all()
            or not np.isfinite(previous_tcp_angular_speed)
        ):
            return _reject(hold, flags, "invalid_prior_tcp_velocity")
        safe = []
        dt = 1 / self.limits.control_hz
        for target in shifted[: self.execute]:
            velocity = (target - previous_q) / dt
            acceleration = (velocity - previous_velocity) / dt
            if np.any(target < self.limits.joint_lower) or np.any(target > self.limits.joint_upper):
                return _reject(hold, flags, "joint_position_limit")
            if np.any(np.abs(target - previous_q) > self.limits.max_joint_step):
                return _reject(hold, flags, "joint_step_limit")
            if np.any(np.abs(velocity) > self.limits.max_joint_velocity):
                return _reject(hold, flags, "joint_velocity_limit")
            if np.any(np.abs(acceleration) > self.limits.max_joint_acceleration):
                return _reject(hold, flags, "joint_acceleration_limit")
            try:
                pose = _pose(self.kinematics.fk(target))
            except ValueError:
                return _reject(hold, flags, "fk_failed")
            tcp_delta = pose[:3, 3] - previous_pose[:3, 3]
            tcp_velocity = tcp_delta / dt
            tcp_acceleration = (tcp_velocity - previous_tcp_velocity) / dt
            if np.any(np.abs(tcp_delta) > self.limits.max_tcp_step_xyz):
                return _reject(hold, flags, "tcp_step_limit")
            if np.any(np.abs(tcp_velocity) > self.limits.max_tcp_velocity_xyz):
                return _reject(hold, flags, "tcp_velocity_limit")
            if np.any(np.abs(tcp_acceleration) > self.limits.max_tcp_acceleration_xyz):
                return _reject(hold, flags, "tcp_acceleration_limit")
            angle = _rotation_angle(previous_pose, pose)
            angular_speed = angle / dt
            if angle > self.limits.max_tcp_angle_step_rad:
                return _reject(hold, flags, "tcp_angle_limit")
            if angular_speed > self.limits.max_tcp_angular_velocity_rad_s:
                return _reject(hold, flags, "tcp_angular_velocity_limit")
            if abs(angular_speed - previous_tcp_angular_speed) / dt > self.limits.max_tcp_angular_acceleration_rad_s2:
                return _reject(hold, flags, "tcp_angular_acceleration_limit")
            if np.any(pose[:3, 3] < self.limits.workspace_lower_xyz) or np.any(
                pose[:3, 3] > self.limits.workspace_upper_xyz
            ):
                return _reject(hold, flags, "workspace_limit")
            safe.append(target)
            previous_q, previous_pose, previous_velocity = target, pose, velocity
            previous_tcp_velocity, previous_tcp_angular_speed = tcp_velocity, angular_speed
        return Projection(executed_q=np.stack(safe), accepted=True, flags=tuple(flags))

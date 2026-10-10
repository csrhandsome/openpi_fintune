"""Contact network configuration and shared feature layout."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ForceStageConfig:
    """Contact experiment settings, separate from ordinary RLT replay/networks."""

    policy: str = "actor"  # actor or vla (frozen VLA, serving only)
    replay_path: str = "./runs/openpi_rlt/force/replay.jsonl"
    checkpoint_path: str = "./runs/openpi_rlt/force/actor_critic.pt"
    readout_dir: str | None = None
    batch_size: int = 64
    hidden_dim: int = 256
    layers: int = 2
    fixed_std: float = 0.02
    max_reference_offset_rad: float = 0.05
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    bc_weight: float = 1.0
    smooth_weight: float = 0.0

    def __post_init__(self):
        if self.policy not in {"actor", "vla"}:
            raise ValueError("rlt.force_stage.policy must be actor or vla")
        for name in ("replay_path", "checkpoint_path"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"rlt.force_stage.{name} must be a nonempty path")
        for name in ("batch_size", "hidden_dim", "layers"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"rlt.force_stage.{name} must be a positive integer")
        for name in (
            "fixed_std",
            "max_reference_offset_rad",
            "gamma",
            "tau",
            "actor_lr",
            "critic_lr",
            "bc_weight",
            "smooth_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"rlt.force_stage.{name} must be finite")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("Force gamma and tau must be in (0,1]")
        if min(self.max_reference_offset_rad, self.actor_lr, self.critic_lr) <= 0:
            raise ValueError("Force offset and learning rates must be positive")
        if min(self.fixed_std, self.bc_weight, self.smooth_weight) < 0:
            raise ValueError("Force noise and loss weights must be nonnegative")


def validate_joint_config(config):
    metadata = config.policy_metadata or {}
    if metadata.get("action_space") != "joint" or metadata.get("action_dim") != 8:
        raise ValueError("Force stage requires a single Franka joint layout (7 joints + gripper); EE needs an adapter")
    if config.rlt.action_representation != "absolute":
        raise ValueError("Force stage requires absolute joint references in radians")


@dataclass(frozen=True)
class NetworkConfig:
    z_rlt_dim: int
    z_grip_dim: int
    horizon: int
    executed_steps: int
    action_dim: int = 7
    hidden_dim: int = 256
    layers: int = 2
    fixed_std: float = 0.02
    max_reference_offset: float = 0.05

    def __post_init__(self):
        if (
            min(
                self.z_rlt_dim,
                self.z_grip_dim,
                self.horizon,
                self.executed_steps,
                self.action_dim,
                self.hidden_dim,
                self.layers,
            )
            <= 0
        ):
            raise ValueError("Network dimensions must be positive")
        if self.executed_steps > self.horizon or self.fixed_std < 0 or self.max_reference_offset <= 0:
            raise ValueError("Invalid H/E or actor distribution")


def _mlp(input_dim: int, hidden: int, layers: int, output_dim: int) -> nn.Sequential:
    blocks = []
    for _ in range(layers):
        blocks += [nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
        input_dim = hidden
    blocks.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*blocks)


def _state(z_rlt, z_grip, q, ref, cfg):
    if z_rlt.shape[-1] != cfg.z_rlt_dim or z_grip.shape[-1] != cfg.z_grip_dim:
        raise ValueError("RL-token/grip dimensions disagree with checkpoint")
    if q.shape[-1] != 7 or ref.shape[-2:] != (cfg.horizon, cfg.action_dim):
        raise ValueError("q/ref dimensions disagree with checkpoint")
    return torch.cat((z_rlt, z_grip, q, ref.flatten(start_dim=1)), dim=-1)

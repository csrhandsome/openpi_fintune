"""Small PyTorch chunk actor and twin Q networks for frozen-VLA contact RL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


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


class ChunkActor(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg
        state_dim = cfg.z_rlt_dim + cfg.z_grip_dim + 7 + cfg.horizon * cfg.action_dim
        self.net = _mlp(state_dim, cfg.hidden_dim, cfg.layers, cfg.horizon * cfg.action_dim)

    def forward(self, z_rlt, z_grip, q, ref):
        residual = self.net(_state(z_rlt, z_grip, q, ref, self.cfg))
        residual = residual.reshape(-1, self.cfg.horizon, self.cfg.action_dim)
        return ref + self.cfg.max_reference_offset * torch.tanh(residual)

    def sample(self, z_rlt, z_grip, q, ref, *, deterministic: bool = False):
        mean = self(z_rlt, z_grip, q, ref)
        if deterministic or self.cfg.fixed_std == 0:
            return mean
        return mean + torch.randn_like(mean) * self.cfg.fixed_std


class QNetwork(nn.Module):
    """Condition Q on the full reference and only the executed E-step action."""

    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg
        state_dim = cfg.z_rlt_dim + cfg.z_grip_dim + 7 + cfg.horizon * cfg.action_dim
        self.net = _mlp(
            state_dim + cfg.executed_steps * cfg.action_dim + cfg.executed_steps, cfg.hidden_dim, cfg.layers, 1
        )

    def forward(self, z_rlt, z_grip, q, ref, executed, executed_mask):
        if executed.shape[-2:] != (self.cfg.executed_steps, self.cfg.action_dim):
            raise ValueError("Critic actions must contain E executed joint targets")
        if executed_mask.shape != executed.shape[:2]:
            raise ValueError("Executed mask must be [B,E]")
        state = _state(z_rlt, z_grip, q, ref, self.cfg)
        action = (executed * executed_mask[..., None]).flatten(start_dim=1)
        features = torch.cat((state, action, executed_mask.float()), dim=-1)
        return self.net(features).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.q1 = QNetwork(cfg)
        self.q2 = QNetwork(cfg)

    def forward(self, *args):
        return self.q1(*args), self.q2(*args)

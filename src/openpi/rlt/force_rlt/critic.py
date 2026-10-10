"""Contact-specific twin critic."""

from __future__ import annotations

import torch
from torch import nn

from openpi.rlt.force_rlt.config import NetworkConfig
from openpi.rlt.force_rlt.config import _mlp
from openpi.rlt.force_rlt.config import _state


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

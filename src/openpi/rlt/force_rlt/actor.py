"""Contact-specific chunk actor."""

from __future__ import annotations

import torch
from torch import nn

from openpi.rlt.force_rlt.config import NetworkConfig
from openpi.rlt.force_rlt.config import _mlp
from openpi.rlt.force_rlt.config import _state


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

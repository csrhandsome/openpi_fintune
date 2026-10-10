"""Replaceable twin critic over RLT state and actual, masked execution chunks."""

from dataclasses import dataclass

import torch
from torch import nn

from openpi.rlt.episode import ReplaySpec
from openpi.rlt.network_utils import InputProjection
from openpi.rlt.network_utils import initialize_linear
from openpi.rlt.network_utils import layer_norm


@dataclass(frozen=True)
class CriticConfig:
    hidden_dim: int = 256
    layers: int = 2
    input_projection: bool = True

    def __post_init__(self):
        if type(self.input_projection) is not bool:
            raise ValueError("input_projection must be boolean")
        if min(self.hidden_dim, self.layers) <= 0:
            raise ValueError("Critic dimensions must be positive")


class QNetwork(nn.Module):
    def __init__(self, spec: ReplaySpec, config: CriticConfig | None = None):
        super().__init__()
        config = config or CriticConfig()
        self.spec = spec
        self.config = config
        input_dim = spec.z_dim + spec.grip_dim + spec.proprio_dim + spec.chunk_len * (spec.action_dim + 1)
        self.projection = (
            InputProjection(spec.z_dim, spec.proprio_dim, spec.chunk_len * spec.action_dim, spec.grip_dim)
            if config.input_projection
            else None
        )
        if self.projection is not None:
            input_dim = self.projection.output_dim
        blocks = []
        for _ in range(config.layers):
            blocks.extend(
                [nn.Linear(input_dim, config.hidden_dim), layer_norm(config.hidden_dim), nn.GELU(approximate="tanh")]
            )
            input_dim = config.hidden_dim
        blocks.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*blocks)
        self.apply(initialize_linear)

    def forward(self, z_rlt, proprio, actions, mask, z_grip=None):
        if actions.shape[1:] != (self.spec.chunk_len, self.spec.action_dim) or mask.shape != actions.shape[:2]:
            raise ValueError("Critic needs [B,E,D] executed actions and [B,E] mask")
        features = [z_rlt, proprio, (actions * mask[..., None]).flatten(start_dim=1), mask.float()]
        if self.spec.grip_dim:
            if z_grip is None or z_grip.shape != (z_rlt.shape[0], self.spec.grip_dim):
                raise ValueError("Critic requires checkpoint-compatible grip features")
            features.append(z_grip)
        inputs = torch.cat(features, dim=-1)
        if self.projection is not None:
            inputs = self.projection(z_rlt, proprio, actions * mask[..., None], z_grip)
        return self.net(inputs).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, spec: ReplaySpec, config: CriticConfig | None = None):
        super().__init__()
        config = config or CriticConfig()
        self.config = config
        self.q1 = QNetwork(spec, config)
        self.q2 = QNetwork(spec, config)

    def forward(self, *args):
        return self.q1(*args), self.q2(*args)

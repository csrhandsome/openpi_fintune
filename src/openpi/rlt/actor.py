"""Replaceable ordinary RLT chunk actor, with no robot-specific inputs."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from openpi.rlt.episode import ReplaySpec
from openpi.rlt.network_utils import InputProjection
from openpi.rlt.network_utils import initialize_linear
from openpi.rlt.network_utils import layer_norm


@dataclass(frozen=True)
class ActorConfig:
    hidden_dim: int = 256
    layers: int = 2
    fixed_std: float = 0.05
    max_reference_offset: float = 0.5
    input_projection: bool = True
    output_mode: str = "direct"

    def __post_init__(self):
        if type(self.input_projection) is not bool or self.output_mode not in {"direct", "residual"}:
            raise ValueError("Invalid actor input projection or output mode")
        if min(self.hidden_dim, self.layers) <= 0:
            raise ValueError("Actor dimensions must be positive")
        if not all(math.isfinite(x) for x in (self.fixed_std, self.max_reference_offset)):
            raise ValueError("Actor distribution must be finite")
        if self.fixed_std < 0 or self.max_reference_offset <= 0:
            raise ValueError("Invalid actor distribution")


class ChunkActor(nn.Module):
    def __init__(self, spec: ReplaySpec, config: ActorConfig | None = None):
        super().__init__()
        config = config or ActorConfig()
        self.spec = spec
        self.config = config
        input_dim = spec.z_dim + spec.grip_dim + spec.proprio_dim + spec.horizon * spec.action_dim
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
        self.output_steps = spec.chunk_len if config.output_mode == "direct" else spec.horizon
        blocks.append(nn.Linear(input_dim, self.output_steps * spec.action_dim))
        self.net = nn.Sequential(*blocks)
        self.apply(initialize_linear)

    def forward(self, z_rlt, proprio, ref_chunk, z_grip=None, *, reference_condition=None):
        condition = ref_chunk if reference_condition is None else reference_condition
        features = [z_rlt, proprio, condition.flatten(start_dim=1)]
        if self.spec.grip_dim:
            if z_grip is None or z_grip.shape != (z_rlt.shape[0], self.spec.grip_dim):
                raise ValueError("Actor requires checkpoint-compatible grip features")
            features.append(z_grip)
        inputs = torch.cat(features, dim=-1)
        if self.projection is not None:
            inputs = self.projection(z_rlt, proprio, condition[:, : self.spec.chunk_len], z_grip)
        prediction = self.net(inputs).reshape(-1, self.output_steps, self.spec.action_dim)
        if self.config.output_mode == "direct":
            return torch.cat([prediction, ref_chunk[:, self.output_steps :]], dim=1)
        return ref_chunk + self.config.max_reference_offset * torch.tanh(prediction)

    def sample(self, z_rlt, proprio, ref_chunk, z_grip=None, *, deterministic: bool = False, reference_condition=None):
        mean = self(z_rlt, proprio, ref_chunk, z_grip, reference_condition=reference_condition)
        if deterministic:
            return mean
        sampled = mean[:, : self.output_steps] + torch.randn_like(mean[:, : self.output_steps]) * self.config.fixed_std
        return torch.cat([sampled, mean[:, self.output_steps :]], dim=1)

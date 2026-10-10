"""Input encoders and initialization used by the reference RLT networks."""

import torch
from torch import nn


def layer_norm(width):
    return nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)


class InputProjection(nn.Module):
    def __init__(self, z_dim, proprio_dim, action_dim, grip_dim=0):
        super().__init__()
        self.z_proj = nn.Linear(z_dim, 256)
        self.proprio_proj = nn.Linear(proprio_dim, 64)
        self.action_proj = nn.Linear(action_dim, 256)
        self.grip_proj = nn.Linear(grip_dim, 256) if grip_dim else None
        self.z_norm = layer_norm(256)
        self.proprio_norm = layer_norm(64)
        self.action_norm = layer_norm(256)
        self.grip_norm = layer_norm(256) if grip_dim else None
        self.output_dim = 576 + (256 if grip_dim else 0)

    def forward(self, z, proprio, actions, grip=None):
        features = [
            self.z_norm(self.z_proj(z)),
            torch.tanh(self.proprio_norm(self.proprio_proj(proprio))),
            torch.tanh(self.action_norm(self.action_proj(actions.flatten(start_dim=1)))),
        ]
        if self.grip_proj is not None:
            features.append(self.grip_norm(self.grip_proj(grip)))
        return torch.cat(features, dim=-1)


def initialize_linear(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)

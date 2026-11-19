"""SIGReg, adapted from lucas-maes/le-wm/module.py (MIT).

Copyright (c) 2026 Lucas Maes. The upstream MIT notice is retained in
third_party/le-wm/LICENSE and docs/licenses/le-wm-MIT.txt.
This is the single-GPU statistic; leading dimensions are separate populations,
and the penultimate dimension MUST be independent batch samples, not time.
"""

import torch
from torch import nn


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        if knots < 2 or num_proj < 1:
            raise ValueError("SIGReg requires knots >= 2 and num_proj >= 1")
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if embeddings.ndim < 2 or embeddings.shape[-2] < 2:
            raise ValueError("SIGReg needs at least two independent samples per population")
        # Keep characteristic functions and quadrature in FP32 under BF16 training.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            features = embeddings.float()
            device = features.device if generator is None else generator.device
            directions = torch.randn(features.shape[-1], self.num_proj, device=device, generator=generator)
            directions = directions.to(features.device)
            directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)
            phase = (features @ directions).unsqueeze(-1) * self.t
            error = (phase.cos().mean(-3) - self.phi).square() + phase.sin().mean(-3).square()
            return ((error @ self.weights) * features.shape[-2]).mean()

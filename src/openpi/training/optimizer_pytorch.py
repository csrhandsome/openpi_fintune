"""Hyperparameters consumed by the PyTorch training entrypoint."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule:
    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6


@dataclasses.dataclass(frozen=True)
class AdamW:
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

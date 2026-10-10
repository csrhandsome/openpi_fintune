"""Shared YAML settings for offline actor training and online serving."""

import dataclasses
import math

from openpi.rlt.actor import ActorConfig
from openpi.rlt.critic import CriticConfig
from openpi.rlt.learner import LearnerConfig


def positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def boolean(name, value):
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")


@dataclasses.dataclass(frozen=True)
class ForceConfig:
    enabled: bool = False
    input: str = "rgb"  # rgb history encoded by server, or precomputed z_grip
    feature_dim: int = 12288  # 4 frames * 2 fingers * 3 features * 512
    weights: str | None = None
    roi: tuple[int, int, int, int] = (0, 0, 224, 224)
    max_skew_s: float = 0.02
    max_age_s: float = 0.2

    @property
    def grip_dim(self):
        return self.feature_dim if self.enabled else 0

    def __post_init__(self):
        boolean("force.enabled", self.enabled)
        positive_int("force.feature_dim", self.feature_dim)
        if self.input not in {"rgb", "features"}:
            raise ValueError("force.input must be rgb or features")
        if len(self.roi) != 4 or any(type(x) is not int for x in self.roi):
            raise ValueError("force.roi must contain four integers")
        x0, y0, x1, y1 = self.roi
        if not 0 <= x0 < x1 or not 0 <= y0 < y1:
            raise ValueError("Invalid force.roi")
        for name in ("max_skew_s", "max_age_s"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"force.{name} must be finite and positive")
        if self.enabled and self.input == "rgb" and self.feature_dim != 12288:
            raise ValueError("RGB finger history requires force.feature_dim=12288")


@dataclasses.dataclass(frozen=True)
class ReplayConfig:
    directory: str = "./runs/openpi_rlt/replay"
    capacity: int = 200_000

    def __post_init__(self):
        positive_int("replay.capacity", self.capacity)
        if not isinstance(self.directory, str) or not self.directory:
            raise ValueError("replay.directory must be a nonempty path")


@dataclasses.dataclass(frozen=True)
class CollectConfig:
    enabled: bool = True

    def __post_init__(self):
        boolean("collect.enabled", self.enabled)


@dataclasses.dataclass(frozen=True)
class WarmupConfig:
    min_transitions: int = 600
    updates: int = 20_000

    def __post_init__(self):
        positive_int("warmup.min_transitions", self.min_transitions)
        positive_int("warmup.updates", self.updates)


@dataclasses.dataclass(frozen=True)
class LearnerSettings(LearnerConfig):
    batch_size: int = 128
    grad_updates_per_cycle: int = 5
    publish_interval: int = 500
    checkpoint_interval: int = 1000
    checkpoint_path: str = "./runs/openpi_rlt/actor_critic.pt"
    freeze_after_warmup: bool = False
    warmup_bc_weight: float = 10.0
    warmup_q_weight: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        for name in ("batch_size", "grad_updates_per_cycle", "publish_interval", "checkpoint_interval"):
            positive_int(f"learner.{name}", getattr(self, name))
        boolean("learner.freeze_after_warmup", self.freeze_after_warmup)
        for name in ("warmup_bc_weight", "warmup_q_weight"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"learner.{name} must be finite and nonnegative")
        if not self.checkpoint_path:
            raise ValueError("learner.checkpoint_path must be nonempty")

    def training_config(self, *, warmup=False):
        values = {field.name: getattr(self, field.name) for field in dataclasses.fields(LearnerConfig)}
        if warmup:
            values.update(bc_weight=self.warmup_bc_weight, q_weight=self.warmup_q_weight)
        return LearnerConfig(**values)


def construct(cls, values, label):
    """Reject misspelled YAML keys rather than silently ignoring them."""
    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise ValueError(f"{label} must be a mapping with string keys")
    unknown = values.keys() - {field.name for field in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {label} settings: {', '.join(sorted(unknown))}")
    return cls(**values)


def load_actor_settings(values):
    values = dict(values)
    for key, cls in (
        ("force", ForceConfig),
        ("replay", ReplayConfig),
        ("learner", LearnerSettings),
        ("actor", ActorConfig),
        ("critic", CriticConfig),
    ):
        nested = values.pop(key, {})
        if key == "force" and "roi" in nested:
            nested = {**nested, "roi": tuple(nested["roi"])}
        values[key] = construct(cls, nested, f"rlt.{key}")
    return values

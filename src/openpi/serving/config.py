"""Server YAMLs reference training model/data settings without duplicating them."""

import dataclasses
import pathlib

import yaml

from openpi.rlt.actor import ActorConfig
from openpi.rlt.critic import CriticConfig
from openpi.rlt.rewards import TerminalReward
from openpi.rlt.settings import CollectConfig
from openpi.rlt.settings import ForceConfig
from openpi.rlt.settings import LearnerSettings
from openpi.rlt.settings import ReplayConfig
from openpi.rlt.settings import WarmupConfig
from openpi.rlt.settings import boolean
from openpi.rlt.settings import construct
from openpi.rlt.settings import load_actor_settings
from openpi.rlt.settings import positive_int
from openpi.training.config import local_path

SERVER_CONFIG_DIR = local_path("config/server")


@dataclasses.dataclass(frozen=True)
class RLTConfig:
    mode: str = "online_rl"
    actor_checkpoint: str | None = None
    chunk_len: int = 10
    proprio_dim: int | None = None
    action_representation: str = "absolute"
    action_norm_stats: str | None = None
    deterministic: bool = False
    critical_phase_only: bool = False
    collect: CollectConfig = dataclasses.field(default_factory=CollectConfig)
    force: ForceConfig = dataclasses.field(default_factory=ForceConfig)
    replay: ReplayConfig = dataclasses.field(default_factory=ReplayConfig)
    warmup: WarmupConfig = dataclasses.field(default_factory=WarmupConfig)
    learner: LearnerSettings = dataclasses.field(default_factory=LearnerSettings)
    actor: ActorConfig = dataclasses.field(default_factory=ActorConfig)
    critic: CriticConfig = dataclasses.field(default_factory=CriticConfig)
    reward: TerminalReward = dataclasses.field(default_factory=TerminalReward)

    def __post_init__(self):
        if self.mode not in {"online_rl", "eval"}:
            raise ValueError("rlt.mode must be online_rl or eval")
        positive_int("rlt.chunk_len", self.chunk_len)
        if self.proprio_dim is not None:
            positive_int("rlt.proprio_dim", self.proprio_dim)
        if self.action_representation not in {"absolute", "ee_delta"}:
            raise ValueError("rlt.action_representation must be absolute or ee_delta")
        boolean("rlt.deterministic", self.deterministic)
        boolean("rlt.critical_phase_only", self.critical_phase_only)
        if self.replay.capacity < self.warmup.min_transitions:
            raise ValueError("replay.capacity cannot be smaller than warmup.min_transitions")
        if self.mode == "online_rl" and self.warmup.updates < self.learner.actor_update_period:
            raise ValueError("warmup.updates must allow at least one actor update")
        if self.mode == "eval" and self.actor_checkpoint is None:
            raise ValueError("RLT eval requires rlt.actor_checkpoint")


@dataclasses.dataclass(frozen=True)
class InferenceConfig:
    mode: str = "vla"
    train_config: str = "config/train/openpi.yaml"
    checkpoint: str | None = None
    readout: str | None = None
    device: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8001
    record: bool = False
    use_torch_compile: bool = False
    use_triton: bool = False
    rlt: RLTConfig = dataclasses.field(default_factory=RLTConfig)

    def __post_init__(self):
        if self.mode not in {"vla", "rlt", "force"}:
            raise ValueError("inference.mode must be vla, rlt or force")
        positive_int("port", self.port)
        if self.port > 65535:
            raise ValueError("port must be <= 65535")
        boolean("record", self.record)
        boolean("use_triton", self.use_triton)
        boolean("use_torch_compile", self.use_torch_compile)


def find_config(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.parent == pathlib.Path(".") and not path.is_absolute():
        return SERVER_CONFIG_DIR / (value if path.suffix else f"{value}.yaml")
    return local_path(value)


def load_config(value: str = "openpi") -> InferenceConfig:
    path = find_config(value)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Inference YAML must be a mapping")
    values = dict(values)
    nested = values.pop("rlt", {})
    if not isinstance(nested, dict):
        raise ValueError("rlt must be a mapping")
    nested = load_actor_settings(nested)
    for key, cls in (("collect", CollectConfig), ("warmup", WarmupConfig), ("reward", TerminalReward)):
        nested[key] = construct(cls, nested.pop(key, {}), f"rlt.{key}")
    nested["force"] = dataclasses.replace(nested["force"], weights=_path(nested["force"].weights))
    nested["replay"] = dataclasses.replace(nested["replay"], directory=_path(nested["replay"].directory))
    nested["learner"] = dataclasses.replace(nested["learner"], checkpoint_path=_path(nested["learner"].checkpoint_path))
    for key in ("actor_checkpoint", "action_norm_stats"):
        if key in nested:
            nested[key] = _path(nested[key])
    values["rlt"] = construct(RLTConfig, nested, "rlt")
    for key in ("train_config", "checkpoint", "readout"):
        if key in values:
            values[key] = _path(values[key])
    return construct(InferenceConfig, values, "inference")


def _path(value):
    return None if value is None else str(local_path(value))

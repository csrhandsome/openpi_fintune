"""PyTorch training settings loaded from repository-root config/train/*.yaml."""

import argparse
from collections.abc import Sequence
import dataclasses
import math
import os
import pathlib
import sys
from typing import Any, Literal

import tyro
import yaml

from openpi.models import pi0_config
from openpi.rlt.actor import ActorConfig
from openpi.rlt.critic import CriticConfig
from openpi.rlt.force_rlt.config import ForceStageConfig
from openpi.rlt.settings import ForceConfig
from openpi.rlt.settings import LearnerSettings
from openpi.rlt.settings import ReplayConfig
from openpi.rlt.settings import load_actor_settings
from openpi.training import data_config as _data
from openpi.training import optimizer_pytorch as _optimizer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
TRAIN_CONFIG_DIR = REPO_ROOT / "config" / "train"
DEFAULT_TRAIN_CONFIG = "openpi"
DEFAULT_RLT_CONFIG = "openpi_rlt"

# Public runtime types used by data loaders and policy consumers.
AssetsConfig = _data.AssetsConfig
DataConfig = _data.DataConfig
DataConfigFactory = _data.DataConfigFactory
FakeDataConfig = _data.FakeDataConfig
LeRobotFrankaDataConfig = _data.LeRobotFrankaDataConfig
LeRobotFrankaEEDataConfig = _data.LeRobotFrankaEEDataConfig
LeRobotFrankaJointDataConfig = _data.LeRobotFrankaJointDataConfig


def _positive_int(key, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


# Runtime settings that may be nested under train: in the YAML.
_TRAIN_FIELD_ORDER = (
    "pytorch_weight_path",
    "jax_weight_path",
    "requires_pytorch_weights",
    "pytorch_training_precision",
    "assets_base_dir",
    "checkpoint_base_dir",
    "seed",
    "batch_size",
    "micro_batch_size",
    "num_gpus",
    "num_workers",
    "num_train_steps",
    "log_interval",
    "save_interval",
    "overwrite",
    "resume",
    "tracking_backend",
)
_TRAIN_FIELDS = frozenset(_TRAIN_FIELD_ORDER)


@dataclasses.dataclass(frozen=True)
class RLTTrainingConfig:
    stage: Literal["sft", "actor_critic", "force"] = "sft"
    vla_checkpoint: str | None = None
    output_dir: str | None = None
    learning_rate: float = 2.5e-5
    embed_dim: int = 2048
    alpha: float = 0.0
    num_tokens: int = 1
    num_layers: int = 2
    num_heads: int = 8
    max_prefix_len: int = 1024
    device: str = "auto"
    force: ForceConfig = dataclasses.field(default_factory=ForceConfig)
    force_stage: ForceStageConfig = dataclasses.field(default_factory=ForceStageConfig)
    replay: ReplayConfig = dataclasses.field(default_factory=ReplayConfig)
    learner: LearnerSettings = dataclasses.field(default_factory=LearnerSettings)
    actor: ActorConfig = dataclasses.field(default_factory=ActorConfig)
    critic: CriticConfig = dataclasses.field(default_factory=CriticConfig)
    actor_checkpoint: str | None = None
    action_representation: str = "absolute"
    action_norm_stats: str | None = None

    def __post_init__(self):
        if self.stage not in {"sft", "actor_critic", "force"}:
            raise ValueError("rlt.stage must be sft, actor_critic or force")
        if self.action_representation not in {"absolute", "ee_delta"}:
            raise ValueError("rlt.action_representation must be absolute or ee_delta")
        for key in ("embed_dim", "num_tokens", "num_layers", "num_heads", "max_prefix_len"):
            _positive_int(f"rlt.{key}", getattr(self, key))
        if self.embed_dim % self.num_heads:
            raise ValueError("rlt.embed_dim must be divisible by rlt.num_heads")
        if (
            not isinstance(self.learning_rate, int | float)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("rlt.learning_rate must be finite and positive")
        if not isinstance(self.alpha, int | float) or not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("rlt.alpha must be finite and nonnegative")
        object.__setattr__(self, "alpha", float(self.alpha))


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    mode: Literal["vla", "rlt"] = "vla"
    project_name: str = "openpi"
    exp_name: str = "ee"
    model: pi0_config.Pi0Config = dataclasses.field(default_factory=pi0_config.Pi0Config)
    data: tyro.conf.Suppress[DataConfigFactory] = dataclasses.field(default_factory=FakeDataConfig)
    pytorch_weight_path: str | None = None
    jax_weight_path: str | None = None
    requires_pytorch_weights: bool = True
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"
    lr_schedule: _optimizer.CosineDecaySchedule = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.AdamW = dataclasses.field(default_factory=_optimizer.AdamW)
    assets_base_dir: str = "./data/assets"
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 5
    # Per-GPU forward batch; None uses batch_size / world_size without accumulation.
    micro_batch_size: int | None = None
    num_gpus: int = 1
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    overwrite: bool = False
    resume: bool = False
    tracking_backend: Literal["none", "wandb", "swanlab"] = "none"
    policy_metadata: tyro.conf.Suppress[dict[str, Any] | None] = None
    rlt: RLTTrainingConfig = dataclasses.field(default_factory=RLTTrainingConfig)
    config_path: tyro.conf.Suppress[str | None] = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        return local_path(self.assets_base_dir) / self.name

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return local_path(self.checkpoint_base_dir) / self.name / self.exp_name

    @property
    def readout_dir(self) -> pathlib.Path:
        return local_path(self.rlt.output_dir) if self.rlt.output_dir else self.checkpoint_dir

    def __post_init__(self):
        if self.mode not in ("vla", "rlt"):
            raise ValueError("mode must be vla or rlt (PyTorch only)")
        if self.pytorch_training_precision not in ("bfloat16", "float32"):
            raise ValueError("pytorch_training_precision must be bfloat16 or float32")
        for key in ("name", "exp_name"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value or value in (".", "..") or pathlib.Path(value).name != value:
                raise ValueError(f"{key} must be a nonempty directory name")
        for key in ("batch_size", "num_train_steps", "log_interval", "save_interval", "num_gpus"):
            _positive_int(key, getattr(self, key))
        if self.micro_batch_size is not None:
            _positive_int("micro_batch_size", self.micro_batch_size)
            if self.batch_size % self.micro_batch_size:
                raise ValueError("batch_size must be divisible by micro_batch_size")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if self.tracking_backend not in ("none", "wandb", "swanlab"):
            raise ValueError("tracking_backend must be none, wandb or swanlab")
        for key in ("requires_pytorch_weights", "overwrite", "resume"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        for key in ("action_dim", "action_horizon", "max_token_len"):
            _positive_int(f"model.{key}", getattr(self.model, key))
        if isinstance(self.data, LeRobotFrankaDataConfig) and self.model.action_dim < self.data.action_dim:
            raise ValueError("model.action_dim (padding) cannot be smaller than data.action_dim")


def local_path(value: str) -> pathlib.Path:
    path = pathlib.Path(os.path.expandvars(value)).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _resolve_path(value: str | None) -> str | None:
    if value is None or "://" in value:
        return value
    return str(local_path(value))


def _mapping(value, label):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a YAML mapping with string keys")
    return dict(value)


def _construct(cls, value, label):
    values = _mapping(value, label)
    unknown = values.keys() - {field.name for field in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {label} settings: {', '.join(sorted(unknown))}")
    return cls(**values)


_DATA_FACTORIES = {
    "franka": LeRobotFrankaDataConfig,
    "franka_ee": LeRobotFrankaEEDataConfig,
    "franka_joint": LeRobotFrankaJointDataConfig,
    "aloha": _data.LeRobotAlohaDataConfig,
    "libero": _data.LeRobotLiberoDataConfig,
    "droid": _data.LeRobotDROIDDataConfig,
    "fake": FakeDataConfig,
}


def _load_data(value):
    values = _mapping(value, "data")
    kind = values.pop("type", "franka")
    if kind not in _DATA_FACTORIES:
        raise ValueError(f"Unknown data.type: {kind}; choose from {', '.join(_DATA_FACTORIES)}")
    assets = _construct(AssetsConfig, values.pop("assets", {}), "data.assets")
    assets = dataclasses.replace(assets, assets_dir=_resolve_path(assets.assets_dir))
    base = DataConfig(prompt_from_task=values.pop("prompt_from_task", True))
    values.update(assets=assets, base_config=base)
    return _construct(_DATA_FACTORIES[kind], values, "data")


def _read_yaml(path):
    with path.open(encoding="utf-8") as stream:
        return _mapping(yaml.safe_load(stream), str(path))


def _find_config(config_name: str | pathlib.Path) -> pathlib.Path:
    name = str(config_name)
    path = pathlib.Path(name)
    if path.is_absolute() or path.parent != pathlib.Path("."):
        path = local_path(name)
    else:
        path = TRAIN_CONFIG_DIR / (name if path.suffix in (".yaml", ".yml") else f"{name}.yaml")
    if path.is_file():
        return path.resolve()
    if pathlib.Path(name).name == name and not pathlib.Path(name).suffix:
        matches = [
            candidate
            for candidate in sorted(TRAIN_CONFIG_DIR.glob("*.yaml"))
            if _read_yaml(candidate).get("name", candidate.stem) == name
        ]
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(f"Training YAML not found: {path}. Configs live under {TRAIN_CONFIG_DIR}")


def require_world_size(config: TrainConfig) -> int:
    """Return the process count selected by train.num_gpus.

    scripts/train.py starts that many processes. A manual torchrun launch must use the same count.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != config.num_gpus:
        raise ValueError(
            f"train.num_gpus is {config.num_gpus}, but WORLD_SIZE is {world}. "
            "Run scripts/train.py so it starts the configured number of processes."
        )
    return world


def gradient_accumulation(config: TrainConfig, world_size: int) -> tuple[int, int]:
    """Return (per-GPU micro batch, optimizer accumulation steps).

    RLinf uses global_batch / (micro_batch_size * world_size). An unset micro
    batch performs one step on the full global batch.
    """
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if config.micro_batch_size is None:
        if config.batch_size % world_size:
            raise ValueError(f"Global batch_size={config.batch_size} must be divisible by world_size={world_size}")
        return config.batch_size // world_size, 1
    if config.batch_size % (config.micro_batch_size * world_size):
        raise ValueError(
            f"batch_size={config.batch_size} must be divisible by "
            f"micro_batch_size={config.micro_batch_size} * world_size={world_size}"
        )
    return config.micro_batch_size, config.batch_size // (config.micro_batch_size * world_size)


def _migrate_tracking_settings(values):
    # Accept logging switches stored by older checkpoint configs.
    enabled_backends = []
    has_legacy_switch = False
    for backend in ("wandb", "swanlab"):
        key = f"{backend}_enabled"
        if key in values:
            has_legacy_switch = True
            enabled = values.pop(key)
            if type(enabled) is not bool:
                raise ValueError(f"{key} must be boolean")
            if enabled:
                enabled_backends.append(backend)
    if "tracking_backend" not in values and has_legacy_switch:
        if len(enabled_backends) > 1:
            raise ValueError("Choose one tracking_backend: wandb or swanlab")
        values["tracking_backend"] = enabled_backends[0] if enabled_backends else "none"


def _hoist_train(values):
    _migrate_tracking_settings(values)
    nested = values.pop("train", None)
    if nested is None:
        return values
    nested = _mapping(nested, "train")
    _migrate_tracking_settings(nested)
    unknown = nested.keys() - _TRAIN_FIELDS
    if unknown:
        raise ValueError(f"Unknown train settings: {', '.join(sorted(unknown))}")
    for key, value in nested.items():
        values.setdefault(key, value)
    return values


def get_config(config_name: str | pathlib.Path = DEFAULT_TRAIN_CONFIG) -> TrainConfig:
    path = _find_config(config_name)
    values = _hoist_train(_read_yaml(path))
    values.setdefault("name", path.stem)
    model_values = _mapping(values.pop("model", {}), "model")
    if model_values.pop("type", "pi0") != "pi0":
        raise ValueError("model.type must be pi0; JAX/FAST training configurations are unsupported")
    values["model"] = _construct(pi0_config.Pi0Config, model_values, "model")
    values["data"] = _load_data(values.pop("data", {}))
    for key, cls in (
        ("lr_schedule", _optimizer.CosineDecaySchedule),
        ("optimizer", _optimizer.AdamW),
        ("rlt", RLTTrainingConfig),
    ):
        nested = values.pop(key, {})
        if key == "rlt":
            nested = load_actor_settings(_mapping(nested, "rlt"))
            nested["force_stage"] = _construct(ForceStageConfig, nested.pop("force_stage", {}), "rlt.force_stage")
        values[key] = _construct(cls, nested, key)
    values["config_path"] = str(path)
    config = _construct(TrainConfig, values, "training")
    return resolve_paths(config)


def resolve_paths(config: TrainConfig) -> TrainConfig:
    """Resolve paths and derive serving metadata after YAML/CLI overrides."""
    if isinstance(config.data, LeRobotFrankaDataConfig):
        metadata = dict(config.policy_metadata or {})
        metadata.update(
            action_space="joint" if config.data.action_space == "joint" else "ee",
            action_representation=config.data.action_space,
            action_dim=config.data.action_dim,
            action_horizon=config.model.action_horizon,
        )
        metadata.setdefault("action_fps", 30.0)
        config = dataclasses.replace(config, policy_metadata=metadata)
    return dataclasses.replace(
        config,
        pytorch_weight_path=_resolve_path(config.pytorch_weight_path or os.environ.get("OPENPI_PYTORCH_WEIGHT_PATH")),
        jax_weight_path=_resolve_path(config.jax_weight_path or os.environ.get("OPENPI_JAX_WEIGHT_PATH")),
        assets_base_dir=str(local_path(config.assets_base_dir)),
        checkpoint_base_dir=str(local_path(config.checkpoint_base_dir)),
        rlt=dataclasses.replace(
            config.rlt,
            vla_checkpoint=_resolve_path(config.rlt.vla_checkpoint),
            output_dir=_resolve_path(config.rlt.output_dir),
            actor_checkpoint=_resolve_path(config.rlt.actor_checkpoint),
            action_norm_stats=_resolve_path(config.rlt.action_norm_stats),
            force=dataclasses.replace(config.rlt.force, weights=_resolve_path(config.rlt.force.weights)),
            force_stage=dataclasses.replace(
                config.rlt.force_stage,
                replay_path=_resolve_path(config.rlt.force_stage.replay_path),
                checkpoint_path=_resolve_path(config.rlt.force_stage.checkpoint_path),
                readout_dir=_resolve_path(config.rlt.force_stage.readout_dir),
            ),
            replay=dataclasses.replace(config.rlt.replay, directory=_resolve_path(config.rlt.replay.directory)),
            learner=dataclasses.replace(
                config.rlt.learner, checkpoint_path=_resolve_path(config.rlt.learner.checkpoint_path)
            ),
        ),
    )


def cli(args: Sequence[str] | None = None, *, default: str = DEFAULT_TRAIN_CONFIG) -> TrainConfig:
    args = list(sys.argv[1:] if args is None else args)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=default)
    if args and not args[0].startswith("-"):
        args = ["--config", args[0], *args[1:]]
    selected, overrides = parser.parse_known_args(args)
    if "--help" in overrides or "-h" in overrides:
        print(f"Select a training YAML with --config PATH (default: config/train/{default}.yaml).")
    config = tyro.cli(TrainConfig, default=get_config(selected.config), args=overrides)
    return resolve_paths(config)


def to_dict(config: TrainConfig) -> dict:
    values = dataclasses.asdict(config)
    kind = next(key for key, cls in _DATA_FACTORIES.items() if type(config.data) is cls)
    data_values = {
        field.name: getattr(config.data, field.name)
        for field in dataclasses.fields(config.data)
        if field.name not in ("assets", "base_config", "repack_transforms")
    }
    data_values.update(
        type=kind,
        assets=dataclasses.asdict(config.data.assets),
        prompt_from_task=(config.data.base_config or DataConfig()).prompt_from_task,
    )
    values["data"] = data_values
    values["model"]["type"] = "pi0"
    values["train"] = {key: values.pop(key) for key in _TRAIN_FIELD_ORDER}
    return values


def save_config(config: TrainConfig, directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(yaml.safe_dump(to_dict(config), sort_keys=False), encoding="utf-8")

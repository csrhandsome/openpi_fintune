import dataclasses
import logging
import pathlib
import sys

import tyro
import wandb

from openpi.models import model as _model
from openpi.models import pi0_audio
import openpi.training.config as _config
import openpi.training.weight_loaders as _weight_loaders

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.train as _train  # noqa: E402


@dataclasses.dataclass(frozen=True)
class StagedTrainArgs:
    configs: tuple[str, ...] = (
        "pi0_base_audio_joint_route_warmup_lora",
        "pi0_base_audio_joint_audio_action_lora",
        "pi0_base_audio_joint_mixed_aa_avla_lora",
        "pi0_base_audio_joint_full_avla_lora",
    )
    exp_name: str = tyro.MISSING
    stage_exp_names: tuple[str, ...] = ()
    checkpoint_base_dir: str = "./checkpoints"
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    qwen: bool = False
    qwen_model_name: str = "Qwen/Qwen3-ASR-0.6B"
    qwen_device: str = "auto"
    num_train_steps: tuple[int, ...] = ()
    save_interval: tuple[int, ...] = ()


def _pick_override(values: tuple[int, ...], stage_index: int) -> int | None:
    if stage_index >= len(values):
        return None
    return values[stage_index]


def _stage_exp_name(args: StagedTrainArgs, config_name: str, stage_index: int) -> str:
    if stage_index < len(args.stage_exp_names):
        return args.stage_exp_names[stage_index]
    slug = config_name.replace("/", "_")
    return f"{args.exp_name}_stage{stage_index}_{slug}"


def _latest_params_path(checkpoint_dir: pathlib.Path) -> pathlib.Path:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    steps = sorted(int(path.name) for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"No numeric checkpoint steps found in {checkpoint_dir}")
    params_path = checkpoint_dir / str(steps[-1]) / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Checkpoint params not found: {params_path}")
    return params_path


def _continuation_weight_loader(
    model: _model.BaseModelConfig,
    params_path: pathlib.Path,
) -> _weight_loaders.WeightLoader:
    if isinstance(model, pi0_audio.Pi0AudioConfig):
        return _weight_loaders.Pi0AudioWeightLoader(str(params_path))
    return _weight_loaders.CheckpointWeightLoader(str(params_path))


def _build_stage_config(
    args: StagedTrainArgs,
    *,
    config_name: str,
    stage_index: int,
    previous_params_path: pathlib.Path | None,
) -> _config.TrainConfig:
    config = _config.get_config(config_name)
    if args.qwen:
        config = _config.enable_qwen_encoder(
            config,
            qwen_model_name=args.qwen_model_name,
            qwen_device=args.qwen_device,
        )

    updates = {
        "exp_name": _stage_exp_name(args, config_name, stage_index),
        "checkpoint_base_dir": args.checkpoint_base_dir,
        "overwrite": args.overwrite,
        "resume": args.resume,
        "wandb_enabled": args.wandb_enabled,
    }
    if (num_train_steps := _pick_override(args.num_train_steps, stage_index)) is not None:
        updates["num_train_steps"] = num_train_steps
    if (save_interval := _pick_override(args.save_interval, stage_index)) is not None:
        updates["save_interval"] = save_interval
    if previous_params_path is not None:
        updates["weight_loader"] = _continuation_weight_loader(config.model, previous_params_path)

    return dataclasses.replace(config, **updates)


def main(args: StagedTrainArgs | None = None) -> None:
    args = tyro.cli(StagedTrainArgs) if args is None else args
    if not args.configs:
        raise ValueError("At least one config is required.")

    logging.basicConfig(level=logging.INFO)
    previous_params_path = None
    for stage_index, config_name in enumerate(args.configs):
        stage_config = _build_stage_config(
            args,
            config_name=config_name,
            stage_index=stage_index,
            previous_params_path=previous_params_path,
        )
        logging.info(
            "Starting stage %s/%s: config=%s exp_name=%s",
            stage_index + 1,
            len(args.configs),
            config_name,
            stage_config.exp_name,
        )
        _train.main(stage_config)
        wandb.finish()
        previous_params_path = _latest_params_path(stage_config.checkpoint_dir)
        logging.info("Finished stage %s, params=%s", stage_index, previous_params_path)


if __name__ == "__main__":
    main()

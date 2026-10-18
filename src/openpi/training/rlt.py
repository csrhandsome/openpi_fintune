"""Openpi-RLT Stage 1 defaults, joint VLA/readout training, EMA and resume."""

import contextlib
import dataclasses
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from safetensors.torch import save_file
import torch
import torch.distributed as dist

from openpi.models_pytorch.rl_token import RLTokenConfig
from openpi.models_pytorch.rl_token import RLTokenReadout
from openpi.models_pytorch.rlinf.model import Observation
from openpi.training import config as train_config
from openpi.training import data_loader
from openpi.training.tracking import init_tracking
from openpi.training.validation import validate_dataset


@dataclasses.dataclass(frozen=True)
class ReadoutDefaults:
    # alpha comes from rlt.alpha. Augmentation, mask, and EMA stay code defaults.
    alpha: float = 0.0
    augment_images: bool = True
    use_mask: bool = False  # openpi-RLT train_rlt passes None to the readout.
    ema_decay: float = 0.99

    def __post_init__(self):
        if not np.isfinite(self.alpha) or self.alpha < 0 or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid joint-loss weight or EMA decay")


def _device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device_tree(item, device) for key, item in value.items()}
    return value


def _lr(step, schedule):
    if step < schedule.warmup_steps:
        initial = schedule.peak_lr / (schedule.warmup_steps + 1)
        return initial + (schedule.peak_lr - initial) * step / schedule.warmup_steps
    progress = min(1.0, (step - schedule.warmup_steps) / max(1, schedule.decay_steps - schedule.warmup_steps))
    return float(schedule.decay_lr + (schedule.peak_lr - schedule.decay_lr) * (1 + np.cos(np.pi * progress)) / 2)


def _next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


class _Stage1Model(torch.nn.Module):
    """Run both losses through DDP.forward, keeping export modules unwrapped."""

    def __init__(self, vla, readout, defaults):
        super().__init__()
        self.vla = vla
        self.readout = readout
        self.defaults = defaults

    def forward(self, observation, actions):
        if self.defaults.alpha > 0:
            per_element, prefix, mask = self.vla.compute_loss_with_prefix(
                observation, actions, train=self.defaults.augment_images
            )
            vla_loss = per_element.mean()
        else:
            with torch.no_grad():
                prefix, mask = self.vla.extract_prefix_embeddings(
                    observation, image_only=True, train=self.defaults.augment_images
                )
            vla_loss = prefix.new_zeros(())
        reconstruction, tokens = self.readout.reconstruction_loss(prefix.detach().float(), mask)
        loss = reconstruction + self.defaults.alpha * vla_loss
        return loss, reconstruction.detach(), vla_loss.detach(), tokens.detach(), prefix.detach()


def _rng_state():
    numpy_state = np.random.get_state()
    return {
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": (numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)), *numpy_state[2:]),
    }


def _restore_rng(saved):
    torch.set_rng_state(saved["rng"].cpu())
    if saved["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
    numpy_rng = saved["numpy_rng"]
    np.random.set_state((numpy_rng[0], numpy_rng[1].cpu().numpy().astype(np.uint32), *numpy_rng[2:]))


def _wrap(module, device):
    if not any(parameter.requires_grad for parameter in module.parameters()):
        return module
    return torch.nn.parallel.DistributedDataParallel(
        module,
        device_ids=[device.index] if getattr(device, "type", None) == "cuda" else None,
        find_unused_parameters=True,
    )


def _latest(output):
    candidates = [path for path in output.iterdir() if path.name.isdecimal() and (path / "training_state.pt").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No resumable Stage 1 checkpoint in {output}")
    return max(candidates, key=lambda path: int(path.name))


def train(config: train_config.TrainConfig, *, defaults: ReadoutDefaults | None = None):
    defaults = ReadoutDefaults(alpha=config.rlt.alpha) if defaults is None else defaults
    if config.mode != "rlt":
        raise ValueError("Readout backend requires mode=rlt")
    validate_dataset(config)
    world_size = train_config.require_world_size(config)
    use_ddp = world_size > 1
    if use_ddp:
        from openpi.training.vla import cleanup_ddp
        from openpi.training.vla import setup_ddp

        _, _, device = setup_ddp()
        if config.rlt.device == "cpu":
            raise ValueError("train.num_gpus > 1 cannot use rlt.device: cpu")
        is_main = dist.get_rank() == 0
    else:
        is_main = True
        device = config.rlt.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.rlt.vla_checkpoint is None:
        raise ValueError("Set rlt.vla_checkpoint to an existing PyTorch VLA checkpoint directory")
    checkpoint = train_config.local_path(config.rlt.vla_checkpoint)
    weight_path = checkpoint / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"PyTorch VLA checkpoint missing: {weight_path}")
    output = config.readout_dir
    if output == checkpoint or output in checkpoint.parents:
        raise ValueError("Readout output must not overwrite the VLA checkpoint or its parent")
    if output.exists() and any(output.iterdir()) and not (config.overwrite or config.resume):
        raise FileExistsError(f"Readout output exists: {output}; choose another output or set overwrite=true")
    if not isinstance(
        config.data,
        train_config.LeRobotFrankaDataConfig
        | train_config.LeRobotFrankaEEDataConfig
        | train_config.LeRobotFrankaJointDataConfig,
    ):
        raise ValueError("Readout requires Franka external/wrist views without finger-camera inputs")
    data = dataclasses.replace(
        config.data, assets=dataclasses.replace(config.data.assets, assets_dir=str(checkpoint / "assets"))
    )
    config = dataclasses.replace(config, data=data)
    if data.create(config.assets_dirs, config.model).norm_stats is None:
        raise ValueError(f"VLA normalization stats missing under {checkpoint / 'assets'}")
    if is_main and config.overwrite and not config.resume and output.exists():
        shutil.rmtree(output)
    if use_ddp:
        dist.barrier()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    vla = config.model.load_pytorch(config, str(weight_path)).to(device)
    if defaults.alpha == 0:
        vla.requires_grad_(requires_grad=False)
        vla.eval()
    else:
        vla.train()
    readout_config = RLTokenConfig(
        input_dim=vla.llm.configs[0].width,
        embed_dim=config.rlt.embed_dim,
        num_rl_tokens=config.rlt.num_tokens,
        num_layers=config.rlt.num_layers,
        num_heads=config.rlt.num_heads,
        max_prefix_len=config.rlt.max_prefix_len,
        use_mask=defaults.use_mask,
    )
    readout = RLTokenReadout(readout_config).to(device).train()
    parameters = [
        parameter for module in (vla, readout) for parameter in module.parameters() if parameter.requires_grad
    ]
    schedule = dataclasses.replace(config.lr_schedule, peak_lr=config.rlt.learning_rate)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=_lr(0, schedule),
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    ema = {
        name: {key: value.detach().clone() for key, value in module.state_dict().items()}
        for name, module in (("vla", vla), ("readout", readout))
    }
    settings = {
        "defaults": dataclasses.asdict(defaults),
        "schedule": dataclasses.asdict(schedule),
        "optimizer": dataclasses.asdict(config.optimizer),
        "readout": dataclasses.asdict(readout_config),
        "model": dataclasses.asdict(config.model),
        "precision": config.pytorch_training_precision,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "micro_batch_size": config.micro_batch_size,
        "num_gpus": config.num_gpus,
        "dataset": config.data.repo_id,
        "data": train_config.to_dict(config)["data"],
        "num_workers": config.num_workers,
    }
    start_step = 0
    saved = None
    if config.resume:
        saved = torch.load(_latest(output) / "training_state.pt", map_location=device, weights_only=True)
        saved_settings = {**saved["settings"]}
        saved_settings.setdefault("num_gpus", 1)  # Checkpoints predating DDP were single-process.
        saved_settings["model"] = {**saved_settings["model"]}
        saved_settings["model"].setdefault("lora_mode", "lora_only")
        if saved_settings != settings:
            raise ValueError("Stage 1 resume settings differ from the saved training state")
        vla.load_state_dict(saved["vla"], strict=True)
        readout.load_state_dict(saved["readout"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        ema, start_step = saved["ema"], saved["step"]
    if start_step > config.num_train_steps:
        raise ValueError("num_train_steps is smaller than the resumed step")
    training_model = _Stage1Model(vla, readout, defaults)
    if use_ddp:
        training_model = _wrap(training_model, device)
        # Initialize identical weights, then use independent augmentation/noise streams.
        torch.manual_seed(config.seed + dist.get_rank())
        np.random.seed(config.seed + dist.get_rank())
    micro_batch, accumulation = train_config.gradient_accumulation(config, world_size)
    loader = data_loader.create_data_loader(config, shuffle=True, batch_size=micro_batch * world_size)
    iterator = iter(loader)
    # TorchDataLoader owns a separately seeded generator. Replaying consumed
    # batches restores its position without consuming training noise/augmentation RNG.
    for _ in range(start_step * accumulation):
        _, iterator = _next(iterator, loader)
    if saved is not None:
        if "rng_by_rank" in saved:
            if len(saved["rng_by_rank"]) != world_size:
                raise ValueError("Stage 1 resume requires the saved number of ranks")
            _restore_rng(saved["rng_by_rank"][dist.get_rank() if use_ddp else 0])
        elif use_ddp:
            raise ValueError("Distributed Stage 1 checkpoint is missing per-rank RNG states")
        else:
            _restore_rng(saved)
        del saved
    metadata = config.policy_metadata or {}
    action_space = metadata.get(
        "action_space", "joint" if isinstance(data, train_config.LeRobotFrankaJointDataConfig) else "ee"
    )
    action_dim = metadata.get("action_dim", 8 if action_space == "joint" else 7)
    manifest = {
        "format": "pytorch-rl-token-v1",
        "architecture": "openpi-rlt-cross-attention-v1",
        "config": config.name,
        "readout_config": dataclasses.asdict(readout_config),
        "image_only": True,
        "mask_protocol": "valid-image-patches-v1" if defaults.use_mask else "unmasked-image-patches-v1",
        "camera_protocol": "franka-droid-external-and-wrist-no-finger-v1",
        "action_horizon": vla.action_horizon,
        "model_action_dim": vla.action_dim,
        "action_space": action_space,
        "action_representation": metadata.get("action_representation", action_space),
        "physical_action_dim": action_dim,
        "joint_training": defaults.alpha > 0,
        "stage1_settings": settings,
        "ema_decay": defaults.ema_decay,
    }
    if action_space == "joint" and action_dim == 8:
        manifest["physical_joint_dim"] = 7

    def save(step):
        local_rng = _rng_state()
        rng_by_rank = [local_rng]
        if use_ddp:
            rng_by_rank = [None] * world_size if is_main else None
            dist.gather_object(local_rng, rng_by_rank, dst=0)
        if not is_main:
            return
        output.mkdir(parents=True, exist_ok=True)
        directory = output / str(step)
        temporary = Path(tempfile.mkdtemp(prefix=".stage1-", dir=output))
        try:
            for name, filename in (("vla", "model.safetensors"), ("readout", "readout.safetensors")):
                save_file(
                    {key: value.detach().cpu().contiguous() for key, value in ema[name].items()}, temporary / filename
                )
            torch.save(
                {
                    "step": step,
                    "settings": settings,
                    "vla": vla.state_dict(),
                    "readout": readout.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema": ema,
                    **local_rng,  # Preserve compatibility with single-process checkpoints.
                    "rng_by_rank": rng_by_rank,
                },
                temporary / "training_state.pt",
            )
            if (checkpoint / "assets").exists():
                shutil.copytree(checkpoint / "assets", temporary / "assets")
            (temporary / "manifest.json").write_text(json.dumps({**manifest, "step": step}, indent=2) + "\n")
            train_config.save_config(config, temporary)
            if directory.exists():
                shutil.rmtree(directory)
            temporary.rename(directory)
            # The root exports the latest matched EMA VLA/readout pair. Resume
            # always uses a complete numeric snapshot, not partially copied exports.
            for filename in ("model.safetensors", "readout.safetensors", "manifest.json", "config.yaml"):
                staged = output / ("." + filename)
                shutil.copyfile(directory / filename, staged)
                staged.replace(output / filename)
            if (directory / "assets").exists():
                shutil.copytree(directory / "assets", output / "assets", dirs_exist_ok=True)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    tracking = init_tracking(config, output, resuming=config.resume) if is_main else None
    last_saved = None
    for step in range(start_step, config.num_train_steps):
        learning_rate = _lr(step, schedule)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        totals = []
        for micro_index in range(accumulation):
            (observation, actions), iterator = _next(iterator, loader)
            observation = Observation.from_observation_like(observation)
            observation = Observation.from_dict(_device_tree(observation.to_dict(), device))
            if defaults.alpha > 0:
                actions = actions.to(device=device, dtype=torch.float32)
            # All-reduce once per optimizer step, after the last micro-batch.
            sync = training_model.no_sync() if use_ddp and micro_index + 1 < accumulation else contextlib.nullcontext()
            with sync:
                loss, reconstruction, vla_loss, tokens, prefix = training_model(observation, actions)
                (loss / accumulation).backward()
            totals.append((loss.detach(), reconstruction, vla_loss, tokens))
        torch.nn.utils.clip_grad_norm_(parameters, config.optimizer.clip_gradient_norm)
        optimizer.step()
        loss = torch.stack([item[0] for item in totals]).mean()
        reconstruction = torch.stack([item[1] for item in totals]).mean()
        vla_loss = torch.stack([item[2] for item in totals]).mean()
        tokens = totals[-1][3]
        with torch.no_grad():
            for name, module in (("vla", vla), ("readout", readout)):
                for key, value in module.state_dict().items():
                    if value.is_floating_point():
                        ema[name][key].lerp_(value, 1 - defaults.ema_decay)
                    else:
                        ema[name][key].copy_(value)
        metrics = {
            "loss": float(loss.detach()),
            "rlt_loss": float(reconstruction.detach()),
            "vla_loss": float(vla_loss.detach()),
            "learning_rate": learning_rate,
            "token_std": float(tokens.std()),
        }
        manifest.update(prefix_shape=list(prefix.shape[1:]), final_reconstruction_mse=metrics["rlt_loss"])
        if is_main and ((step + 1) % config.log_interval == 0 or step == start_step):
            print(json.dumps({"step": step + 1, **metrics}))
            if tracking is not None:
                tracking.log(metrics, step=step + 1)
        if (step + 1) % config.save_interval == 0:
            save(step + 1)
            last_saved = step + 1
            if use_ddp:
                dist.barrier()
    if last_saved != config.num_train_steps:
        if is_main and start_step == config.num_train_steps:
            manifest.update(json.loads((_latest(output) / "manifest.json").read_text()))
        save(config.num_train_steps)
    if use_ddp:
        dist.barrier()
    if tracking is not None:
        tracking.finish()
    if use_ddp:
        cleanup_ddp()

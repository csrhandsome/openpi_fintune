"""Train an image JEPA on verified, local LeRobot v3 tactile videos."""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from force_encoder.data import BalancedSourceSampler
from force_encoder.data import LeRobotTactileDataset
from force_encoder.model import ImageJEPA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config/train/force_encoder_jepa.yaml"


def read_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    for source in config["data"]["sources"]:
        root = Path(source["root"]).expanduser()
        source["root"] = str(root if root.is_absolute() else PROJECT_ROOT / root)
    output = Path(config["output_dir"]).expanduser()
    config["output_dir"] = str(output if output.is_absolute() else PROJECT_ROOT / output)
    return config


def make_dataset(config: dict, split: str) -> LeRobotTactileDataset:
    return LeRobotTactileDataset(
        **config["data"], image_size=config["model"]["image_size"], split=split, seed=config["seed"]
    )


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def amp_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError("precision must be bf16 or fp32")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; set precision=fp32")
    return torch.autocast(device.type, dtype=torch.bfloat16)


def memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    gib = 1024**3
    return {
        "cuda_allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "cuda_reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }


def schedule(step: int, cfg: dict) -> tuple[float, float, float]:
    progress = min(step / max(1, cfg["max_steps"] - 1), 1.0)
    weight_decay = cfg["weight_decay"] + progress * (cfg["final_weight_decay"] - cfg["weight_decay"])
    momentum = cfg["ema_start"] + progress * (cfg["ema_end"] - cfg["ema_start"])
    if step < cfg["warmup_steps"]:
        lr = cfg["learning_rate"] * (step + 1) / max(1, cfg["warmup_steps"])
    else:
        decay = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"] - 1)
        lr = cfg["min_learning_rate"] + 0.5 * (cfg["learning_rate"] - cfg["min_learning_rate"]) * (
            1 + math.cos(math.pi * min(decay, 1.0))
        )
    return lr, weight_decay, momentum


def save_checkpoint(path: Path, model: ImageJEPA, optimizer, step: int, config: dict, best_loss: float) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.model_config,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "best_loss": best_loss,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(),
        },
        temporary,
    )
    temporary.replace(path)


@torch.no_grad()
def validate(model: ImageJEPA, loader: DataLoader, device: torch.device, cfg: dict) -> dict:
    model.eval()
    generator = torch.Generator().manual_seed(12345)
    totals = {}
    samples = 0
    # Fixed validation masks without perturbing training's mask RNG.
    for index, batch in enumerate(loader):
        if index >= cfg["val_batches"]:
            break
        images = batch.to(device, non_blocking=True)
        with amp_context(device, cfg["precision"]):
            metrics = model(images, generator=generator)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * len(images)
        samples += len(images)
    model.train()
    if samples == 0:
        raise ValueError("Validation produced no images")
    return {f"val/{key}": value / samples for key, value in totals.items()}


def run(config: dict, *, resume: str | None = None, inspect: bool = False) -> None:
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    train_data = make_dataset(config, "train")
    val_data = make_dataset(config, "val")
    print(json.dumps({"train": train_data.summary, "val": val_data.summary}, indent=2), flush=True)
    if inspect:
        # Decode one sample per source; only shape/range statistics leave this process.
        for name, start, _end in train_data.source_ranges:
            image = train_data[start]
            print(
                json.dumps({"source": name, "shape": list(image.shape), "finite": bool(image.isfinite().all())}),
                flush=True,
            )
        return
    cfg = config["training"]
    if cfg["max_steps"] <= 0 or cfg["steps_per_epoch"] <= 0 or cfg["batch_size"] <= 0:
        raise ValueError("max_steps, steps_per_epoch and batch_size must be positive")
    if min(cfg[key] for key in ("validate_every", "val_batches", "save_every", "log_every")) <= 0:
        raise ValueError("Validation, checkpoint and log intervals must be positive")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a small test")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists() and resume is None:
        raise FileExistsError(f"{output}/latest.pt exists; use --resume or a new --output-dir")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "data_summary.json").write_text(
        json.dumps({"train": train_data.summary, "val": val_data.summary}, indent=2)
    )
    sampler = BalancedSourceSampler(
        train_data, num_samples=cfg["steps_per_epoch"] * cfg["batch_size"], seed=config["seed"]
    )
    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(config["seed"]),
    }
    if cfg["num_workers"]:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    train_loader = DataLoader(train_data, sampler=sampler, drop_last=True, **loader_kwargs)
    # Balance validation too, so a large dataset cannot dominate the reported metric.
    val_sampler = BalancedSourceSampler(val_data, num_samples=cfg["val_batches"] * cfg["batch_size"], seed=12345)
    val_loader = DataLoader(val_data, sampler=val_sampler, **loader_kwargs)
    model = ImageJEPA(**config["model"]).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    decay = [p for p in parameters if p.ndim > 1]
    no_decay = [p for p in parameters if p.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}], lr=cfg["learning_rate"]
    )
    start_step, best_loss = 0, math.inf
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != model.model_config:
            raise ValueError("Resume model configuration differs from checkpoint")
        if checkpoint["config"]["data"] != config["data"] or checkpoint["config"]["seed"] != config["seed"]:
            raise ValueError("Resume data paths, sampling and split seed must match checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step, best_loss = checkpoint["step"], checkpoint["best_loss"]
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": sum(p.numel() for p in parameters),
                "encoder_parameters": sum(p.numel() for p in model.context_encoder.parameters()),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "start_step": start_step,
                "initialization": "resume" if resume else "random",
                "resume_checkpoint": resume,
            }
        ),
        flush=True,
    )
    step = start_step
    model.train()
    started = time.monotonic()
    with (output / "metrics.jsonl").open("a") as log:
        while step < cfg["max_steps"]:
            epoch = step // cfg["steps_per_epoch"]
            sampler.set_epoch(epoch)
            # Bound the final partial epoch so workers cannot keep decoding
            # unused NAS batches after the requested last optimizer step.
            sampler.num_samples = (
                min(cfg["steps_per_epoch"], cfg["max_steps"] - epoch * cfg["steps_per_epoch"]) * cfg["batch_size"]
            )
            skip_batches = step % cfg["steps_per_epoch"]
            for batch_index, batch in enumerate(train_loader):
                if batch_index < skip_batches:
                    continue
                images = batch.to(device, non_blocking=True)
                lr, weight_decay, momentum = schedule(step, cfg)
                for index, group in enumerate(optimizer.param_groups):
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay if index == 0 else 0.0
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, cfg["precision"]):
                    metrics = model(images)
                if not torch.isfinite(metrics["loss"]):
                    raise FloatingPointError(f"Nonfinite JEPA loss at step {step}")
                metrics["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                model.update_teacher(momentum)
                step += 1
                if step == 1 or step % cfg["log_every"] == 0 or step == cfg["max_steps"]:
                    record = {
                        "step": step,
                        **{key: float(value) for key, value in metrics.items()},
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "momentum": momentum,
                        "grad_norm": float(grad_norm),
                        "elapsed_seconds": time.monotonic() - started,
                        **memory_stats(device),
                    }
                    print(json.dumps(record), flush=True)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                final = step == cfg["max_steps"]
                if step % cfg["validate_every"] == 0 or final:
                    record = {"step": step, **validate(model, val_loader, device, cfg), **memory_stats(device)}
                    print(json.dumps(record), flush=True)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    if record["val/loss"] < best_loss:
                        best_loss = record["val/loss"]
                        save_checkpoint(output / "best.pt", model, optimizer, step, config, best_loss)
                if step % cfg["save_every"] == 0 or final:
                    save_checkpoint(output / "latest.pt", model, optimizer, step, config, best_loss)
                if final:
                    break
    print(f"Training completed at step {step}; checkpoint: {output / 'latest.pt'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--output-dir")
    for key in ("max-steps", "batch-size", "num-workers", "val-batches", "steps-per-epoch"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--max-episodes-per-source", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    args = parser.parse_args()
    config = read_config(args.config)
    for key in ("max_steps", "batch_size", "num_workers", "val_batches", "steps_per_epoch", "device", "precision"):
        if getattr(args, key) is not None:
            config["training"][key] = getattr(args, key)
    if args.output_dir:
        config["output_dir"] = str(Path(args.output_dir).resolve())
    if args.max_episodes_per_source is not None:
        config["data"]["max_episodes_per_source"] = args.max_episodes_per_source
    run(config, resume=args.resume, inspect=args.inspect)


if __name__ == "__main__":
    main()

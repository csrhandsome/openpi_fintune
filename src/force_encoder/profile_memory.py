"""Measure complete JEPA optimizer steps in isolated CUDA processes.

The sweep caches two real batches locally, so every model sees identical data.
Repeated batches measure memory and compute, not convergence or NAS throughput.
No activation checkpointing, accumulation, parameter offload or dtype conversion
is applied: the FP32 parameters and BF16 autocast match ``force_encoder.train``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from force_encoder.data import BalancedSourceSampler
from force_encoder.model import ImageJEPA
from force_encoder.model import sample_masks
from force_encoder.train import PROJECT_ROOT
from force_encoder.train import amp_context
from force_encoder.train import make_dataset
from force_encoder.train import read_config
from force_encoder.train import schedule
from force_encoder.train import seed_worker

GIB = 1024**3


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_sweep(path: str | Path) -> dict:
    sweep = yaml.safe_load(Path(path).read_text())
    if sweep["batch_size"] <= 0 or sweep["warmup_steps"] < 0 or sweep["steps"] <= sweep["warmup_steps"]:
        raise ValueError("Require positive batch_size and steps > warmup_steps >= 0")
    if not 0 < sweep["target_utilization"] <= 1:
        raise ValueError("target_utilization must be in (0, 1]")
    names = [profile["name"] for profile in sweep["profiles"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Profile names must be nonempty and unique")
    if any(not name.replace("_", "").replace("-", "").isalnum() for name in names):
        raise ValueError("Profile names must be filename-safe")
    return sweep


def profile_config(base: dict, sweep: dict, profile: dict) -> dict:
    config = copy.deepcopy(base)
    config["model"].update(profile["model"])
    config["training"]["batch_size"] = sweep["batch_size"]
    return config


def mask_extremes(model_config: dict, seed: int, candidates: int = 32) -> dict:
    """Find actual reproducible context-length extrema among candidate masks."""
    keys = ("num_local_targets", "num_global_targets", "local_scale", "global_scale", "max_target_overlap")
    kwargs = {key: model_config[key] for key in keys if key in model_config}
    counts = []
    for candidate in range(seed, seed + candidates):
        masks = sample_masks(
            model_config["image_size"] // model_config["patch_size"],
            generator=torch.Generator().manual_seed(candidate),
            **kwargs,
        )
        counts.append((len(masks.context), candidate))
    minimum, maximum = min(counts), max(counts)
    return {
        "candidate_count": candidates,
        "min_context_tokens": minimum[0],
        "min_context_seed": minimum[1],
        "max_context_tokens": maximum[0],
        "max_context_seed": maximum[1],
        "qualification": "Extrema among sampled masks, not an exhaustive mathematical bound",
    }


def batch_metadata(dataset, indices: list[int]) -> dict:
    return {
        "summary": dataset.summary,
        "samples": [
            {
                key: value
                for key, value in dataset.sample_metadata(index).items()
                if key in {"source", "episode_index", "video_key", "frame_index"}
            }
            for index in indices
        ],
    }


def validate_cached_batches(base: dict, sweep: dict, output: Path) -> Path:
    """Verify cache shape and reproducible data IDs without decoding any frames."""
    path = output / "batches.pt"
    cached = torch.load(path, map_location="cpu", weights_only=True)
    size = base["model"]["image_size"]
    expected = (sweep["batch_size"], 3, size, size)
    if set(cached) != {"train", "val"} or any(tuple(batch.shape) != expected for batch in cached.values()):
        raise ValueError(f"Existing cache does not match expected batch shape {expected}")
    metadata = json.loads((output / "batch_metadata.json").read_text())
    for split, seed in (("train", base["seed"]), ("val", 12345)):
        dataset = make_dataset(base, split)
        try:
            indices = list(BalancedSourceSampler(dataset, num_samples=sweep["batch_size"], seed=seed))
            expected_metadata = batch_metadata(dataset, indices)
            if json.dumps(metadata[split], sort_keys=True) != json.dumps(expected_metadata, sort_keys=True):
                raise ValueError(f"Cached {split} data IDs/configuration changed; use a new output directory")
        finally:
            dataset.close()
    return path


def prepare_batches(base: dict, sweep: dict, output: Path) -> Path:
    """Decode exactly one balanced batch per split; workers cannot prefetch more."""
    batches, metadata = {}, {}
    for split, seed in (("train", base["seed"]), ("val", 12345)):
        dataset = make_dataset(base, split)
        try:
            sampler = BalancedSourceSampler(dataset, num_samples=sweep["batch_size"], seed=seed)
            indices = list(sampler)
            workers = base["training"].get("num_workers", 4)
            kwargs = {"multiprocessing_context": "spawn", "prefetch_factor": 1} if workers else {}
            loader = DataLoader(
                dataset,
                sampler=indices,
                batch_size=sweep["batch_size"],
                num_workers=workers,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(seed),
                **kwargs,
            )
            # Exhaust the finite loader to join its nonpersistent workers.
            decoded = list(loader)
            if len(decoded) != 1 or len(decoded[0]) != sweep["batch_size"]:
                raise ValueError(f"{split} did not yield one full batch")
            batches[split] = decoded[0]
            metadata[split] = batch_metadata(dataset, indices)
        finally:
            dataset.close()
    path = output / "batches.pt"
    torch.save(batches, path)
    (output / "batch_metadata.json").write_text(json.dumps(metadata, indent=2))
    return path


def annotate_result(result: dict, target: float = 0.9) -> dict:
    """Separate a completed computation from a suitable physical-memory fit.

    Some CUDA environments (including WSL) allow virtual GPU allocations above
    physical capacity. Completing steps therefore does not prove healthy fit.
    These statistics suggest overcommit/paging; they cannot establish paging.
    """
    result = dict(result)
    baseline = result.get("baseline", {})
    nonallocator = max(0, baseline.get("sampled_device_used_gib", 0) - baseline.get("reserved_gib", 0))
    result["sampled_peak_device_used_gib"] = max(
        result.get("sampled_peak_device_used_gib", 0),
        max((stage.get("sampled_device_used_gib", 0) for stage in result.get("stage_peaks", [])), default=0),
    )
    virtual_peak = result.get("peak_reserved_gib", 0) + nonallocator
    result["virtual_reserved_plus_baseline_gib"] = virtual_peak
    warmed_times = [step["seconds"] for step in result.get("steps", []) if step.get("warmed")]
    result["warmed_mean_step_seconds"] = sum(warmed_times) / len(warmed_times) if warmed_times else None
    total = result.get("device_total_gib", 0)
    overcommitted = bool(total and (virtual_peak > total or result.get("peak_allocated_gib", 0) > total))
    result["virtual_allocator_exceeds_physical"] = overcommitted
    if overcommitted:
        suitability = "overcommit_or_paging_suspected"
    elif result["status"] != "ok" or not total:
        suitability = "not_completed"
    elif result["sampled_peak_device_used_gib"] / total > 0.95:
        suitability = "high_pressure"
    elif result.get("estimated_peak_device_used_gib", virtual_peak) / total <= target:
        suitability = "within_budget"
    else:
        suitability = "above_budget"
    result["memory_suitability"] = suitability
    result["memory_qualification"] = (
        "Allocator allocated/reserved values may describe virtual allocations and exceed physical VRAM under WSL. "
        "Sampled device use comes from mem_get_info at phase boundaries and can miss transient physical peaks. "
        "Overcommit/paging is suspected from these observations, not proven."
    )
    return result


def recommend(results: list[dict], target: float) -> str | None:
    candidates = [
        result
        for result in (annotate_result(item, target) for item in results)
        if result["status"] == "ok"
        and result["memory_suitability"] == "within_budget"
        and result["estimated_peak_device_used_gib"] / result["device_total_gib"] <= target
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda result: result["parameters"]["encoder"])["profile"]


def write_summary(results: list[dict], sweep: dict, output: Path) -> None:
    results = [annotate_result(result, sweep["target_utilization"]) for result in results]
    for result in results:
        (output / f"{result['profile']}.json").write_text(json.dumps(result, indent=2))
    summary = {
        "batch_size": sweep["batch_size"],
        "steps": sweep["steps"],
        "target_utilization": sweep["target_utilization"],
        "recommended_profile": recommend(results, sweep["target_utilization"]),
        "results": results,
        "limitations": [
            "Repeated real batches measure memory/compute, not convergence or data throughput.",
            "Masks include sampled context extrema; unseen masks may use more memory.",
            "Device-used peak is estimated from allocator reserve plus baseline nonallocator usage; other GPU processes can change it.",
            "Under WSL, virtual allocated/reserved statistics can exceed physical VRAM; successful steps do not prove a healthy memory fit.",
            "Overcommit/paging labels indicate suspicion, not proof. Physical device use is sampled at phase boundaries.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        f"Fixed batch {sweep['batch_size']}; {sweep['steps']} complete optimizer steps per profile.",
        "",
        "| Profile | Status | Memory suitability | Encoder params | Allocated GiB | Reserved GiB | Sampled device GiB | Estimated demand GiB | Warm step s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {result['profile']} | {result['status']} | {result['memory_suitability']} | {result.get('parameters', {}).get('encoder', '—')} "
        f"| {result.get('peak_allocated_gib', 0):.3f} | {result.get('peak_reserved_gib', 0):.3f} "
        f"| {result['sampled_peak_device_used_gib']:.3f} | {result.get('estimated_peak_device_used_gib', 0):.3f} "
        f"| {result['warmed_mean_step_seconds'] or 0:.3f} |"
        for result in results
    )
    lines += [
        "",
        f"Recommended at {sweep['target_utilization']:.0%} capacity: {summary['recommended_profile'] or 'none'}.",
    ]
    lines += ["", *summary["limitations"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def run_worker(sweep: dict, profile: dict, cache: Path, output: Path) -> dict:
    config = profile_config(read_config(project_path(sweep["base_config"])), sweep, profile)
    cfg = config["training"]
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device(cfg["device"])
    result = {
        "profile": profile["name"],
        "status": "error",
        "completed_steps": 0,
        "stage": "initialization",
        "model_config": config["model"],
        "batch_size": sweep["batch_size"],
        "precision": cfg["precision"],
        "parameter_dtype": "float32",
        "stage_peaks": [],
        "steps": [],
        "gradient_accumulation": 1,
        "activation_checkpointing": False,
        "parameter_offload": False,
        "pytorch_version": torch.__version__,
    }
    baseline_nonallocator = 0.0

    def snapshot() -> dict:
        free, total = torch.cuda.mem_get_info(device)
        return {
            "allocated_gib": torch.cuda.memory_allocated(device) / GIB,
            "reserved_gib": torch.cuda.memory_reserved(device) / GIB,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
            "sampled_device_used_gib": (total - free) / GIB,
        }

    def phase(name, action):
        result["stage"] = name
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        value = action()
        torch.cuda.synchronize(device)
        result["stage_peaks"].append(
            {"stage": name, "step": result["completed_steps"] + 1, "seconds": time.monotonic() - started, **snapshot()}
        )
        return value

    print(json.dumps({"profile": profile["name"], "event": "start"}), flush=True)
    try:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Memory profiling requires a CUDA training device")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = snapshot()
        result["baseline"] = baseline
        result["device_total_gib"] = torch.cuda.get_device_properties(device).total_memory / GIB
        result["device_name"] = torch.cuda.get_device_name(device)
        baseline_nonallocator = max(0, baseline["sampled_device_used_gib"] - baseline["reserved_gib"])
        result["mask_search"] = mask_extremes(config["model"], config["seed"])
        cached = torch.load(cache, map_location="cpu", weights_only=True)
        expected = (sweep["batch_size"], 3, config["model"]["image_size"], config["model"]["image_size"])
        if any(tuple(cached[key].shape) != expected for key in ("train", "val")):
            raise ValueError(f"Cached batches must have shape {expected}")
        model = ImageJEPA(**config["model"])
        result["parameters"] = {
            "encoder": sum(p.numel() for p in model.context_encoder.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters()),
        }
        print(
            json.dumps({"profile": profile["name"], "event": "initialized_cpu", "parameters": result["parameters"]}),
            flush=True,
        )
        phase("model_to_device", lambda: model.to(device))
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in parameters if p.ndim > 1]},
                {"params": [p for p in parameters if p.ndim <= 1], "weight_decay": 0.0},
            ],
            lr=cfg["learning_rate"],
        )
        images = phase("train_batch_to_device", lambda: cached["train"].to(device))
        model.train()
        search = result["mask_search"]
        for step in range(sweep["steps"]):
            lr, weight_decay, momentum = schedule(step, cfg)
            for index, group in enumerate(optimizer.param_groups):
                group["lr"] = lr
                group["weight_decay"] = weight_decay if index == 0 else 0.0
            # Both extrema occur before and after optimizer state initialization.
            cycle = step % 3
            mask_seed = (
                search["min_context_seed"]
                if cycle == 0
                else (search["max_context_seed"] if cycle == 1 else config["seed"] + 1000 + step)
            )
            generator = torch.Generator().manual_seed(mask_seed)
            optimizer.zero_grad(set_to_none=True)

            def forward(generator=generator):
                with amp_context(device, cfg["precision"]):
                    metrics = model(images, generator=generator)
                if not bool(torch.isfinite(metrics["loss"])):
                    raise FloatingPointError("Nonfinite JEPA loss")
                return metrics

            metrics = phase("forward", forward)
            phase("backward", lambda metrics=metrics: metrics["loss"].backward())
            grad_norm = phase(
                "clip_grad",
                lambda: torch.nn.utils.clip_grad_norm_(parameters, cfg["grad_clip"], error_if_nonfinite=True),
            )
            phase("optimizer", optimizer.step)
            phase("ema", lambda momentum=momentum: model.update_teacher(momentum))
            record = {
                "step": step + 1,
                "mask_seed": mask_seed,
                "grad_norm": float(grad_norm),
                **{key: float(value) for key, value in metrics.items()},
            }
            step_stages = result["stage_peaks"][-5:]
            record.update(
                {
                    "peak_allocated_gib": max(item["peak_allocated_gib"] for item in step_stages),
                    "peak_reserved_gib": max(item["peak_reserved_gib"] for item in step_stages),
                    "seconds": sum(item["seconds"] for item in step_stages),
                    "warmed": step >= sweep["warmup_steps"],
                }
            )
            result["steps"].append(record)
            result["completed_steps"] = step + 1
            del metrics, grad_norm
            if step + 1 in {2, sweep["steps"]}:
                print(json.dumps({"profile": profile["name"], **record}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        images = None
        val_images = phase("val_batch_to_device", lambda: cached["val"].to(device))
        model.eval()

        def validation():
            with torch.no_grad(), amp_context(device, cfg["precision"]):
                metrics = model(val_images, generator=torch.Generator().manual_seed(12345))
            if not bool(torch.isfinite(metrics["loss"])):
                raise FloatingPointError("Nonfinite validation loss")
            return {key: float(value) for key, value in metrics.items()}

        result["validation"] = phase("validation", validation)
        result["status"] = "ok"
        result["stage"] = "complete"
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        if device.type == "cuda" and torch.cuda.is_initialized():
            result["stage_peaks"].append({"stage": result["stage"], **snapshot()})
        stages = result["stage_peaks"]
        result["peak_allocated_gib"] = max((x["peak_allocated_gib"] for x in stages), default=0.0)
        result["peak_reserved_gib"] = max((x["peak_reserved_gib"] for x in stages), default=0.0)
        result["estimated_peak_device_used_gib"] = max(
            baseline_nonallocator + result["peak_reserved_gib"],
            max((x["sampled_device_used_gib"] for x in stages), default=0.0),
        )
        result["first_step_peak_allocated_gib"] = result["steps"][0]["peak_allocated_gib"] if result["steps"] else None
        warmed = [item for item in result["steps"] if item["warmed"]]
        result["warmed_peak_allocated_gib"] = max((x["peak_allocated_gib"] for x in warmed), default=None)
        result = annotate_result(result, sweep["target_utilization"])
        (output / f"{profile['name']}.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    "profile": profile["name"],
                    "event": "complete",
                    "status": result["status"],
                    "peak_allocated_gib": result["peak_allocated_gib"],
                    "peak_reserved_gib": result["peak_reserved_gib"],
                }
            ),
            flush=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Explicit temporary sweep configuration")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--append", action="store_true", help="Reuse verified batches and preserve previous results")
    parser.add_argument("--worker-profile", help=argparse.SUPPRESS)
    parser.add_argument("--batch-cache", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    sweep = read_sweep(args.config)
    output = project_path(args.output_dir or sweep["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    profiles = {item["name"]: item for item in sweep["profiles"]}
    if args.worker_profile:
        result = run_worker(sweep, profiles[args.worker_profile], args.batch_cache, output)
        raise SystemExit(0 if result["status"] == "ok" else 2)
    names = args.profiles or list(profiles)
    if any(name not in profiles for name in names):
        parser.error(f"Unknown profiles: {set(names) - profiles.keys()}")
    base = read_config(project_path(sweep["base_config"]))
    results = []
    if args.append:
        previous = json.loads((output / "summary.json").read_text())
        if previous["batch_size"] != sweep["batch_size"] or previous["steps"] != sweep["steps"]:
            parser.error("Appending requires the same batch size and optimizer-step count")
        results = previous["results"]
        cache = validate_cached_batches(base, sweep, output)
        print(json.dumps({"event": "reuse_verified_batches", "batch_size": sweep["batch_size"]}), flush=True)
    else:
        print(json.dumps({"event": "prepare_real_batches", "batch_size": sweep["batch_size"]}), flush=True)
        cache = prepare_batches(base, sweep, output)
    (output / "sweep.yaml").write_text(yaml.safe_dump(sweep, sort_keys=False))
    (output / "base_config.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    for name in names:
        result_path = output / f"{name}.json"
        # Never interpret a stale successful result as this subprocess's result.
        result_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "force_encoder.profile_memory",
            "--config",
            str(args.config.resolve()),
            "--output-dir",
            str(output),
            "--worker-profile",
            name,
            "--batch-cache",
            str(cache),
        ]
        print(json.dumps({"event": "launch", "profile": name}), flush=True)
        with (output / f"{name}.log").open("w") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
                fallback = f"Worker exited {process.returncode} without a result"
            except subprocess.TimeoutExpired:
                fallback = "Worker timed out after 1800 seconds"
        result = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {
                "profile": name,
                "status": "error",
                "error": fallback,
                "completed_steps": 0,
            }
        )
        results = [existing for existing in results if existing["profile"] != name] + [result]
        write_summary(results, sweep, output)
        print(
            json.dumps(
                {
                    "event": "result",
                    "profile": name,
                    "status": result["status"],
                    "peak_allocated_gib": result.get("peak_allocated_gib"),
                }
            ),
            flush=True,
        )
        if result["status"] in {"oom", "error"}:
            break


if __name__ == "__main__":
    main()

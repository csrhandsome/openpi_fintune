"""Benchmark real-checkpoint Policy.infer with independent compile/Triton switches."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
import triton

from openpi.models_pytorch.compile_inference import configure_torch_compile
from openpi.models_pytorch.triton_inference import configure_triton
from openpi.policies import policy_config
from openpi.serving.runner import Args
from openpi.serving.runner import latest_checkpoint
from openpi.serving.runner import load_settings
from openpi.training import config as training_config

MODES = {
    "baseline": (False, False),
    "triton": (False, True),
    "compile": (True, False),
    "compile_triton": (True, True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/server/openpi.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--output", default="/tmp/openpi-compile-benchmark.json")
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.rounds) < 1:
        parser.error("warmup, repeats and rounds must be positive")
    settings = load_settings(Args(config=args.config, checkpoint=args.checkpoint))
    config = training_config.get_config(settings.train_config)
    checkpoint = settings.checkpoint or latest_checkpoint(config)
    torch.manual_seed(42)
    policy = policy_config.create_trained_policy(config, checkpoint, pytorch_device="cuda")
    model = policy._model  # noqa: SLF001 -- compare one loaded model with identical weights
    rng = np.random.default_rng(42)

    def make_input():
        obs = {
            "observation.exterior_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation.wrist_image_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation.joint_position": np.zeros(7, np.float32),
            "observation.gripper_position": np.zeros(1, np.float32),
            "prompt": "pick up the object",
        }
        noise = rng.standard_normal((config.model.action_horizon, config.model.action_dim)).astype(np.float32)
        return obs, noise

    def select(mode):
        compiled, fused = MODES[mode]
        configure_torch_compile(model, enabled=False)
        configure_triton(model, enabled=fused)
        configure_torch_compile(model, enabled=compiled)

    inputs = [make_input() for _ in range(3)]
    obs, noise = inputs[0]
    outputs, first_call_ms = {}, {}
    for mode in MODES:
        print(f"Warmup {mode}", flush=True)
        select(mode)
        torch.cuda.synchronize()
        started = time.perf_counter()
        policy.infer(obs, noise=noise)
        torch.cuda.synchronize()
        first_call_ms[mode] = (time.perf_counter() - started) * 1000
        for _ in range(args.warmup):
            policy.infer(obs, noise=noise)
        outputs[mode] = [policy.infer(item, noise=z)["actions"].copy() for item, z in inputs]
        torch.cuda.synchronize()

    correctness = {}
    for mode in MODES:
        deltas = [
            np.abs(actual - expected) for actual, expected in zip(outputs[mode], outputs["baseline"], strict=True)
        ]
        if not all(np.isfinite(actual).all() for actual in outputs[mode]):
            raise AssertionError(f"Nonfinite actions in {mode}")
        within_tolerance = [
            delta <= 2e-3 + 2e-3 * np.abs(expected) for delta, expected in zip(deltas, outputs["baseline"], strict=True)
        ]
        correctness[mode] = {
            "passed": bool(np.all(within_tolerance)),
            "violations": int(np.size(within_tolerance) - np.count_nonzero(within_tolerance)),
            "elements": int(np.size(within_tolerance)),
            "max_action_abs_diff_by_dimension": np.max(deltas, axis=(0, 1)).tolist(),
            "max_action_abs_diff": float(np.max(deltas)),
            "mean_action_abs_diff": float(np.mean(deltas)),
        }

    latencies = {mode: [] for mode in MODES}
    modes = list(MODES)
    for round_index in range(args.rounds):
        # Rotate order so every mode occupies each position in four rounds.
        for mode in modes[round_index % len(modes) :] + modes[: round_index % len(modes)]:
            select(mode)
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                policy.infer(obs, noise=noise)
                torch.cuda.synchronize()
                latencies[mode].append((time.perf_counter() - started) * 1000)
    medians = {mode: statistics.median(values) for mode, values in latencies.items()}
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "checkpoint": str(checkpoint),
        "model": dataclasses.asdict(config.model),
        "input": "synthetic Franka RGB/state, fixed noise, real checkpoint; Policy.infer incl. transforms",
        "compile_options": {
            "mode": "default",
            "fullgraph": True,
            "dynamic": False,
            "emulate_precision_casts": True,
            "force_same_precision": True,
        },
        "compiled_methods": ["run_suffix"],
        "warmup_per_mode": args.warmup,
        "samples_per_mode": args.repeats * args.rounds,
        "correctness_inputs": len(inputs),
        "action_tolerance": {"atol": 2e-3, "rtol": 2e-3},
        "correctness": correctness,
        "first_call_ms_including_compile": first_call_ms,
        "median_ms": medians,
        "speedup_vs_baseline": {mode: medians["baseline"] / value for mode, value in medians.items()},
        "latencies_ms": latencies,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str), flush=True)
    if not all(check["passed"] for check in correctness.values()):
        raise SystemExit("Strict action tolerance failed; full correctness and latency results saved to " + args.output)


if __name__ == "__main__":
    main()

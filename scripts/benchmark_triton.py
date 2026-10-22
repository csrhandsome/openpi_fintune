"""Compare real-checkpoint policy latency with synchronized CUDA timing.

Example: python scripts/benchmark_triton.py --checkpoint checkpoints/.../30000
The same model, transformed inputs, fixed noise and existing torch.compile
settings are used for both modes. Compilation and warmup are excluded.
"""

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

from openpi.models_pytorch.triton_inference import configure_triton
from openpi.policies import policy_config
from openpi.serving.runner import Args
from openpi.serving.runner import latest_checkpoint
from openpi.serving.runner import load_settings
from openpi.training import config as training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/server/openpi.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", default="/tmp/openpi-triton-benchmark.json")
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.rounds) < 1:
        parser.error("warmup, repeats and rounds must be positive")
    settings = load_settings(Args(config=args.config, checkpoint=args.checkpoint))
    config = training_config.get_config(settings.train_config)
    checkpoint = settings.checkpoint or latest_checkpoint(config)
    torch.manual_seed(42)
    policy = policy_config.create_trained_policy(config, checkpoint, pytorch_device="cuda")
    rng = np.random.default_rng(42)
    # Use the public Franka policy API, including tokenization and normalization.
    obs = {
        "observation.exterior_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation.wrist_image_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation.joint_position": np.zeros(7, np.float32),
        "observation.gripper_position": np.zeros(1, np.float32),
        "prompt": "pick up the object",
    }
    noise = rng.standard_normal((config.model.action_horizon, config.model.action_dim)).astype(np.float32)
    model = policy._model  # noqa: SLF001 -- benchmark toggles one already-loaded model
    latencies = {False: [], True: []}
    outputs = {}
    for enabled in (False, True):
        configure_triton(model, enabled=enabled)
        for _ in range(args.warmup):
            outputs[enabled] = policy.infer(obs, noise=noise)["actions"]
        torch.cuda.synchronize()
    deltas = [np.abs(outputs[True] - outputs[False])]
    # Check physical, unnormalized actions after all ten denoising steps.
    np.testing.assert_allclose(outputs[True], outputs[False], atol=2e-3, rtol=2e-3)
    # Exercise additional images and noise without including them in timing.
    for _ in range(2):
        check_obs = {
            **obs,
            "observation.exterior_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation.wrist_image_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        }
        check_noise = rng.standard_normal(noise.shape).astype(np.float32)
        pair = []
        for enabled in (False, True):
            configure_triton(model, enabled=enabled)
            pair.append(policy.infer(check_obs, noise=check_noise)["actions"])
        np.testing.assert_allclose(pair[1], pair[0], atol=2e-3, rtol=2e-3)
        deltas.append(np.abs(pair[1] - pair[0]))
    for round_index in range(args.rounds):
        for enabled in (False, True) if round_index % 2 == 0 else (True, False):
            configure_triton(model, enabled=enabled)
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                policy.infer(obs, noise=noise)
                torch.cuda.synchronize()
                latencies[enabled].append((time.perf_counter() - started) * 1000)
    baseline, accelerated = (statistics.median(latencies[mode]) for mode in (False, True))
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "checkpoint": str(checkpoint),
        "model": dataclasses.asdict(config.model),
        "input": "synthetic Franka RGB/state, fixed noise, real checkpoint; Policy.infer incl. transforms",
        "warmup_per_mode": args.warmup,
        "samples_per_mode": args.repeats * args.rounds,
        "baseline_median_ms": baseline,
        "triton_median_ms": accelerated,
        "speedup": baseline / accelerated,
        "latency_reduction_percent": 100 * (1 - accelerated / baseline),
        "correctness_inputs": len(deltas),
        "action_tolerance": {"atol": 2e-3, "rtol": 2e-3},
        "max_action_abs_diff": float(np.max(deltas)),
        "mean_action_abs_diff": float(np.mean(deltas)),
        "latencies_ms": {str(key): value for key, value in latencies.items()},
    }
    Path(args.output).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Train B or C PyTorch chunk actor-critic from a validated contact replay journal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from openpi.force_rlt.learner import ChunkLearner
from openpi.force_rlt.learner import LearnerConfig
from openpi.force_rlt.networks import NetworkConfig
from openpi.force_rlt.replay import ReplayJournal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--group", choices=("B", "C"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-reference-offset-rad", type=float, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.max_reference_offset_rad <= 0:
        parser.error("steps, batch-size and max-reference-offset-rad must be positive")
    with args.replay.open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
    state = first["state"]
    network = NetworkConfig(
        z_rlt_dim=len(state["z_rlt"]),
        z_grip_dim=len(state["z_grip"]),
        horizon=len(state["ref_q"]),
        executed_steps=len(first["executed_q"]),
        max_reference_offset=args.max_reference_offset_rad,
    )
    journal = ReplayJournal(args.replay, network)
    records = [record for record in journal.read_all() if record.source != "HUMAN" and not record.intervention_flag]
    if not records:
        parser.error("Replay contains no autonomous transitions")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    learner = ChunkLearner(
        LearnerConfig(network=network, zero_grip=args.group == "B"),
        device=args.device,
    )
    for step in range(args.steps):
        indices = rng.integers(0, len(records), size=args.batch_size)
        metrics = learner.update([records[index] for index in indices])
        if (step + 1) % 100 == 0 or step == 0:
            print(f"step={step + 1} critic={metrics['critic_loss']:.5f} actor={metrics['actor_loss']:.5f}")
    learner.save(args.output)
    print(f"saved {args.group} actor: {args.output}")


if __name__ == "__main__":
    main()

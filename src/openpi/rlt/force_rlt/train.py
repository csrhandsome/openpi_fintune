"""Train a contact actor-critic from YAML or the legacy group CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from openpi.rlt.force_rlt.config import validate_joint_config
from openpi.rlt.force_rlt.learner import ChunkLearner
from openpi.rlt.force_rlt.learner import LearnerConfig
from openpi.rlt.force_rlt.networks import NetworkConfig
from openpi.rlt.force_rlt.replay import ReplayJournal


def train_from_config(config):
    """Train a contact actor selected by stage=force and force.enabled."""
    validate_joint_config(config)
    settings = config.rlt.force_stage
    if settings.policy == "vla":
        raise ValueError("force_stage.policy=vla uses a frozen VLA and has no actor to train")
    output = Path(settings.checkpoint_path)
    resume_path = config.rlt.actor_checkpoint or (str(output) if config.resume else None)
    if output.exists() and not config.overwrite and not resume_path:
        raise FileExistsError(f"Force actor output exists: {output}; set overwrite or rlt.actor_checkpoint")
    with Path(settings.replay_path).open(encoding="utf-8") as stream:
        first_line = next((line for line in stream if line.strip()), None)
    if first_line is None:
        raise ValueError("Force replay contains no transitions")
    first = json.loads(first_line)
    network = NetworkConfig(
        z_rlt_dim=config.rlt.embed_dim * config.rlt.num_tokens,
        z_grip_dim=config.rlt.force.feature_dim,
        horizon=config.model.action_horizon,
        executed_steps=len(first["executed_q"]),
        hidden_dim=settings.hidden_dim,
        layers=settings.layers,
        fixed_std=settings.fixed_std,
        max_reference_offset=settings.max_reference_offset_rad,
    )
    records = ReplayJournal(settings.replay_path, network).read_all()
    if not records:
        raise ValueError("Force replay contains no transitions")
    learner_config = LearnerConfig(
        network=network,
        gamma=settings.gamma,
        tau=settings.tau,
        actor_lr=settings.actor_lr,
        critic_lr=settings.critic_lr,
        beta_bc=settings.bc_weight,
        beta_smooth=settings.smooth_weight,
        zero_grip=not config.rlt.force.enabled,
    )
    device = config.rlt.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    learner = (
        ChunkLearner.load(resume_path, device=device) if resume_path else ChunkLearner(learner_config, device=device)
    )
    if learner.config != learner_config:
        raise ValueError("Force checkpoint settings differ from YAML (including force.enabled and replay layout)")
    for step in range(config.num_train_steps):
        indices = rng.integers(0, len(records), size=settings.batch_size)
        metrics = learner.update([records[index] for index in indices])
        if step == 0 or (step + 1) % config.log_interval == 0:
            print(json.dumps({"step": learner.steps, "force_enabled": config.rlt.force.enabled, **metrics}))
        if (step + 1) % config.save_interval == 0:
            learner.save(output)
    learner.save(output)
    from openpi.training.config import save_config

    save_config(config, output.parent)


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
    records = journal.read_all()
    if not records:
        parser.error("Replay contains no transitions")
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

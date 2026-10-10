"""Offline ordinary RLT actor/critic training from an episode Pickle replay directory."""

import argparse
import json
from pathlib import Path

import torch

from openpi.rlt.action_representation import ActionAdapter
from openpi.rlt.actor import ActorConfig
from openpi.rlt.critic import CriticConfig
from openpi.rlt.learner import ChunkLearner
from openpi.rlt.learner import LearnerConfig
from openpi.rlt.replay import ReplayBuffer
from openpi.rlt.replay import ReplayJournal


def train_from_config(config):
    """Train the small networks using the same force schema as inference."""
    settings = config.rlt
    journal = ReplayJournal(settings.replay.directory)
    if journal.spec.grip_dim != settings.force.grip_dim:
        raise ValueError("Training force config differs from replay")
    metadata = config.policy_metadata or {}
    if (
        journal.spec.horizon != config.model.action_horizon
        or journal.spec.action_dim != metadata.get("action_dim")
        or journal.spec.action_space != metadata.get("action_space")
    ):
        raise ValueError("Training action layout differs from replay")
    replay = ReplayBuffer(journal.spec, capacity=settings.replay.capacity, seed=config.seed)
    replay.extend(journal.read_all())
    if not len(replay):
        raise ValueError("Replay contains no annotated transitions")
    output = Path(settings.learner.checkpoint_path)
    resume_path = settings.actor_checkpoint or (str(output) if config.resume else None)
    if output.exists() and not config.overwrite and not resume_path:
        raise FileExistsError(f"Actor output exists: {output}; set overwrite or rlt.actor_checkpoint")
    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.seed)
    learner = (
        ChunkLearner.load(resume_path, device=device)
        if resume_path
        else ChunkLearner(
            journal.spec,
            settings.learner.training_config(),
            actor_config=settings.actor,
            critic_config=settings.critic,
            adapter=ActionAdapter.from_stats(
                settings.action_norm_stats,
                action_dim=journal.spec.action_dim,
                representation=settings.action_representation,
            ),
            device=device,
        )
    )
    if learner.spec != journal.spec:
        raise ValueError("Actor checkpoint specification differs from replay")
    model_id = json.loads((journal.path / "manifest.json").read_text()).get("provenance", {}).get("model_id")
    prior_id = learner.runtime_state.get("model_id")
    if prior_id is not None and model_id is not None and prior_id != model_id:
        raise ValueError("Actor checkpoint frozen model identity differs from replay")
    if model_id is not None:
        learner.runtime_state["model_id"] = model_id
    if learner.config.optimizer != settings.learner.optimizer:
        raise ValueError("Resume requires the checkpoint optimizer kind")
    learner.config = settings.learner.training_config()
    for optimizer, lr in (
        (learner.actor_opt, settings.learner.actor_lr),
        (learner.critic_opt, settings.learner.critic_lr),
    ):
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = settings.learner.weight_decay
    for step in range(config.num_train_steps):
        metrics = learner.update(replay.sample(settings.learner.batch_size))
        if step == 0 or (step + 1) % config.log_interval == 0:
            print(json.dumps({"step": learner.steps, **metrics}))
        if (step + 1) % config.save_interval == 0:
            learner.save(output)
    learner.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--action-norm-stats", type=Path)
    parser.add_argument("--representation", choices=("absolute", "ee_delta"), default="absolute")
    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("steps and batch-size must be positive")
    torch.manual_seed(args.seed)
    journal = ReplayJournal(args.replay)
    replay = ReplayBuffer(journal.spec, capacity=args.capacity, seed=args.seed)
    replay.extend(journal.read_all())
    if not len(replay):
        parser.error("Replay contains no annotated transitions")
    learner = (
        ChunkLearner.load(args.resume, device=args.device)
        if args.resume
        else ChunkLearner(
            journal.spec,
            LearnerConfig(bc_weight=args.bc_weight, q_weight=args.q_weight),
            actor_config=ActorConfig(hidden_dim=args.actor_hidden_dim),
            critic_config=CriticConfig(hidden_dim=args.critic_hidden_dim),
            adapter=ActionAdapter.from_stats(
                args.action_norm_stats, action_dim=journal.spec.action_dim, representation=args.representation
            ),
            device=args.device,
        )
    )
    if learner.spec != journal.spec:
        raise ValueError("Checkpoint specification differs from replay")
    for step in range(args.steps):
        metrics = learner.update(replay.sample(args.batch_size))
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            print(json.dumps({"step": learner.steps, **metrics}))
    learner.save(args.output)


if __name__ == "__main__":
    main()

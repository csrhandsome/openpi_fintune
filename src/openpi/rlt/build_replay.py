"""Build ordinary RLT replay from saved data_collect episodes and frontend labels."""

import argparse
import json
from pathlib import Path

from openpi.rlt.dataset import import_dataset
from openpi.rlt.episode import ReplaySpec
from openpi.rlt.features import FeatureCache
from openpi.rlt.replay import ReplayJournal
from openpi.rlt.rewards import TerminalReward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", default="config/train/openpi_rlt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--readout", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--episode-index", type=int, action="append")
    parser.add_argument("--action-source", choices=("command", "measured"), default="command")
    parser.add_argument(
        "--include-noncritical",
        action="store_true",
        help="Import full episodes, including noncritical stages and legacy unmarked data. By default, only "
        "intervals starting at/after the recorded critical-phase marker are imported; unmarked episodes are skipped. "
        "Use a separate output directory for each mode.",
    )
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--failure-reward", type=float, default=0.0)
    parser.add_argument("--step-reward", type=float, default=0.0)
    args = parser.parse_args()
    # Import model and dataset dependencies only in the executable entry point.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from openpi.serving.runner import Args
    from openpi.serving.runner import create_policy
    from openpi.training.config import get_config
    from openpi.training.config import local_path

    policy = create_policy(
        Args(
            config=get_config(args.config).config_path,
            checkpoint=args.checkpoint,
            readout=args.readout,
            device=args.device,
        )
    )
    metadata = policy.metadata
    if not (metadata.get("rlt_readout_protocol") or metadata.get("rlt_protocol")):
        raise ValueError("Replay feature extraction requires a frozen VLA with an attached readout")
    from openpi.rlt.features import frozen_model_identity

    model_id = frozen_model_identity(
        get_config(args.config).config_path, metadata, local_path(args.checkpoint), local_path(args.readout)
    )
    spec = ReplaySpec(
        int(metadata["z_rlt_dim"]),
        int(metadata["action_dim"]),
        int(metadata["action_horizon"]),
        args.chunk_len,
        int(metadata["action_dim"]),
        metadata["action_space"],
        float(metadata["action_fps"]),
    )
    root = args.dataset.resolve()
    dataset = LeRobotDataset(repo_id=f"local/{root.name}", root=root)
    journal = ReplayJournal(args.output, spec)
    cache = FeatureCache(
        args.feature_cache or args.output / "features",
        policy,
        model_id=model_id,
        horizon=spec.horizon,
        action_dim=spec.action_dim,
    )
    stats = import_dataset(
        dataset,
        root,
        cache,
        journal,
        reward=TerminalReward(args.success_reward, args.failure_reward, args.step_reward),
        stride=args.stride,
        action_source=args.action_source,
        episode_indices=args.episode_index,
        critical_phase_only=not args.include_noncritical,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

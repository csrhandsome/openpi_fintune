"""Train from a PyTorch YAML: python scripts/train.py --config config/train/openpi.yaml."""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.training import config
from openpi.training import runner


def _uses_data_parallel(settings: config.TrainConfig) -> bool:
    return settings.mode == "vla" or (settings.mode == "rlt" and settings.rlt.stage == "sft")


def _maybe_relaunch(settings: config.TrainConfig) -> None:
    """Start train.num_gpus processes when this is the parent launch."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not _uses_data_parallel(settings):
        if settings.num_gpus != 1:
            raise ValueError("actor_critic and force use one process; set train.num_gpus: 1")
        return
    if world == settings.num_gpus:
        return
    if world == 1 and "LOCAL_RANK" not in os.environ and settings.num_gpus > 1:
        os.execvp(
            sys.executable,
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={settings.num_gpus}",
                *sys.argv,
            ],
        )
    raise ValueError(
        f"train.num_gpus is {settings.num_gpus}, but WORLD_SIZE is {world}. "
        "Run scripts/train.py so it starts the configured number of processes."
    )


def main(settings: config.TrainConfig | None = None):
    settings = config.cli() if settings is None else settings
    _maybe_relaunch(settings)
    runner.train(settings)


if __name__ == "__main__":
    main()

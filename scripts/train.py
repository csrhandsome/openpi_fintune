"""Train from a PyTorch YAML: python scripts/train.py --config config/train/openpi.yaml."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.training import config
from openpi.training import runner


def main(settings: config.TrainConfig | None = None):
    runner.train(settings if settings is not None else config.cli())


if __name__ == "__main__":
    main()

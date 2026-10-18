"""Select the experiment tracker shared by PyTorch training backends."""

import importlib
import logging
import os
from pathlib import Path

from openpi.training import config as train_config

LOG_ROOT = train_config.REPO_ROOT / "data"


def init_tracking(config: train_config.TrainConfig, directory: Path, *, resuming: bool):
    """Initialize the selected SDK on the main process and retain its run ID."""
    backend = config.tracking_backend
    if backend == "none":
        return None

    logdir = LOG_ROOT / backend / config.name / config.exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    if backend == "swanlab":
        os.environ.setdefault("SWANLAB_LOG_DIR", str(LOG_ROOT / "swanlog"))
    else:
        os.environ.setdefault("WANDB_CACHE_DIR", str(LOG_ROOT / "wandb" / "cache"))
        os.environ.setdefault("WANDB_DATA_DIR", str(LOG_ROOT / "wandb" / "staging"))
    sdk = importlib.import_module(backend)
    directory.mkdir(parents=True, exist_ok=True)
    id_path = directory / f"{backend}_id.txt"
    options = {"project": config.project_name, "config": train_config.to_dict(config)}
    if resuming and id_path.is_file():
        options.update(id=id_path.read_text().strip(), resume="must")
    elif resuming:
        logging.info("No %s run ID in checkpoint directory; starting a new experiment", backend)
    if backend == "swanlab":
        run = sdk.init(experiment_name=config.exp_name, logdir=str(logdir), **options)
    else:
        run = sdk.init(name=config.exp_name, dir=str(logdir), **options)
    # SwanLab offline/disabled modes do not have a cloud experiment ID.
    if run.id is not None:
        id_path.write_text(run.id)
    return sdk

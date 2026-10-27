"""Select the experiment tracker shared by PyTorch training backends."""

from collections.abc import Mapping
import importlib
import logging
import os
from pathlib import Path

from openpi.training import config as train_config

LOG_ROOT = train_config.REPO_ROOT / "data"


def _tracking_view(config: train_config.TrainConfig | Mapping):
    """Return backend, run names, and the config snapshot stored by the tracker."""
    if isinstance(config, Mapping):
        backend = config.get("tracking_backend", "none")
        snapshot = dict(config)
        name = config.get("name")
        exp_name = config.get("exp_name")
        project_name = config.get("project_name")
    else:
        backend = config.tracking_backend
        snapshot = train_config.to_dict(config) if backend != "none" else None
        name = config.name
        exp_name = config.exp_name
        project_name = config.project_name
    return backend, name, exp_name, project_name, snapshot


def init_tracking(config: train_config.TrainConfig | Mapping, directory: Path, *, resuming: bool):
    """Initialize the selected SDK on the main process and retain its run ID.

    ``config`` is either a VLA ``TrainConfig`` or a mapping with ``tracking_backend``,
    ``name``, ``exp_name``, and ``project_name``. A missing backend selects no tracker.
    """
    backend, name, exp_name, project_name, snapshot = _tracking_view(config)
    if backend == "none":
        return None
    if backend not in ("wandb", "swanlab"):
        raise ValueError("tracking_backend must be none, wandb or swanlab")

    logdir = LOG_ROOT / backend / name / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    if backend == "swanlab":
        os.environ.setdefault("SWANLAB_LOG_DIR", str(LOG_ROOT / "swanlog"))
    else:
        os.environ.setdefault("WANDB_CACHE_DIR", str(LOG_ROOT / "wandb" / "cache"))
        os.environ.setdefault("WANDB_DATA_DIR", str(LOG_ROOT / "wandb" / "staging"))
    sdk = importlib.import_module(backend)
    directory.mkdir(parents=True, exist_ok=True)
    id_path = directory / f"{backend}_id.txt"
    options = {"project": project_name, "config": snapshot}
    if resuming and id_path.is_file():
        options.update(id=id_path.read_text().strip(), resume="must")
    elif resuming:
        logging.info("No %s run ID in checkpoint directory; starting a new experiment", backend)
    if backend == "swanlab":
        run = sdk.init(experiment_name=exp_name, logdir=str(logdir), **options)
    else:
        run = sdk.init(name=exp_name, dir=str(logdir), **options)
    # SDK modes that omit a run ID cannot be resumed by ID.
    if run.id is not None:
        id_path.write_text(run.id)
    return sdk

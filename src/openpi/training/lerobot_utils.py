"""Resolve local collector directories without changing their Hub repo IDs."""

import os
from pathlib import Path

DEFAULT_DATASET_HOME = Path(__file__).resolve().parents[4] / "data_collect" / "data" / "dataset"


def resolve_dataset_root(repo_id: str) -> Path:
    if root := os.environ.get("OPENPI_DATASET_ROOT"):
        return Path(root).expanduser()
    home = Path(os.environ.get("HF_LEROBOT_HOME", str(DEFAULT_DATASET_HOME))).expanduser()
    nested = home / repo_id
    flat = home / repo_id.rsplit("/", 1)[-1]
    if (flat / "meta/info.json").is_file():
        return flat
    if (nested / "meta/info.json").is_file():
        return nested
    return flat

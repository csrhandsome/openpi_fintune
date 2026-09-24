"""Shared helpers for ModelScope download/upload scripts."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "config.py"


def load_access_token(cli_token: str | None = None, *, required: bool = False) -> str | None:
    """Resolve token: --token > env > scripts/modelscope/config.py."""
    token = (cli_token or "").strip() or None
    if not token:
        token = (
            os.environ.get("MODELSCOPE_API_TOKEN")
            or os.environ.get("MODELSCOPE_TOKEN")
            or ""
        ).strip() or None
    if not token and _CONFIG_PATH.is_file():
        spec = importlib.util.spec_from_file_location("modelscope_local_config", _CONFIG_PATH)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            token = (getattr(mod, "ACCESS_TOKEN", None) or "").strip() or None

    if required and not token:
        raise SystemExit(
            "Missing ModelScope token. Set ACCESS_TOKEN in scripts/modelscope/config.py "
            "(see config.example.py), pass --token, or export MODELSCOPE_API_TOKEN "
            "(https://modelscope.cn/my/myaccesstoken)."
        )
    return token

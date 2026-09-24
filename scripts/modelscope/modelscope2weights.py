#!/usr/bin/env python3
"""Download a model (or dataset) from ModelScope Hub to a local directory.

Examples:
  # Download threecheng/livevla checkpoints into the repo-local checkpoints/ tree
  # (remote path checkpoints/... -> ./checkpoints/...).
  uv run python scripts/modelscope/modelscope2weights.py threecheng/livevla \\
      --local-dir . --include 'checkpoints/**'

  # Download a whole model into a dedicated folder.
  uv run python scripts/modelscope/modelscope2weights.py Qwen/Qwen2.5-0.5B \\
      --local-dir ./models/Qwen2.5-0.5B

  # Filter by glob patterns.
  uv run python scripts/modelscope/modelscope2weights.py owner/name \\
      --local-dir ./out --include '*.safetensors' --exclude 'onnx/*'

  # Dataset download.
  uv run python scripts/modelscope/modelscope2weights.py owner/dataset_name \\
      --repo-type dataset --local-dir ./data/dataset_name

Token priority: --token > MODELSCOPE_API_TOKEN / MODELSCOPE_TOKEN >
scripts/modelscope/config.py (ACCESS_TOKEN). See config.example.py.
Get a token: https://modelscope.cn/my/myaccesstoken
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import load_access_token


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a ModelScope repo to a local directory (modelscope2weights).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "repo_id",
        help="ModelScope repo id, e.g. Qwen/Qwen2.5-0.5B or namespace/model_name",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local directory to place files (flat layout, ignores cache_dir).",
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset"),
        default="model",
        help="Repository type (default: model).",
    )
    parser.add_argument("--revision", default=None, help="Branch / tag / commit revision.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional ModelScope cache dir (ignored when --local-dir is set).",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        metavar="GLOB",
        help="Only download files matching these globs.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        metavar="GLOB",
        help="Skip files matching these globs.",
    )
    parser.add_argument("--token", default=None, help="SDK access token (optional for public repos).")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel download workers.",
    )
    args = parser.parse_args()

    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi

    token = load_access_token(args.token)
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    if token:
        HubApi().login(token)

    print(f"Downloading {args.repo_type} '{args.repo_id}' -> {local_dir}", flush=True)
    kwargs: dict = {
        "local_dir": str(local_dir),
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "allow_patterns": args.include,
        "ignore_patterns": args.exclude,
        "max_workers": args.max_workers,
        "token": token,
    }
    if args.repo_type == "model":
        kwargs["model_id"] = args.repo_id
    else:
        kwargs["repo_id"] = args.repo_id
        kwargs["repo_type"] = args.repo_type

    path = snapshot_download(**{k: v for k, v in kwargs.items() if v is not None})
    print(f"Done. Saved to: {path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)

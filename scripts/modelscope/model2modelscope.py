#!/usr/bin/env python3
"""Upload a local model directory to ModelScope Hub.

Examples:
  export MODELSCOPE_API_TOKEN=ms-xxxxxxxx
  uv run python scripts/modelscope/model2modelscope.py your-name/my-model ./checkpoints/final

  uv run python scripts/modelscope/model2modelscope.py your-name/my-model ./out \\
      --commit-message 'upload v1' --visibility public --license 'Apache License 2.0'

Notes:
  - Token priority: --token > MODELSCOPE_API_TOKEN / MODELSCOPE_TOKEN >
    scripts/modelscope/config.py (ACCESS_TOKEN). See config.example.py.
  - Get a token: https://modelscope.cn/my/myaccesstoken
  - Prefer a configuration.json (or .yaml/.yml) in model_dir; missing it only warns.
  - Repo is created automatically if it does not exist (via push_model).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import load_access_token


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload a local model directory to ModelScope (model2modelscope).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "repo_id",
        help="Target ModelScope model id, e.g. your-name/my-model",
    )
    parser.add_argument(
        "model_dir",
        help="Local directory to upload (files under this path).",
    )
    parser.add_argument("--token", default=None, help="SDK access token.")
    parser.add_argument(
        "--revision",
        default=None,
        help="Target branch / revision (default: master).",
    )
    parser.add_argument(
        "--path-in-repo",
        default="",
        help="Destination prefix inside the remote repo (default: repo root).",
    )
    parser.add_argument("--commit-message", default=None, help="Commit message.")
    parser.add_argument("--commit-description", default=None, help="Commit description.")
    parser.add_argument(
        "--visibility",
        default=None,
        choices=("public", "private", "internal", "1", "3", "5"),
        help="Repo visibility when creating: public/private/internal (or 5/1/3).",
    )
    parser.add_argument(
        "--license",
        default=None,
        help="License when creating repo, e.g. 'Apache License 2.0' or 'MIT'.",
    )
    parser.add_argument("--chinese-name", default=None, help="Chinese display name for new repos.")
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        metavar="GLOB",
        help="Only upload files matching these globs.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        metavar="GLOB",
        help="Skip files matching these globs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel upload workers (default: 4).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable resumable upload cache (.ms_upload_cache).",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model_dir is not a directory: {model_dir}")

    token = load_access_token(args.token, required=True)
    assert token is not None

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)

    visibility = args.visibility
    if visibility in {"public", "private", "internal"}:
        visibility = {"private": "1", "internal": "3", "public": "5"}[visibility]

    print(f"Uploading {model_dir} -> ModelScope '{args.repo_id}'", flush=True)
    api.push_model(
        model_id=args.repo_id,
        model_dir=str(model_dir),
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
        commit_description=args.commit_description,
        visibility=visibility,
        license=args.license,
        chinese_name=args.chinese_name,
        allow_patterns=args.include,
        ignore_patterns=args.exclude,
        max_workers=args.max_workers,
        use_cache=not args.no_cache,
    )
    print(f"Done. https://modelscope.cn/models/{args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)

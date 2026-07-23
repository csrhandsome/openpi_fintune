#!/usr/bin/env python3
"""Download a Hub dataset subdirectory without listing the whole repo.

`hf download --include ...` falls back to full-repo `list_repo_tree` when the
Hub reports an unreliable file count. For large datasets (OmniAction) that
means hundreds of thousands of paginated API calls -> HTTP 429 on hf-mirror.

This script lists only `path_in_repo` then downloads files (skips existing).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _bootstrap_mirror() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # Import wrapper patches from sibling script.
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import importlib.util

    spec = importlib.util.spec_from_file_location("hf_with_mirror", root / "hf_with_mirror.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod._apply_mirror_patches()


def _list_files(repo_id: str, repo_type: str, path_in_repo: str) -> list[str]:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    paths: list[str] = []
    for item in api.list_repo_tree(
        repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        recursive=True,
    ):
        if isinstance(item, RepoFile):
            paths.append(item.path)
        elif getattr(item, "type", None) == "file":
            paths.append(item.path)
    return paths


def _download_one(repo_id: str, repo_type: str, filename: str, local_dir: Path) -> str:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    dest = local_dir / filename
    if dest.is_file() and dest.stat().st_size > 0:
        return "skip"

    delays = [2, 5, 10, 20, 40, 60]
    last_err: Exception | None = None
    for attempt, delay in enumerate([0, *delays], start=1):
        if delay:
            time.sleep(delay)
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                local_dir=str(local_dir),
            )
            return "ok"
        except HfHubHTTPError as e:
            last_err = e
            # 429 / 5xx: retry; 404: give up
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                raise
            if status not in (429, 500, 502, 503, 504) and attempt > 3:
                raise
        except Exception as e:  # noqa: BLE001 — network flakes
            last_err = e
    assert last_err is not None
    raise last_err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--path", required=True, help="Subdirectory in repo, e.g. speech/overlap")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("HF_MAX_WORKERS", "4")))
    parser.add_argument("--list-sleep", type=float, default=0.5, help="Pause between tree pages is handled by hub; extra settle after list")
    args = parser.parse_args()

    _bootstrap_mirror()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    path = args.path.strip("/")

    print(f"Listing {args.repo_id}:{path} ...", flush=True)
    files = _list_files(args.repo_id, args.repo_type, path)
    print(f"Found {len(files)} files under {path}", flush=True)
    if args.list_sleep:
        time.sleep(args.list_sleep)

    existing = 0
    todo: list[str] = []
    for f in files:
        dest = local_dir / f
        if dest.is_file() and dest.stat().st_size > 0:
            existing += 1
        else:
            todo.append(f)
    print(f"Already local: {existing}; to download: {len(todo)}; workers={args.max_workers}", flush=True)

    ok = skip = fail = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = {
            ex.submit(_download_one, args.repo_id, args.repo_type, f, local_dir): f for f in todo
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            name = futs[fut]
            try:
                status = fut.result()
                if status == "skip":
                    skip += 1
                else:
                    ok += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                errors.append(f"{name}: {e}")
            if done % 50 == 0 or done == len(todo):
                print(f"  progress {done}/{len(todo)} ok={ok} fail={fail}", flush=True)

    print(f"Done {path}: downloaded={ok} skipped={skip + existing} failed={fail}", flush=True)
    if errors:
        print("First errors:", flush=True)
        for line in errors[:10]:
            print(" ", line, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

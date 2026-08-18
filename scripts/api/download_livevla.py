"""Download the LiveVLA checkpoint from ModelScope into ``checkpoints/``.

Usage:
    uv run python scripts/api/download_livevla.py
    uv run python scripts/api/download_livevla.py --dry-run

The ModelScope SDK resumes an interrupted local snapshot automatically. Files are
placed directly in ``checkpoints/threecheng_livevla`` so OpenPI can use the
directory without a subsequent move or copy.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import shutil

MODEL_ID = "threecheng/livevla"
MIN_FREE_GIB = 95


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def gibibytes(num_bytes: int) -> float:
    return num_bytes / 1024**3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download threecheng/livevla with the ModelScope Python SDK."
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=project_root() / "checkpoints" / "threecheng_livevla",
        help="Destination for the ModelScope snapshot (default: checkpoints/threecheng_livevla).",
    )
    parser.add_argument(
        "--revision",
        help="Optional ModelScope revision (the repository default is used when omitted).",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=MIN_FREE_GIB,
        help=f"Require this much free disk space before downloading (default: {MIN_FREE_GIB}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Optional number of concurrent ModelScope download workers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show the preflight result; do not contact ModelScope or write files.",
    )
    return parser.parse_args(argv)


def nearest_existing_directory(path: Path) -> Path:
    """Return an existing directory on the destination filesystem."""

    candidate = path.resolve()
    while not candidate.exists():
        candidate = candidate.parent
    if not candidate.is_dir():
        raise NotADirectoryError(f"Destination ancestor is not a directory: {candidate}")
    return candidate


def check_free_space(local_dir: Path, min_free_gib: float) -> None:
    if min_free_gib < 0:
        raise ValueError("--min-free-gib must be non-negative.")

    disk_path = nearest_existing_directory(local_dir)
    free_bytes = shutil.disk_usage(disk_path).free
    required_bytes = int(min_free_gib * 1024**3)
    print(
        f"Destination: {local_dir}\n"
        f"Available space: {gibibytes(free_bytes):.2f} GiB\n"
        f"Required space: {min_free_gib:.2f} GiB"
    )
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Only {gibibytes(free_bytes):.2f} GiB is available at {disk_path}; "
            f"need at least {min_free_gib:.2f} GiB. Download was not started."
        )


def download(local_dir: Path, revision: str | None, max_workers: int | None) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as error:
        raise RuntimeError("ModelScope is unavailable. Install project dependencies with `uv sync`.") from error

    local_dir = local_dir.resolve()
    kwargs: dict[str, str | int] = {"model_id": MODEL_ID, "local_dir": str(local_dir)}
    if revision:
        kwargs["revision"] = revision
    if max_workers is not None:
        if max_workers < 1:
            raise ValueError("--max-workers must be positive.")
        kwargs["max_workers"] = max_workers

    print(f"Downloading ModelScope snapshot {MODEL_ID!r}...")
    downloaded_dir = Path(snapshot_download(**kwargs)).resolve()
    if downloaded_dir != local_dir:
        raise RuntimeError(
            f"ModelScope returned {downloaded_dir}, expected {local_dir}. "
            "Refusing to leave weights outside the requested checkpoints directory."
        )
    print(f"LiveVLA checkpoint is ready at: {downloaded_dir}")
    return downloaded_dir


def main(argv: Sequence[str] | None = None) -> Path | None:
    args = parse_args(argv)
    local_dir = args.local_dir.expanduser()
    check_free_space(local_dir, args.min_free_gib)
    if args.dry_run:
        print("Dry run succeeded; ModelScope was not contacted.")
        return None
    return download(local_dir, args.revision, args.max_workers)


if __name__ == "__main__":
    main()

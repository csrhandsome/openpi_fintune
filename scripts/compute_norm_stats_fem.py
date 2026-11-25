#!/usr/bin/env python3
"""Compute normalization statistics (mean, std, min, max) for FEM labels.

Reads ``observation.fem_label`` from one or more LeRobot v3 datasets (either
directly via ``--roots`` or from a training YAML config's ``data.sources``
entries that set ``has_fem: true``) and writes a JSON stats file for
``DualCameraForceJEPA`` training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import tqdm
import yaml

FEM_LABEL_NAMES = [
    "force_x_N",
    "force_y_N",
    "force_z_N",
    "force_norm_N",
    "torque_x_Nm",
    "torque_y_Nm",
    "torque_z_Nm",
    "elastic_energy_J",
    "max_displacement_m",
    "max_von_mises_pa",
    "min_jacobian",
]


class _RunningVectorStats:
    """Numerically stable Welford / Chan parallel accumulator for 11-D vectors."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None
        self.min_val: np.ndarray | None = None
        self.max_val: np.ndarray | None = None

    def update(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return
        flat = batch.reshape(-1, batch.shape[-1]).astype(np.float64)
        finite_mask = np.isfinite(flat).all(axis=1)
        flat = flat[finite_mask]
        if flat.shape[0] == 0:
            return

        n_b = flat.shape[0]
        mean_b = flat.mean(axis=0)
        m2_b = np.sum((flat - mean_b) ** 2, axis=0)
        min_b = flat.min(axis=0)
        max_b = flat.max(axis=0)

        if self.count == 0:
            self.count = n_b
            self.mean = mean_b
            self.m2 = m2_b
            self.min_val = min_b
            self.max_val = max_b
            return

        n_a = self.count
        n_total = n_a + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n_total)
        self.m2 = self.m2 + m2_b + (delta**2) * (n_a * n_b / n_total)
        self.count = n_total
        self.min_val = np.minimum(self.min_val, min_b)
        self.max_val = np.maximum(self.max_val, max_b)

    def finalize(self, eps: float = 1e-6) -> dict[str, Any]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("No finite FEM labels found to compute statistics")
        var = self.m2 / self.count
        std = np.maximum(np.sqrt(var), eps)
        return {
            "count": int(self.count),
            "names": FEM_LABEL_NAMES,
            "mean": self.mean.astype(np.float64).tolist(),
            "std": std.astype(np.float64).tolist(),
            "min": self.min_val.astype(np.float64).tolist(),
            "max": self.max_val.astype(np.float64).tolist(),
        }


def _discover_dataset_roots(roots: list[Path] | None, config_path: Path | None) -> list[Path]:
    discovered: list[Path] = []
    if roots:
        for root in roots:
            resolved = root.expanduser().resolve()
            if (resolved / "data").is_dir():
                discovered.append(resolved)
            elif resolved.is_dir():
                for child in sorted(resolved.iterdir()):
                    if child.is_dir() and (child / "data").is_dir():
                        discovered.append(child)
    if config_path is not None:
        cfg = yaml.safe_load(config_path.read_text())
        project_root = config_path.resolve().parents[2] if len(config_path.resolve().parents) >= 3 else Path.cwd()
        for source in cfg.get("data", {}).get("sources", []):
            if not source.get("has_fem", False):
                continue
            src_root = Path(source["root"]).expanduser()
            if not src_root.is_absolute():
                src_root = project_root / src_root
            if (src_root / "data").is_dir():
                discovered.append(src_root.resolve())
    # Preserve order while deduplicating.
    return list(dict.fromkeys(discovered))


def compute_fem_stats(
    dataset_roots: list[Path],
    *,
    feature_name: str = "observation.fem_label",
    max_files_per_dataset: int | None = None,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Compute global normalization statistics across all given LeRobot v3 roots."""
    if not dataset_roots:
        raise ValueError("No valid LeRobot v3 dataset roots found")
    stats = _RunningVectorStats()
    parquet_files: list[Path] = []
    for root in dataset_roots:
        files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
        if max_files_per_dataset is not None:
            files = files[:max_files_per_dataset]
        parquet_files.extend(files)
    if not parquet_files:
        raise ValueError("No parquet files found under the specified dataset roots")

    for pq_path in tqdm.tqdm(parquet_files, desc="Scanning FEM parquet files"):
        table = pq.read_table(pq_path, columns=[feature_name])
        values = np.asarray(table[feature_name].to_pylist(), dtype=np.float64)
        stats.update(values)

    result = stats.finalize(eps=eps)
    result["feature_name"] = feature_name
    result["num_datasets"] = len(dataset_roots)
    result["num_parquet_files"] = len(parquet_files)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=None,
        help="LeRobot v3 dataset roots or parent directories containing task datasets",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional training YAML; scans sources with has_fem: true",
    )
    parser.add_argument("--feature-name", default="observation.fem_label")
    parser.add_argument("--output", type=Path, default=Path("config/train/fem_norm_stats.json"))
    parser.add_argument("--max-files-per-dataset", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args(argv)

    if args.roots is None and args.config is None:
        parser.error("Provide at least one of --roots or --config")

    roots = _discover_dataset_roots(args.roots, args.config)
    result = compute_fem_stats(
        roots,
        feature_name=args.feature_name,
        max_files_per_dataset=args.max_files_per_dataset,
        eps=args.eps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved FEM norm stats ({result['count']} finger-frames) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validated, atomic evaluation artifacts, with no dependency on tactile_ssl."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import quote

import numpy as np

SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_group_id(source: str, episode: int | str) -> str:
    return f"{quote(str(source), safe='')}/{quote(str(episode), safe='')}"


def make_sample_id(source: str, episode: int | str, camera: str, frame: int) -> str:
    return f"{make_group_id(source, episode)}/{quote(camera, safe='')}/{int(frame)}"


def split_fingerprint(sample_id: np.ndarray, group_id: np.ndarray) -> str:
    pairs = sorted(zip(np.asarray(sample_id).tolist(), np.asarray(group_id).tolist(), strict=True))
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class EvaluationArtifact:
    sample_id: np.ndarray
    group_id: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    manifest: dict[str, Any]
    path: Path


def _strings(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "US" or not len(value):
        raise ValueError(f"{name} must be a nonempty one-dimensional string array")
    value = value.astype(str)
    if np.any(value == ""):
        raise ValueError(f"{name} contains empty IDs")
    return value


def validate_artifact(artifact: EvaluationArtifact) -> None:
    sample_id = _strings(artifact.sample_id, "sample_id")
    group_id = _strings(artifact.group_id, "group_id")
    y_true, y_pred, manifest = artifact.y_true, artifact.y_pred, artifact.manifest
    if len(np.unique(sample_id)) != len(sample_id):
        raise ValueError("sample_id must be unique; repeated frames cannot be independent observations")
    if not (len(group_id) == len(y_true) == len(y_pred) == len(sample_id)):
        raise ValueError("All prediction arrays must have the same sample count")
    if y_true.dtype.kind not in "iuf" or y_pred.dtype.kind not in "iuf":
        raise ValueError("Labels and predictions must be numeric")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Labels and predictions must be finite")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported artifact schema_version")
    for name in ("task", "dataset_fingerprint", "split_fingerprint"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"Missing nonempty {name}")
    if manifest.get("split") != "test":
        raise ValueError("Only held-out test artifacts may be reported")
    labels = manifest.get("label_names")
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x for x in labels):
        raise ValueError("label_names must be a nonempty list of strings")
    if len(set(labels)) != len(labels):
        raise ValueError("label_names must be unique")
    task_type = manifest.get("task_type")
    if task_type == "regression":
        if y_true.ndim != 2 or y_pred.shape != y_true.shape or y_true.shape[1] != len(labels):
            raise ValueError("Regression arrays must have identical [N, D] shapes matching label_names")
    elif task_type == "classification":
        if y_true.ndim != 1 or y_true.dtype.kind not in "iu":
            raise ValueError("Classification y_true must contain integer class IDs [N]")
        if np.any(y_true < 0) or np.any(y_true >= len(labels)):
            raise ValueError("Class IDs are outside label_names")
        if y_pred.ndim == 1:
            if y_pred.dtype.kind not in "iu" or np.any(y_pred < 0) or np.any(y_pred >= len(labels)):
                raise ValueError("Classification predictions must be integer class IDs or [N, C] scores")
        elif y_pred.ndim != 2 or y_pred.shape[1] != len(labels):
            raise ValueError("Classification score shape must be [N, C] matching label_names")
    else:
        raise ValueError("task_type must be regression or classification")
    if split_fingerprint(sample_id, group_id) != manifest["split_fingerprint"]:
        raise ValueError("split_fingerprint does not match evaluated sample IDs and episode groups")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or not protocol.get("name"):
        raise ValueError("protocol must describe the held-out split with a nonempty name")
    partitions = {name: set(protocol.get(f"{name}_group_ids", [])) for name in ("train", "validation", "test")}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if partitions[left] & partitions[right]:
            raise ValueError(f"{left}/{right} episode overlap in split protocol")
    if not partitions["test"] or set(group_id) != partitions["test"]:
        raise ValueError("protocol.test_group_ids must exactly match evaluated episode groups")
    if not partitions["train"]:
        raise ValueError("protocol.train_group_ids is required to check train/test leakage")


def _evaluation_dir(path: Path) -> Path:
    path = Path(path)
    return path if (path / "manifest.json").exists() or path.name == "evaluation" else path / "evaluation"


def save_evaluation_artifact(
    run_root: Path,
    *,
    task: str,
    sample_id: Any,
    group_id: Any,
    y_true: Any,
    y_pred: Any,
    label_names: list[str],
    dataset_fingerprint: str,
    protocol: Mapping[str, Any],
    task_type: str | None = None,
    seed: int | None = None,
    checkpoint: str | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``run_root/evaluation/{predictions.npz,manifest.json}``.

    The protocol contains name, train_group_ids, validation_group_ids, and
    test_group_ids. Groups identify source/episode, shared across all cameras.
    Each replacement is atomic; the manifest checksum detects interrupted pairs.
    """
    sample_id = _strings(sample_id, "sample_id")
    group_id = _strings(group_id, "group_id")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "task_type": task_type or task,
        "label_names": list(label_names),
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint(sample_id, group_id),
        "split": "test",
        "protocol": dict(protocol),
        "seed": seed,
        "checkpoint": checkpoint,
        "num_examples": len(sample_id),
        "num_groups": len(np.unique(group_id)),
        "prediction_format": "scores"
        if np.asarray(y_pred).ndim == 2 and (task_type or task) == "classification"
        else "values",
    }
    if extra_manifest:
        collisions = set(extra_manifest) & set(manifest)
        if collisions:
            raise ValueError(f"extra_manifest cannot override reserved fields: {sorted(collisions)}")
        manifest.update(extra_manifest)
    destination = _evaluation_dir(Path(run_root))
    artifact = EvaluationArtifact(
        sample_id, group_id, np.asarray(y_true), np.asarray(y_pred), manifest, destination / "predictions.npz"
    )
    validate_artifact(artifact)
    # Fail before writing any file when custom metadata is not valid JSON.
    json.dumps(manifest, allow_nan=False)
    destination.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile("wb", dir=destination, suffix=".npz", delete=False) as handle:
            temporary.append(Path(handle.name))
            np.savez_compressed(
                handle, sample_id=sample_id, group_id=group_id, y_true=artifact.y_true, y_pred=artifact.y_pred
            )
        manifest["predictions_sha256"] = _file_sha256(temporary[0])
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination, delete=False) as handle:
            temporary.append(Path(handle.name))
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary[0], artifact.path)
        os.replace(temporary[1], destination / "manifest.json")
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return artifact.path


def load_evaluation_artifact(run_root: Path) -> EvaluationArtifact:
    destination = _evaluation_dir(Path(run_root))
    path = destination / "predictions.npz"
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if _file_sha256(path) != manifest.get("predictions_sha256"):
        raise ValueError("Prediction checksum mismatch: artifact write was interrupted or file changed")
    with np.load(path, allow_pickle=False) as values:
        artifact = EvaluationArtifact(
            **{key: values[key].copy() for key in ("sample_id", "group_id", "y_true", "y_pred")},
            manifest=manifest,
            path=path,
        )
    validate_artifact(artifact)
    return artifact

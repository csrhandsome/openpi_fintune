"""Persistent frozen-VLA feature cache, keyed by model identity and exact observation."""

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import yaml

from openpi.models_pytorch.readout_checkpoint import resolve_readout_directory
from openpi.rlt.episode import finite_array
from openpi.rlt.replay import atomic_pickle


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_model_identity(config_path, metadata, checkpoint, readout, *, force=None):
    """Identify frozen weights and preprocessing, shared by offline and live replay."""
    readout = resolve_readout_directory(readout)
    if (Path(readout) / "model.safetensors").is_file():
        checkpoint = Path(readout)
    settings = yaml.safe_load(Path(config_path).read_text())
    data = dict(settings.get("data", {}))
    for key in ("assets", "repo_id", "prompt_from_task"):
        data.pop(key, None)
    values = {"model": settings.get("model", {}), "data": data, "files": {}}
    values["metadata"] = {
        key: metadata[key]
        for key in ("action_space", "action_representation", "action_dim", "action_horizon", "action_fps")
        if key in metadata
    }
    for folder, name in (
        (checkpoint, "model.safetensors"),
        (readout, "readout.safetensors"),
        (readout, "manifest.json"),
    ):
        values["files"][name] = file_digest(Path(folder) / name)
    for path in sorted((Path(checkpoint) / "assets").rglob("*.json")):
        values["files"][str(path.relative_to(checkpoint))] = file_digest(path)
    if force is not None and force.enabled:
        values["force"] = {"input": force.input, "dim": force.feature_dim, "roi": force.roi}
        if force.input == "rgb":
            values["files"]["finger_weights"] = file_digest(Path(force.weights))
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def observation_digest(observation: dict, model_id: str) -> str:
    digest = hashlib.sha256(model_id.encode())
    for name, value in sorted(observation.items()):
        digest.update(name.encode())
        if isinstance(value, str):
            digest.update(json.dumps(value).encode())
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


class FeatureCache:
    def __init__(self, path: str | Path, provider, *, model_id: str, horizon: int, action_dim: int):
        if not model_id:
            raise ValueError("Frozen model identity is required")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.model_id = model_id
        self.horizon = horizon
        self.action_dim = action_dim

    def get(self, observation: dict, proprio: np.ndarray) -> dict:
        key = observation_digest(observation, self.model_id)
        path = self.path / f"{key}.pkl"
        if path.exists():
            with path.open("rb") as stream:
                values = pickle.load(stream)
        else:
            response = self.provider.infer(observation)
            z_rlt = np.asarray(response["z_rlt"], dtype=np.float32)
            if z_rlt.ndim != 1 or not z_rlt.size or not np.isfinite(z_rlt).all():
                raise ValueError("Frozen policy must return a finite RLT feature vector")
            ref = finite_array(response["ref_chunk"], (self.horizon, self.action_dim), "ref_chunk")
            values = {"z_rlt": z_rlt, "ref_chunk": ref}
            atomic_pickle(path, values)
        finite_array(values["ref_chunk"], (self.horizon, self.action_dim), "cached ref_chunk")
        return {**values, "proprio": np.asarray(proprio, dtype=np.float32).copy()}

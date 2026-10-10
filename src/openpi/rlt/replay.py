"""Episode-replaceable Pickle storage and bounded CPU replay sampling."""

from dataclasses import asdict
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np

from openpi.rlt.episode import ReplaySpec
from openpi.rlt.episode import RLTTransition


def atomic_pickle(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ReplayJournal:
    """One local .pkl per episode; reannotation replaces stale rewards."""

    def __init__(self, path: str | Path, spec: ReplaySpec | None = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        manifest_path = self.path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["format"] != "openpi-rlt-replay-v1":
                raise ValueError("Unsupported replay format")
            self.spec = ReplaySpec(**manifest["spec"])
            if spec is not None and self.spec != spec:
                raise ValueError("Replay specification changed; use a different directory")
        else:
            if spec is None:
                raise ValueError("A new replay directory requires a ReplaySpec")
            self.spec = spec
            manifest_path.write_text(json.dumps({"format": "openpi-rlt-replay-v1", "spec": asdict(spec)}, indent=2))

    def _episode_path(self, episode_id: str) -> Path:
        digest = hashlib.sha256(episode_id.encode()).hexdigest()
        return self.path / f"episode_{digest}.pkl"

    def bind_provenance(self, provenance: dict):
        path = self.path / "manifest.json"
        manifest = json.loads(path.read_text())
        if "provenance" in manifest:
            if manifest["provenance"] != provenance:
                raise ValueError("Replay model/action semantics changed; use a different output directory")
            return
        manifest["provenance"] = provenance
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def replace_episode(self, records: list[RLTTransition], *, metadata: dict):
        if not records or len({record.episode_id for record in records}) != 1:
            raise ValueError("Write one nonempty complete episode at a time")
        if len({r.step_id for r in records}) != len(records) or not any(r.done for r in records):
            raise ValueError("Episode needs distinct windows and a terminal window")
        if len({r.success for r in records}) != 1:
            raise ValueError("All windows must share the episode annotation")
        for record in records:
            record.validate(self.spec)
        atomic_pickle(
            self._episode_path(records[0].episode_id),
            {
                "format": "openpi-rlt-episode-v1",
                "spec": asdict(self.spec),
                "metadata": metadata,
                "transitions": [
                    {field.name: getattr(record, field.name) for field in fields(record)} for record in records
                ],
            },
        )

    def remove_episode(self, episode_id: str):
        self._episode_path(episode_id).unlink(missing_ok=True)

    def _load_episode(self, path: Path) -> dict:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload["format"] != "openpi-rlt-episode-v1" or ReplaySpec(**payload["spec"]) != self.spec:
            raise ValueError(f"Episode format or specification mismatch: {path}")
        return payload

    def episode_metadata(self, episode_id: str) -> dict | None:
        path = self._episode_path(episode_id)
        return self._load_episode(path)["metadata"] if path.exists() else None

    def read_all(self) -> list[RLTTransition]:
        records = []
        for path in sorted(self.path.glob("episode_*.pkl"), key=lambda p: p.stat().st_mtime_ns):
            episode_records = [RLTTransition(**v) for v in self._load_episode(path)["transitions"]]
            if not episode_records or self._episode_path(episode_records[0].episode_id) != path:
                raise ValueError(f"Episode identity mismatch: {path}")
            for record in episode_records:
                record.validate(self.spec)
            records.extend(episode_records)
        return records

    def __len__(self):
        return len(self.read_all())


class ReplayBuffer:
    def __init__(self, spec: ReplaySpec, *, capacity: int = 200_000, seed: int = 0):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.spec = spec
        self.capacity = capacity
        self.records: list[RLTTransition] = []
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def extend(self, records):
        for record in records:
            record.validate(self.spec)
            if len(self.records) < self.capacity:
                self.records.append(record)
            else:
                self.records[self.position] = record
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[RLTTransition]:
        if type(batch_size) is not int or batch_size <= 0 or not self.records:
            raise ValueError("Sampling requires positive batch size and nonempty replay")
        return [self.records[int(i)] for i in self.rng.integers(0, len(self.records), size=batch_size)]

    def __len__(self):
        return len(self.records)

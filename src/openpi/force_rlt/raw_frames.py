"""Lossless, bounded raw RGB journal for replaying frozen finger features."""

from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re

import numpy as np


class RawFrameJournal:
    def __init__(self, root: str | Path, episode_id: str, *, max_pending: int = 8):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id):
            raise ValueError("episode_id may contain only letters, digits, dash and underscore")
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.root = Path(root) / episode_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: list[Future] = []

    @staticmethod
    def _write(path: Path, frames: dict[str, np.ndarray]):
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **frames)
        os.replace(temporary, path)

    def append(self, step_id: int, **frames: np.ndarray | None) -> str:
        if step_id < -1 or not frames:
            raise ValueError("Invalid raw frame entry")
        for future in self._pending:
            if future.done():
                future.result()
        self._pending = [future for future in self._pending if not future.done()]
        if len(self._pending) >= self.max_pending:
            raise RuntimeError("Raw RGB writer fell behind; hold robot and stop episode")
        path = self.root / ("baseline.npz" if step_id == -1 else f"step_{step_id:06d}.npz")
        if path.exists():
            raise FileExistsError(path)
        arrays = {}
        for name, value in frames.items():
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(f"{name} must be uint8 HWC RGB")
            arrays[name] = array.copy()
        self._pending.append(self._executor.submit(self._write, path, arrays))
        return str(path)

    def close(self):
        self._executor.shutdown(wait=True)
        for future in self._pending:
            future.result()

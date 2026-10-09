"""Timestamped two-finger observations and four-frame deformation history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FingerFrame:
    frame_id: str
    monotonic_ts: float
    rgb: np.ndarray | None


@dataclass(frozen=True)
class GripFeatures:
    z_grip: np.ndarray
    valid_mask: np.ndarray
    left_frame_ids: tuple[str, ...]
    right_frame_ids: tuple[str, ...]
    timestamps: np.ndarray


class FingerHistory:
    """A synchronized, ordered left/right ring buffer with no-contact baseline."""

    def __init__(self, encoder, *, history: int = 4, max_skew_s: float, max_age_s: float):
        if history <= 0 or max_skew_s <= 0 or max_age_s <= 0:
            raise ValueError("History and timing limits must be positive")
        self.encoder = encoder
        self.history = history
        self.max_skew_s = max_skew_s
        self.max_age_s = max_age_s
        self._frames = deque(maxlen=history)
        self._baseline = None

    @property
    def feature_dim(self) -> int:
        return self.history * 2 * 3 * self.encoder.feature_dim

    def start_episode(self, left: FingerFrame, right: FingerFrame) -> None:
        if left.rgb is None or right.rgb is None or abs(left.monotonic_ts - right.monotonic_ts) > self.max_skew_s:
            raise ValueError("A synchronized no-contact frame is required for each finger")
        self._baseline = self.encoder(left.rgb, right.rgb)
        self._frames.clear()

    def append(self, left: FingerFrame, right: FingerFrame, *, robot_ts: float) -> GripFeatures:
        if self._baseline is None:
            raise RuntimeError("Call start_episode before collecting contact features")
        valid = (
            left.rgb is not None
            and right.rgb is not None
            and abs(left.monotonic_ts - right.monotonic_ts) <= self.max_skew_s
            and 0 <= robot_ts - left.monotonic_ts <= self.max_age_s
            and 0 <= robot_ts - right.monotonic_ts <= self.max_age_s
        )
        if valid:
            features = self.encoder(left.rgb, right.rgb)
            self._frames.append(
                (features, True, left.frame_id, right.frame_id, (left.monotonic_ts, right.monotonic_ts))
            )
        else:
            features = (np.zeros(self.encoder.feature_dim, np.float32),) * 2
            self._frames.append(
                (features, False, left.frame_id, right.frame_id, (left.monotonic_ts, right.monotonic_ts))
            )

        rows = []
        masks = []
        left_ids = []
        right_ids = []
        timestamps = []
        padding = self.history - len(self._frames)
        for _ in range(padding):
            rows.append(np.zeros(6 * self.encoder.feature_dim, np.float32))
            masks.append(False)
            left_ids.append("")
            right_ids.append("")
            timestamps.append((np.nan, np.nan))
        last_valid = self._baseline
        for current, is_valid, left_id, right_id, ts in self._frames:
            if is_valid:
                parts = []
                for finger in range(2):
                    now = current[finger]
                    parts.extend((now, now - self._baseline[finger], now - last_valid[finger]))
                rows.append(np.concatenate(parts).astype(np.float32))
                last_valid = current
            else:
                rows.append(np.zeros(6 * self.encoder.feature_dim, np.float32))
            masks.append(is_valid)
            left_ids.append(left_id)
            right_ids.append(right_id)
            timestamps.append(ts)
        return GripFeatures(
            z_grip=np.concatenate(rows),
            valid_mask=np.asarray(masks, dtype=bool),
            left_frame_ids=tuple(left_ids),
            right_frame_ids=tuple(right_ids),
            timestamps=np.asarray(timestamps, dtype=np.float64),
        )

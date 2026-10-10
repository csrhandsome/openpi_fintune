"""Validate and batch-encode timestamped Panda RGB histories on the GPU server."""

import numpy as np


class FingerRGBInput:
    protocol = "panda-finger-rgb-history-v1"

    def __init__(self, encoder, *, max_skew_s=0.02, max_age_s=0.2):
        self.encoder = encoder
        self.max_skew_s = max_skew_s
        self.max_age_s = max_age_s
        self.session = None
        self.baseline = None
        self.cache = {}

    @property
    def feature_dim(self):
        return 4 * 2 * 3 * self.encoder.feature_dim

    def __call__(self, obs):
        left = np.asarray(obs["observation.gripper_image_left"])
        right = np.asarray(obs["observation.gripper_image_right"])
        times = np.asarray(obs["finger_timestamps"], dtype=np.float64)
        mask = np.asarray(obs["finger_valid_mask"], dtype=bool)
        robot_ts = float(obs["monotonic_ts"])
        if (
            left.shape != right.shape
            or left.ndim != 4
            or left.shape[0] != 4
            or left.shape[-1] != 3
            or left.dtype != np.uint8
            or right.dtype != np.uint8
        ):
            raise ValueError("C requires four matching THWC uint8 finger RGB frames")
        if (
            times.shape != (4, 2)
            or not np.isfinite(times).all()
            or np.any(np.diff(times, axis=0) <= 0)
            or np.any(abs(times[:, 0] - times[:, 1]) > self.max_skew_s)
        ):
            raise ValueError("C finger timestamps must be ordered and synchronized")
        ages = robot_ts - times
        if (
            not np.isfinite(robot_ts)
            or mask.shape != (4,)
            or not mask.all()
            or np.any(ages < -self.max_skew_s)
            or np.any(ages > self.max_age_s)
        ):
            raise ValueError("C requires a complete valid finger history without stale frames")
        session = obs["finger_session_id"]
        if not isinstance(session, str) or not session:
            raise ValueError("C requires a finger session identifier")
        fresh_session = session != self.session
        baseline_left = np.asarray(obs["finger_baseline_left"])
        baseline_right = np.asarray(obs["finger_baseline_right"])
        if (
            baseline_left.shape != left.shape[1:]
            or baseline_right.shape != right.shape[1:]
            or baseline_left.dtype != np.uint8
            or baseline_right.dtype != np.uint8
        ):
            raise ValueError("C baseline must match the finger RGB frames")
        cache = {} if fresh_session else self.cache
        keys = [tuple(ts) for ts in times]
        missing = [i for i, key in enumerate(keys) if key not in cache]
        # Baseline and all uncached frames share one encoder call.
        batch_left = ([baseline_left] if fresh_session else []) + [left[i] for i in missing]
        batch_right = ([baseline_right] if fresh_session else []) + [right[i] for i in missing]
        baseline = self.baseline
        if batch_left:
            encoded = self.encoder.encode_pairs(np.stack(batch_left), np.stack(batch_right))
            offset = int(fresh_session)
            if fresh_session:
                baseline = encoded[0]
            cache.update({keys[i]: encoded[j + offset] for j, i in enumerate(missing)})
        current = np.stack([cache[key] for key in keys])
        previous = np.concatenate([baseline[None], current[:-1]])
        # Same frame/finger/[absolute, baseline delta, temporal delta] order as FingerHistory.
        features = np.stack([current, current - baseline, current - previous], axis=2)
        z_grip = features.reshape(-1).astype(np.float32)
        if z_grip.shape != (self.feature_dim,) or not np.isfinite(z_grip).all():
            raise ValueError("Invalid encoded finger history")
        self.session, self.baseline = session, baseline
        self.cache = {key: cache[key] for key in keys}
        return z_grip

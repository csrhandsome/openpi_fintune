"""Align the ROS collector's 100 Hz action trace to its 30 Hz LeRobot frames."""

import json
from pathlib import Path

import numpy as np
import torch


def aligned_episode_actions(
    root: Path, episode: int, frame_count: int, action_horizon: int, *, max_skew_ms: float = 50.0
) -> torch.Tensor:
    """Return 30 Hz future action chunks for each camera observation.

    The collector's frame state is sampled on the control tick recorded as
    ``host_frame_monotonic_ns``. Read the 100 Hz trace at frame time + k/30 s
    for k=1..action_horizon, even when a camera frame was dropped.
    """
    sync_path = root / f"episode_{episode:06d}.sync.json"
    if not sync_path.is_file():
        raise FileNotFoundError(f"Missing ROS camera/action timing sidecar: {sync_path}")
    sync = json.loads(sync_path.read_text(encoding="utf-8"))
    frames = sync["frame_records"]
    if int(sync["episode_index"]) != episode or len(frames) != frame_count:
        raise ValueError(f"Frame count/index mismatch in {sync_path}: {len(frames)} != {frame_count}")
    if not np.isclose(float(sync["camera_fps"]), 30.0, atol=0.01):
        raise ValueError(f"Expected 30 Hz camera frames in {sync_path}")

    trace_path = root / sync["action_trace"]
    if trace_path.parent.resolve() != root.resolve():
        raise ValueError(f"Action trace must be inside dataset root: {trace_path}")
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing ROS 100 Hz action trace: {trace_path}")
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"Empty action trace: {trace_path}")
    sample_ns = np.asarray([int(row["host_sample_monotonic_ns"]) for row in rows], dtype=np.int64)
    frame_ns = np.asarray([int(frame["host_frame_monotonic_ns"]) for frame in frames], dtype=np.int64)
    if np.any(np.diff(sample_ns) <= 0) or np.any(np.diff(frame_ns) <= 0):
        raise ValueError(f"Nonmonotonic timestamps in {sync_path} or {trace_path}")
    if any(int(frame["frame_index"]) != i for i, frame in enumerate(frames)):
        raise ValueError(f"Nonsequential frame indices in {sync_path}")

    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    target_ns = frame_ns[:, None] + np.rint(
        np.arange(1, action_horizon + 1, dtype=np.float64) * (1e9 / 30.0)
    ).astype(np.int64)[None, :]
    right = np.searchsorted(sample_ns, target_ns)
    left = np.clip(right - 1, 0, len(rows) - 1)
    right = np.clip(right, 0, len(rows) - 1)
    nearest = np.where(abs(sample_ns[left] - target_ns) <= abs(sample_ns[right] - target_ns), left, right)
    skew_ms = abs(sample_ns[nearest] - target_ns) / 1e6
    # The final frame has no future camera step; pad it with the last recorded
    # next-measured action, matching LeRobot's end-of-episode chunk padding.
    if np.any((target_ns <= sample_ns[-1]) & (skew_ms > max_skew_ms)):
        raise ValueError(f"Camera/action timestamp skew exceeds {max_skew_ms:g} ms in {sync_path}")
    measured = np.asarray(
        [[*row["joint_position"], row["gripper_position"]] for row in rows], dtype=np.float32
    )
    final_action = np.asarray(
        [*rows[-1]["action_joint_position"], rows[-1]["action_gripper_position"]], dtype=np.float32
    )
    actions = measured[nearest]
    actions[target_ns > sample_ns[-1]] = final_action
    if actions.shape != (frame_count, action_horizon, 8) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid 8D action data in {trace_path}")
    return torch.from_numpy(actions)


class MultirateFrankaDataset:
    """Replace LeRobot's frame action chunks with timestamp-aligned trace actions."""

    def __init__(self, dataset, action_horizon: int):
        self.dataset = dataset
        self.action_horizon = action_horizon
        self.episode_bounds = {}
        self.episode_actions = {}
        for episode in range(dataset.meta.total_episodes):
            start = int(dataset.episode_data_index["from"][episode])
            end = int(dataset.episode_data_index["to"][episode])
            self.episode_bounds[episode] = (start, end)
            self.episode_actions[episode] = aligned_episode_actions(
                Path(dataset.root), episode, end - start, action_horizon
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        episode = int(item["episode_index"])
        start, end = self.episode_bounds[episode]
        item["actions"] = self.episode_actions[episode][index - start]
        return item

import json

import numpy as np
import pytest
import torch

from openpi.training.franka_multirate import MultirateFrankaDataset, aligned_episode_actions


def _write_episode(root, episode, frame_times, action_times):
    (root / f"episode_{episode:06d}.sync.json").write_text(
        json.dumps(
            {
                "episode_index": episode,
                "camera_fps": 30,
                "action_trace": f"episode_{episode:06d}.actions.jsonl",
                "frame_records": [
                    {"frame_index": i, "host_frame_monotonic_ns": t}
                    for i, t in enumerate(frame_times)
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / f"episode_{episode:06d}.actions.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "host_sample_monotonic_ns": t,
                    "joint_position": [float(i)] * 7,
                    "gripper_position": float(i % 2),
                    "action_joint_position": [float(i)] * 7,
                    "action_gripper_position": float(i % 2),
                }
            )
            for i, t in enumerate(action_times)
        ),
        encoding="utf-8",
    )


def test_trace_actions_follow_camera_frame_time_and_chunk_stays_in_episode(tmp_path):
    base = 1_000_000_000
    action_times = [base + i * 10_000_000 for i in range(9)]
    _write_episode(tmp_path, 0, [base + 2_000_000, base + 35_000_000], action_times)
    _write_episode(tmp_path, 1, [base + 62_000_000], action_times)
    first = aligned_episode_actions(tmp_path, 0, 2, 3)
    np.testing.assert_array_equal(first[:, :, 0].numpy(), [[4, 7, 8], [7, 8, 8]])

    class Dataset:
        root = tmp_path
        meta = type("Meta", (), {"total_episodes": 2})()
        episode_data_index = {"from": torch.tensor([0, 2]), "to": torch.tensor([2, 3])}

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"episode_index": torch.tensor(0 if index < 2 else 1)}

    wrapped = MultirateFrankaDataset(Dataset(), action_horizon=3)
    np.testing.assert_array_equal(wrapped[0]["actions"][:, 0].numpy(), [4, 7, 8])
    np.testing.assert_array_equal(wrapped[1]["actions"][:, 0].numpy(), [7, 8, 8])
    np.testing.assert_array_equal(wrapped[2]["actions"][:, 0].numpy(), [8, 8, 8])


def test_alignment_rejects_large_clock_skew(tmp_path):
    _write_episode(tmp_path, 0, [1_000_000_000], [2_000_000_000, 2_040_000_000])
    with pytest.raises(ValueError, match="skew"):
        aligned_episode_actions(tmp_path, 0, 1, 3)


def test_dropped_camera_frame_does_not_change_action_spacing(tmp_path):
    base = 1_000_000_000
    _write_episode(
        tmp_path,
        0,
        [base, base + 70_000_000],
        [base + i * 10_000_000 for i in range(15)],
    )
    chunks = aligned_episode_actions(tmp_path, 0, 2, 2)
    np.testing.assert_array_equal(chunks[0, :, 0].numpy(), [3, 7])
    np.testing.assert_array_equal(chunks[1, :, 0].numpy(), [10, 14])

import numpy as np
import pytest

from openpi.force_rlt.finger_input import FingerRGBInput
from openpi.force_rlt.observation import FingerFrame
from openpi.force_rlt.observation import FingerHistory


class TinyEncoder:
    feature_dim = 1

    def __init__(self):
        self.batches = []

    def __call__(self, left, right):
        return np.array([left.mean()], np.float32), np.array([right.mean()], np.float32)

    def encode_pairs(self, left, right):
        self.batches.append(len(left))
        return np.array([self(left_image, right_image) for left_image, right_image in zip(left, right, strict=True)])


def request():
    left = np.stack([np.full((2, 2, 3), i, np.uint8) for i in range(1, 5)])
    right = left * 2
    return {
        "observation.gripper_image_left": left,
        "observation.gripper_image_right": right,
        "finger_timestamps": np.repeat(np.array([9.9, 9.93, 9.96, 9.99])[:, None], 2, axis=1),
        "finger_baseline_left": np.zeros((2, 2, 3), np.uint8),
        "finger_baseline_right": np.zeros((2, 2, 3), np.uint8),
        "finger_valid_mask": np.ones(4, bool),
        "finger_session_id": "one",
        "monotonic_ts": 10.0,
    }


def test_rgb_batch_matches_training_history_and_reuses_cached_frames():
    encoder = TinyEncoder()
    adapter = FingerRGBInput(encoder)
    obs = request()
    training = FingerHistory(TinyEncoder(), max_skew_s=0.02, max_age_s=0.2)
    training.start_episode(
        FingerFrame("b", 9.8, obs["finger_baseline_left"]), FingerFrame("b", 9.8, obs["finger_baseline_right"])
    )
    for i, (left, right, times) in enumerate(
        zip(
            obs["observation.gripper_image_left"],
            obs["observation.gripper_image_right"],
            obs["finger_timestamps"],
            strict=True,
        )
    ):
        value = training.append(
            FingerFrame(str(i), times[0], left), FingerFrame(str(i), times[1], right), robot_ts=10.0
        )
    np.testing.assert_array_equal(adapter(obs), value.z_grip)
    np.testing.assert_array_equal(adapter(obs), value.z_grip)
    assert encoder.batches == [5]
    changed = {**obs, "finger_timestamps": obs["finger_timestamps"] + 0.01}
    adapter(changed)
    assert encoder.batches == [5, 4]
    adapter({**obs, "finger_session_id": "new-episode"})
    assert encoder.batches == [5, 4, 5]


@pytest.mark.parametrize(
    "changed",
    [
        {"finger_valid_mask": [True, False, True, True]},
        {"monotonic_ts": 11.0},
        {"finger_timestamps": np.ones((4, 2))},
        {"finger_session_id": ""},
        {"observation.gripper_image_left": np.zeros((2, 2, 3), np.uint8)},
    ],
)
def test_invalid_rgb_history_is_rejected(changed):
    with pytest.raises(ValueError, match="C "):
        FingerRGBInput(TinyEncoder())({**request(), **changed})

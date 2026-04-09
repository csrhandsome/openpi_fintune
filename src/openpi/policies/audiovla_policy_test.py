import numpy as np

from openpi.policies import audiovla_policy
from openpi.shared import audio_tools


def test_audiovla_inputs_materialize_audio_window():
    transform = audiovla_policy.AudiovlaInputs(
        action_dim=32,
        audio_num_samples=6,
        audio_sample_rate=16_000,
    )
    outputs = transform(
        {
            "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros((7,), dtype=np.float32),
            "observation/gripper_position": np.ones((1,), dtype=np.float32),
            "observation/audio": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "observation/audio_sample_rate": np.int32(16_000),
            "actions": np.zeros((50, 8), dtype=np.float32),
            "prompt": "test",
        }
    )

    assert outputs["state"].shape == (32,)
    assert outputs["actions"].shape == (50, 32)
    assert outputs["audio"].shape == (6,)
    assert np.allclose(outputs["audio"], np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert outputs["audio_mask"] == np.True_

    parsed = audiovla_policy.AudiovlaOutputs()({"actions": np.zeros((50, 32), dtype=np.float32)})
    assert parsed["actions"].shape == (50, 8)


def test_audiovla_inputs_can_fall_back_to_episode_frame_audio():
    transform = audiovla_policy.AudiovlaInputs(
        action_dim=32,
        audio_num_samples=6,
        audio_sample_rate=16_000,
        dataset_root="/tmp/does_not_exist",
        audio_strict=False,
    )
    outputs = transform(
        {
            "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros((7,), dtype=np.float32),
            "observation/gripper_position": np.ones((1,), dtype=np.float32),
            "episode_index": np.int64(0),
            "frame_index": np.int64(0),
            "actions": np.zeros((50, 8), dtype=np.float32),
            "prompt": "test",
        }
    )

    assert outputs["audio"].shape == (6,)
    assert np.allclose(outputs["audio"], np.zeros((6,), dtype=np.float32))
    assert outputs["audio_mask"] == np.False_


def test_audiovla_inputs_extract_whisper_embedding(monkeypatch):
    def fake_whisper_embedding(audio, *, sample_rate, model_name, device, pooling):
        assert audio.shape == (6,)
        assert sample_rate == 16_000
        assert model_name == "openai/whisper-tiny"
        assert device == "cpu"
        assert pooling == "mean"
        return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    monkeypatch.setattr(audio_tools, "compute_whisper_audio_embedding", fake_whisper_embedding)

    transform = audiovla_policy.AudiovlaInputs(
        action_dim=32,
        audio_num_samples=6,
        audio_sample_rate=16_000,
        use_whisper=True,
        whisper_model_name="openai/whisper-tiny",
        whisper_output_dim=4,
        whisper_device="cpu",
    )
    outputs = transform(
        {
            "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros((7,), dtype=np.float32),
            "observation/gripper_position": np.ones((1,), dtype=np.float32),
            "observation/audio": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "observation/audio_sample_rate": np.int32(16_000),
            "actions": np.zeros((50, 8), dtype=np.float32),
            "prompt": "test",
        }
    )

    assert outputs["audio"].shape == (4,)
    assert np.allclose(outputs["audio"], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert outputs["audio_mask"] == np.True_

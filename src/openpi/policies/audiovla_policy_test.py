import numpy as np

from openpi.policies import audiovla_policy
from openpi.shared import audio_tools


def test_audiovla_inputs_materialize_audio_window():
    transform = audiovla_policy.AudiovlaInputs(
        action_dim=32,
        audio_num_samples=6,
        instruction_audio_num_samples=4,
        audio_sample_rate=16_000,
        audio_strict=False,
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
    assert outputs["ambient_audio"].shape == (6,)
    assert outputs["instruction_audio"].shape == (4,)
    assert np.allclose(outputs["audio"], np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(outputs["ambient_audio"], np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(outputs["instruction_audio"], np.zeros((4,), dtype=np.float32))
    assert outputs["audio_mask"] == np.True_
    assert outputs["ambient_audio_mask"] == np.True_
    assert outputs["instruction_audio_mask"] == np.False_

    parsed = audiovla_policy.AudiovlaOutputs()({"actions": np.zeros((50, 32), dtype=np.float32)})
    assert parsed["actions"].shape == (50, 8)


def test_audiovla_inputs_can_fall_back_to_episode_frame_audio():
    transform = audiovla_policy.AudiovlaInputs(
        action_dim=32,
        audio_num_samples=6,
        instruction_audio_num_samples=4,
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
    assert outputs["ambient_audio"].shape == (6,)
    assert outputs["instruction_audio"].shape == (4,)
    assert np.allclose(outputs["audio"], np.zeros((6,), dtype=np.float32))
    assert np.allclose(outputs["ambient_audio"], np.zeros((6,), dtype=np.float32))
    assert np.allclose(outputs["instruction_audio"], np.zeros((4,), dtype=np.float32))
    assert outputs["audio_mask"] == np.False_
    assert outputs["ambient_audio_mask"] == np.False_
    assert outputs["instruction_audio_mask"] == np.False_


def test_audiovla_inputs_extract_whisper_embedding(monkeypatch):
    def fake_whisper_sequence(audio, *, sample_rate, model_name, device):
        assert audio.shape in ((6,), (4,))
        assert sample_rate == 16_000
        assert model_name == "openai/whisper-tiny"
        assert device == "cpu"
        return np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

    monkeypatch.setattr(audio_tools, "compute_whisper_audio_sequence", fake_whisper_sequence)

    transform = audiovla_policy.AudiovlaInputs(
        action_dim=32,
        audio_num_samples=6,
        instruction_audio_num_samples=4,
        audio_sample_rate=16_000,
        use_whisper=True,
        whisper_model_name="openai/whisper-tiny",
        whisper_output_dim=4,
        whisper_device="cpu",
        audio_strict=False,
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

    assert outputs["audio"].shape == (1, 4)
    assert outputs["ambient_audio"].shape == (1, 4)
    assert outputs["instruction_audio"].shape == (1, 4)
    assert np.allclose(outputs["audio"], np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    assert np.allclose(outputs["ambient_audio"], np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    assert np.allclose(outputs["instruction_audio"], np.zeros((1, 4), dtype=np.float32))
    assert outputs["audio_mask"] == np.True_
    assert outputs["ambient_audio_mask"] == np.True_
    assert outputs["instruction_audio_mask"] == np.False_


def test_audiovla_inputs_extract_qwen_sequence(monkeypatch):
    def fake_qwen_sequence(audio, *, sample_rate, model_name, device):
        assert audio.shape in ((6,), (4,))
        assert sample_rate == 16_000
        assert model_name == "Qwen/Qwen3-ASR-0.6B"
        assert device == "auto"
        return np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

    monkeypatch.setattr(audio_tools, "compute_qwen_asr_audio_sequence", fake_qwen_sequence)

    transform = audiovla_policy.AudiovlaInputs(
        action_dim=32,
        audio_num_samples=6,
        instruction_audio_num_samples=4,
        audio_sample_rate=16_000,
        use_qwen=True,
        qwen_model_name="Qwen/Qwen3-ASR-0.6B",
        qwen_output_dim=4,
        qwen_device="auto",
        audio_strict=False,
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

    assert outputs["audio"].shape == (1, 4)
    assert outputs["ambient_audio"].shape == (1, 4)
    assert outputs["instruction_audio"].shape == (1, 4)
    assert np.allclose(outputs["audio"], np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    assert np.allclose(outputs["ambient_audio"], np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    assert np.allclose(outputs["instruction_audio"], np.zeros((1, 4), dtype=np.float32))
    assert outputs["audio_mask"] == np.True_
    assert outputs["ambient_audio_mask"] == np.True_
    assert outputs["instruction_audio_mask"] == np.False_


def test_audiovla_inputs_extract_whisper_embedding_baseline(monkeypatch):
    def fake_whisper_embedding(audio, *, sample_rate, model_name, device, pooling):
        assert audio.shape in ((6,), (4,))
        assert sample_rate == 16_000
        assert model_name == "openai/whisper-tiny"
        assert device == "cpu"
        assert pooling == "mean"
        return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    monkeypatch.setattr(audio_tools, "compute_whisper_audio_embedding", fake_whisper_embedding)

    transform = audiovla_policy.AudiovlaInputs(
        action_dim=32,
        audio_num_samples=6,
        instruction_audio_num_samples=4,
        audio_sample_rate=16_000,
        use_whisper=True,
        audio_tokenizer_type="pooled_baseline",
        whisper_model_name="openai/whisper-tiny",
        whisper_output_dim=4,
        whisper_device="cpu",
        audio_strict=False,
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
    assert outputs["ambient_audio"].shape == (4,)
    assert outputs["instruction_audio"].shape == (4,)
    assert np.allclose(outputs["audio"], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert np.allclose(outputs["ambient_audio"], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert np.allclose(outputs["instruction_audio"], np.zeros((4,), dtype=np.float32))
    assert outputs["audio_mask"] == np.True_
    assert outputs["ambient_audio_mask"] == np.True_
    assert outputs["instruction_audio_mask"] == np.False_


def test_audiovla_inputs_loads_instruction_audio_from_vad_sidecar(monkeypatch):
    def fake_extract_instruction_audio(**kwargs):
        assert kwargs["episode_index"] == 3
        assert kwargs["frame_index"] == 11
        return np.array([9.0, 8.0, 7.0, 6.0], dtype=np.float32), True

    monkeypatch.setattr(audio_tools, "extract_episode_instruction_audio", fake_extract_instruction_audio)

    transform = audiovla_policy.AudiovlaInputs(
        action_dim=32,
        audio_num_samples=6,
        instruction_audio_num_samples=4,
        audio_sample_rate=16_000,
        dataset_root="/tmp/example",
        audio_strict=False,
    )
    outputs = transform(
        {
            "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros((7,), dtype=np.float32),
            "observation/gripper_position": np.ones((1,), dtype=np.float32),
            "episode_index": np.int64(3),
            "frame_index": np.int64(11),
            "actions": np.zeros((50, 8), dtype=np.float32),
            "prompt": "test",
        }
    )

    assert np.allclose(outputs["instruction_audio"], np.array([9.0, 8.0, 7.0, 6.0], dtype=np.float32))
    assert outputs["instruction_audio_mask"] == np.True_

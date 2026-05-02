import types

import numpy as np
import torch

from openpi.shared import audio_tools


def test_fit_audio_window_handles_empty_audio():
    output = audio_tools.fit_audio_window(np.array([], dtype=np.float32), 4)

    assert output.shape == (4,)
    assert np.allclose(output, np.zeros((4,), dtype=np.float32))


def test_pcm16_bytes_to_mono_float32_stereo_average():
    raw = np.array([[32767, 32767], [-32767, -32767]], dtype=np.int16).tobytes()

    output = audio_tools.pcm16_bytes_to_mono_float32(raw, channels=2)

    assert output.shape == (2,)
    assert np.allclose(output, np.array([1.0, -1.0], dtype=np.float32))


def test_pcm_array_to_mono_float32_stereo_average():
    pcm = np.array([[32767, -32767], [16384, -16384]], dtype=np.int16)

    output = audio_tools.pcm_array_to_mono_float32(pcm)

    assert output.shape == (2,)
    assert np.allclose(output, np.zeros((2,), dtype=np.float32), atol=1e-4)


def test_compute_whisper_audio_sequence_pads_for_encoder_and_keeps_real_prefix(monkeypatch):
    calls = {}

    class FakeFeatureExtractor:
        def __call__(self, audio, *, sampling_rate, return_tensors, padding, truncation):
            calls["feature_extractor"] = {
                "audio_shape": audio.shape,
                "sampling_rate": sampling_rate,
                "return_tensors": return_tensors,
                "padding": padding,
                "truncation": truncation,
            }
            return types.SimpleNamespace(input_features=torch.zeros((1, 80, 3000)))

    class FakeEncoder:
        def __call__(self, *, input_features):
            del input_features
            hidden = torch.arange(1500, dtype=torch.float32).reshape(1, 1500, 1)
            return types.SimpleNamespace(last_hidden_state=hidden)

    fake_model = types.SimpleNamespace(encoder=FakeEncoder())
    monkeypatch.setattr(
        audio_tools,
        "_load_whisper_components",
        lambda model_name, device: (FakeFeatureExtractor(), fake_model, "cpu"),
    )

    output = audio_tools.compute_whisper_audio_sequence(
        np.ones((16_000,), dtype=np.float32),
        sample_rate=16_000,
        model_name="openai/whisper-tiny",
        device="cpu",
    )

    assert calls["feature_extractor"] == {
        "audio_shape": (16_000,),
        "sampling_rate": 16_000,
        "return_tensors": "pt",
        "padding": "max_length",
        "truncation": True,
    }
    assert output.shape == (50, 1)
    assert np.allclose(output[:, 0], np.arange(50, dtype=np.float32))

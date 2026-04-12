import numpy as np

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

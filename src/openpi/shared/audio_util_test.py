from openpi.shared import audio_util


def test_window_num_samples_for_duration_preserves_duration_across_sample_rates():
    assert audio_util.window_num_samples_for_duration(
        16_000,
        sample_rate=16_000,
        target_sample_rate=16_000,
    ) == 16_000
    assert audio_util.window_num_samples_for_duration(
        16_000,
        sample_rate=48_000,
        target_sample_rate=16_000,
    ) == 48_000
    assert audio_util.window_num_samples_for_duration(
        32_000,
        sample_rate=48_000,
        target_sample_rate=16_000,
    ) == 96_000

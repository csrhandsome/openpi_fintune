from __future__ import annotations


def window_num_samples_for_duration(
    window_num_samples: int,
    *,
    sample_rate: int,
    target_sample_rate: int,
) -> int:
    """Convert a target-rate window length into the equivalent raw sample count."""
    if window_num_samples <= 0:
        raise ValueError("window_num_samples must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be > 0")

    duration_s = float(window_num_samples) / float(target_sample_rate)
    return max(int(round(duration_s * float(sample_rate))), 1)

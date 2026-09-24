from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

AudioDegradationMethod = Literal["random", "vad_trim", "repeat_speech"]
RepeatMode = Literal["tail", "twice"]

_ERROR_TYPES: tuple[Literal["vad_trim", "repeat_speech"], ...] = ("vad_trim", "repeat_speech")
_VAD_TRIM_FAILURE_CASES: tuple[tuple[int, int], ...] = (
    (700, 700),
    (800, 800),
    (900, 900),
    (900, 1100),
    (1100, 900),
    (1100, 1100),
    (1200, 1200),
)
_REPEAT_SPEECH_FAILURE_CASES: tuple[tuple[RepeatMode, int, int], ...] = (
    ("tail", 600, 0),
    ("tail", 600, 200),
    ("tail", 1000, 0),
    ("tail", 1000, 200),
    ("tail", 1400, 0),
    ("tail", 1400, 200),
    ("tail", 1800, 0),
    ("twice", 0, 0),
    ("twice", 0, 200),
)


@dataclass(frozen=True)
class AudioDegradationConfig:
    """Small ASR failure cases: VAD edge miss or repeated/interrupted speech."""

    enabled: bool = True
    method: AudioDegradationMethod = "random"
    trim_start_ms: int | None = None
    trim_end_ms: int | None = None
    min_remaining_ms: int = 250
    repeat_mode: RepeatMode | None = None
    repeat_start_after_ms: int | None = None
    repeat_gap_ms: int | None = None


@dataclass
class AudioDegradationReport:
    enabled: bool
    method: str
    error_type: str | None
    sample_rate: int
    original_samples: int
    degraded_samples: int
    trim_start_ms: int = 0
    trim_end_ms: int = 0
    repeat_mode: str | None = None
    repeat_start_after_ms: int | None = None
    repeat_gap_ms: int | None = None
    repeated_samples: int = 0


def _clip_float_audio(audio: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)


def _ms_to_samples(ms: int, sample_rate: int) -> int:
    return max(0, round(int(ms) * sample_rate / 1000.0))


def trim_vad_missed_edges(
    audio: np.ndarray,
    *,
    sample_rate: int,
    trim_start_ms: int = 500,
    trim_end_ms: int = 500,
    min_remaining_ms: int = 250,
) -> tuple[np.ndarray, int, int]:
    """Simulate VAD missing the start/end of the speech command."""

    audio = _clip_float_audio(audio)
    if audio.size == 0:
        return audio, 0, 0

    start_samples = _ms_to_samples(trim_start_ms, sample_rate)
    end_samples = _ms_to_samples(trim_end_ms, sample_rate)
    min_remaining_samples = min(audio.size, max(1, _ms_to_samples(min_remaining_ms, sample_rate)))
    crop_budget = max(0, audio.size - min_remaining_samples)
    requested = start_samples + end_samples

    if requested > crop_budget:
        if requested <= 0 or crop_budget <= 0:
            start_samples = 0
            end_samples = 0
        else:
            scale = crop_budget / requested
            start_samples = round(start_samples * scale)
            end_samples = max(0, crop_budget - start_samples)

    stop = audio.size - end_samples if end_samples > 0 else audio.size
    trimmed = audio[start_samples:stop].astype(np.float32)
    actual_start_ms = round(start_samples * 1000 / sample_rate)
    actual_end_ms = round(end_samples * 1000 / sample_rate)
    return trimmed, actual_start_ms, actual_end_ms


def repeat_interrupted_speech(
    audio: np.ndarray,
    *,
    sample_rate: int,
    start_after_ms: int = 1000,
    gap_ms: int = 0,
) -> tuple[np.ndarray, int]:
    """Append the utterance again, but start the repeat after the first second."""

    audio = _clip_float_audio(audio)
    if audio.size == 0:
        return audio, 0

    start_samples = min(audio.size, _ms_to_samples(start_after_ms, sample_rate))
    repeated = audio[start_samples:]
    if repeated.size == 0:
        repeated = audio

    if gap_ms > 0:
        gap = np.zeros(_ms_to_samples(gap_ms, sample_rate), dtype=np.float32)
        output = np.concatenate([audio, gap, repeated]).astype(np.float32)
    else:
        output = np.concatenate([audio, repeated]).astype(np.float32)
    return output, int(repeated.size)


def repeat_full_speech(
    audio: np.ndarray,
    *,
    sample_rate: int,
    gap_ms: int = 0,
) -> tuple[np.ndarray, int]:
    """Append the complete utterance a second time."""

    audio = _clip_float_audio(audio)
    if audio.size == 0:
        return audio, 0

    if gap_ms > 0:
        gap = np.zeros(_ms_to_samples(gap_ms, sample_rate), dtype=np.float32)
        output = np.concatenate([audio, gap, audio]).astype(np.float32)
    else:
        output = np.concatenate([audio, audio]).astype(np.float32)
    return output, int(audio.size)


def _sample_case(cases: tuple, rng: np.random.Generator):
    return cases[int(rng.integers(0, len(cases)))]


def degrade_audio_for_asr_baseline(
    audio: np.ndarray,
    *,
    sample_rate: int,
    config: AudioDegradationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, AudioDegradationReport]:
    config = config or AudioDegradationConfig()
    rng = rng or np.random.default_rng()
    audio = _clip_float_audio(audio)
    method = config.method
    error_type = _sample_case(_ERROR_TYPES, rng) if method == "random" else method
    report = AudioDegradationReport(
        enabled=config.enabled,
        method=method,
        error_type=error_type,
        sample_rate=sample_rate,
        original_samples=int(audio.size),
        degraded_samples=int(audio.size),
    )

    if not config.enabled:
        report.method = "disabled"
        report.error_type = None
        return audio, report

    if error_type == "vad_trim":
        if config.trim_start_ms is None or config.trim_end_ms is None:
            trim_start_ms, trim_end_ms = _sample_case(_VAD_TRIM_FAILURE_CASES, rng)
        else:
            trim_start_ms = config.trim_start_ms
            trim_end_ms = config.trim_end_ms

        degraded, trim_start_ms, trim_end_ms = trim_vad_missed_edges(
            audio,
            sample_rate=sample_rate,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
            min_remaining_ms=config.min_remaining_ms,
        )
        report.trim_start_ms = trim_start_ms
        report.trim_end_ms = trim_end_ms
    elif error_type == "repeat_speech":
        if config.repeat_mode is None or config.repeat_gap_ms is None or (
            config.repeat_mode == "tail" and config.repeat_start_after_ms is None
        ):
            repeat_mode, repeat_start_after_ms, repeat_gap_ms = _sample_case(_REPEAT_SPEECH_FAILURE_CASES, rng)
        else:
            repeat_mode = config.repeat_mode
            repeat_start_after_ms = config.repeat_start_after_ms or 0
            repeat_gap_ms = config.repeat_gap_ms

        report.repeat_mode = repeat_mode
        report.repeat_start_after_ms = repeat_start_after_ms
        report.repeat_gap_ms = repeat_gap_ms

        if repeat_mode == "tail":
            degraded, repeated_samples = repeat_interrupted_speech(
                audio,
                sample_rate=sample_rate,
                start_after_ms=repeat_start_after_ms,
                gap_ms=repeat_gap_ms,
            )
        elif repeat_mode == "twice":
            degraded, repeated_samples = repeat_full_speech(
                audio,
                sample_rate=sample_rate,
                gap_ms=repeat_gap_ms,
            )
        else:
            raise ValueError(f"Unsupported repeat mode: {repeat_mode!r}")
        report.repeated_samples = repeated_samples
    else:
        raise ValueError(f"Unsupported ASR degradation method: {method!r}")

    degraded = _clip_float_audio(degraded)
    report.degraded_samples = int(degraded.size)
    return degraded, report

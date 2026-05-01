from __future__ import annotations

import dataclasses
import importlib.metadata
from typing import Any
import zlib

import numpy as np
from silero_vad import get_speech_timestamps
from silero_vad import load_silero_vad

from openpi.shared import audio_tools


@dataclasses.dataclass(frozen=True)
class AudioConditioningConfig:
    audio_required: bool = True
    audio_device: str | None = None
    audio_sample_rate: int = 16_000
    audio_channels: int = 1
    audio_num_samples: int = 16_000
    instruction_audio_num_samples: int = 16_000
    audio_buffer_num_samples: int = 32_000
    audio_block_duration_s: float = 0.05
    audio_startup_timeout_s: float = 1.0
    vad_history_num_samples: int = 80_000
    vad_threshold: float = 0.5
    vad_neg_threshold: float | None = None
    vad_min_speech_duration_ms: int = 250
    vad_max_speech_duration_s: float = 5.0
    vad_min_silence_duration_ms: int = 100
    vad_speech_pad_ms: int = 30


def audio_conditioning_config_from_args(args: Any) -> AudioConditioningConfig:
    return AudioConditioningConfig(
        audio_required=bool(args.audio_required),
        audio_device=args.audio_device,
        audio_sample_rate=int(args.audio_sample_rate),
        audio_channels=int(args.audio_channels),
        audio_num_samples=int(args.audio_num_samples),
        instruction_audio_num_samples=int(args.instruction_audio_num_samples),
        audio_buffer_num_samples=int(args.audio_buffer_num_samples),
        audio_block_duration_s=float(args.audio_block_duration_s),
        audio_startup_timeout_s=float(args.audio_startup_timeout_s),
        vad_history_num_samples=int(args.vad_history_num_samples),
        vad_threshold=float(args.vad_threshold),
        vad_neg_threshold=(
            None if args.vad_neg_threshold is None else float(args.vad_neg_threshold)
        ),
        vad_min_speech_duration_ms=int(args.vad_min_speech_duration_ms),
        vad_max_speech_duration_s=float(args.vad_max_speech_duration_s),
        vad_min_silence_duration_ms=int(args.vad_min_silence_duration_ms),
        vad_speech_pad_ms=int(args.vad_speech_pad_ms),
    )


def validate_audio_conditioning_config(config: AudioConditioningConfig) -> None:
    if config.audio_sample_rate <= 0:
        raise ValueError("--audio-sample-rate must be > 0")
    if config.audio_channels <= 0:
        raise ValueError("--audio-channels must be > 0")
    if config.audio_num_samples <= 0:
        raise ValueError("--audio-num-samples must be > 0")
    if config.instruction_audio_num_samples <= 0:
        raise ValueError("--instruction-audio-num-samples must be > 0")
    if config.audio_buffer_num_samples <= 0:
        raise ValueError("--audio-buffer-num-samples must be > 0")
    if config.audio_block_duration_s <= 0.0:
        raise ValueError("--audio-block-duration-s must be > 0")
    if config.audio_startup_timeout_s < 0.0:
        raise ValueError("--audio-startup-timeout-s must be >= 0")
    if config.vad_history_num_samples <= 0:
        raise ValueError("--vad-history-num-samples must be > 0")
    if config.vad_threshold <= 0.0 or config.vad_threshold >= 1.0:
        raise ValueError("--vad-threshold must be in (0, 1)")
    if config.vad_neg_threshold is not None and (
        config.vad_neg_threshold <= 0.0 or config.vad_neg_threshold >= 1.0
    ):
        raise ValueError("--vad-neg-threshold must be in (0, 1)")
    if config.vad_min_speech_duration_ms <= 0:
        raise ValueError("--vad-min-speech-duration-ms must be > 0")
    if config.vad_max_speech_duration_s <= 0.0:
        raise ValueError("--vad-max-speech-duration-s must be > 0")
    if config.vad_min_silence_duration_ms < 0:
        raise ValueError("--vad-min-silence-duration-ms must be >= 0")
    if config.vad_speech_pad_ms < 0:
        raise ValueError("--vad-speech-pad-ms must be >= 0")


def _audio_to_mono_float(audio: np.ndarray) -> np.ndarray:
    pcm = np.asarray(audio)
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1)
    elif pcm.ndim > 2:
        raise ValueError(f"Expected mono/stereo audio, got shape {pcm.shape}")

    if np.issubdtype(pcm.dtype, np.integer):
        dtype_info = np.iinfo(pcm.dtype)
        if pcm.dtype.kind == "u":
            midpoint = float(dtype_info.max + 1) / 2.0
            return ((pcm.astype(np.float32) - midpoint) / midpoint).astype(np.float32)
        scale = max(float(dtype_info.max), float(-dtype_info.min), 1.0)
        return (pcm.astype(np.float32) / scale).astype(np.float32)

    return np.clip(pcm.astype(np.float32), -1.0, 1.0)


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)

    duration_s = float(audio.shape[0]) / float(source_rate)
    target_len = max(1, int(round(duration_s * float(target_rate))))
    source_x = np.linspace(0.0, duration_s, num=audio.shape[0], endpoint=False)
    target_x = np.linspace(0.0, duration_s, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


def _prepare_vad_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    mono = _audio_to_mono_float(audio)
    if sample_rate in (8_000, 16_000):
        return np.ascontiguousarray(mono, dtype=np.float32), int(sample_rate)
    return _resample_linear(mono, int(sample_rate), 16_000), 16_000


def _detect_segments(
    *,
    audio: np.ndarray,
    sample_rate: int,
    model: Any,
    threshold: float,
    min_speech_duration_ms: int,
    max_speech_duration_s: float,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    neg_threshold: float | None,
) -> tuple[list[dict[str, int]], int, int]:
    import torch

    vad_audio, vad_sample_rate = _prepare_vad_audio(audio, sample_rate)
    timestamps = get_speech_timestamps(
        torch.from_numpy(vad_audio),
        model,
        threshold=float(threshold),
        sampling_rate=int(vad_sample_rate),
        min_speech_duration_ms=int(min_speech_duration_ms),
        max_speech_duration_s=float(max_speech_duration_s),
        min_silence_duration_ms=int(min_silence_duration_ms),
        speech_pad_ms=int(speech_pad_ms),
        return_seconds=False,
        neg_threshold=neg_threshold,
    )
    normalized = [
        {"start": int(item["start"]), "end": int(item["end"])}
        for item in timestamps
        if int(item.get("end", 0)) > int(item.get("start", 0))
    ]
    return normalized, int(vad_sample_rate), int(vad_audio.shape[0])


class SileroVADInstructionTracker:
    """Track the latest completed speech segment and expose it as instruction audio."""

    def __init__(self, config: AudioConditioningConfig):
        self._history_num_samples = int(config.vad_history_num_samples)
        self._target_sample_rate = int(config.audio_sample_rate)
        self._instruction_num_samples = int(config.instruction_audio_num_samples)
        self._threshold = float(config.vad_threshold)
        self._neg_threshold = config.vad_neg_threshold
        self._min_speech_duration_ms = int(config.vad_min_speech_duration_ms)
        self._max_speech_duration_s = float(config.vad_max_speech_duration_s)
        self._min_silence_duration_ms = int(config.vad_min_silence_duration_ms)
        self._speech_pad_ms = int(config.vad_speech_pad_ms)
        self._completion_margin_ms = max(self._min_silence_duration_ms, self._speech_pad_ms)
        self._model = load_silero_vad()
        self._model_info = {
            "tool": "silero-vad",
            "tool_version": importlib.metadata.version("silero-vad"),
            "model_loader": "silero_vad.load_silero_vad",
        }
        self.reset()

    @property
    def history_num_samples(self) -> int:
        return self._history_num_samples

    @property
    def model_info(self) -> dict[str, Any]:
        return self._model_info

    def reset(self) -> None:
        self._instruction_audio = np.zeros((self._instruction_num_samples,), dtype=np.float32)
        self._instruction_audio_mask = np.bool_(False)
        self._instruction_audio_sample_rate = np.int32(self._target_sample_rate)
        self._last_segment_crc: int | None = None

    def current(self) -> tuple[np.ndarray, np.int32, np.bool_]:
        return (
            np.array(self._instruction_audio, copy=True),
            np.int32(self._instruction_audio_sample_rate),
            np.bool_(self._instruction_audio_mask),
        )

    def update(self, audio: np.ndarray, *, sample_rate: int, has_audio: bool) -> tuple[np.ndarray, np.int32, np.bool_]:
        history_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not has_audio or history_audio.size == 0:
            return self.current()

        segments, vad_sample_rate, vad_audio_num_samples = _detect_segments(
            audio=history_audio,
            sample_rate=int(sample_rate),
            model=self._model,
            threshold=self._threshold,
            min_speech_duration_ms=self._min_speech_duration_ms,
            max_speech_duration_s=self._max_speech_duration_s,
            min_silence_duration_ms=self._min_silence_duration_ms,
            speech_pad_ms=self._speech_pad_ms,
            neg_threshold=self._neg_threshold,
        )

        completion_margin_samples = int(round(vad_sample_rate * self._completion_margin_ms / 1000.0))
        latest_segment = None
        for segment in segments:
            if int(segment["end"]) > vad_audio_num_samples - completion_margin_samples:
                continue
            latest_segment = segment

        if latest_segment is None:
            return self.current()

        start_sec = float(latest_segment["start"]) / float(vad_sample_rate)
        end_sec = float(latest_segment["end"]) / float(vad_sample_rate)
        start_sample = int(np.clip(round(start_sec * sample_rate), 0, history_audio.shape[0]))
        end_sample = int(np.clip(round(end_sec * sample_rate), start_sample, history_audio.shape[0]))
        segment_audio = np.ascontiguousarray(history_audio[start_sample:end_sample], dtype=np.float32)
        if segment_audio.size == 0:
            return self.current()

        segment_crc = zlib.crc32(segment_audio.view(np.uint8))
        if segment_crc != self._last_segment_crc:
            duration_s = float(segment_audio.shape[0]) / float(sample_rate)
            print(f"[Audio] Updated instruction_audio from Silero-VAD: {duration_s:.2f}s segment at {sample_rate} Hz")

        self._instruction_audio = segment_audio
        self._instruction_audio_sample_rate = np.int32(sample_rate)
        self._instruction_audio_mask = np.bool_(True)
        self._last_segment_crc = segment_crc
        return self.current()


def _start_audio_buffer(config: AudioConditioningConfig) -> audio_tools.MultiprocessAudioBuffer | None:
    window_num_samples = max(int(config.audio_num_samples), int(config.instruction_audio_num_samples))
    buffer_num_samples = max(
        int(config.audio_buffer_num_samples),
        int(config.audio_num_samples),
        int(config.instruction_audio_num_samples),
        int(config.vad_history_num_samples),
    )
    audio_buffer = audio_tools.MultiprocessAudioBuffer(
        sample_rate=config.audio_sample_rate,
        channels=config.audio_channels,
        window_num_samples=window_num_samples,
        buffer_num_samples=buffer_num_samples,
        block_duration_s=config.audio_block_duration_s,
        device=config.audio_device,
    )
    try:
        audio_buffer.start()
        ready = audio_buffer.wait_until_ready(config.audio_startup_timeout_s)
    except Exception as exc:
        audio_buffer.stop()
        if config.audio_required:
            raise RuntimeError(f"Failed to start microphone capture: {exc}") from exc
        print(f"[Audio] Warning: microphone capture unavailable, continuing without audio: {exc}")
        return None

    if not ready:
        audio_buffer.stop()
        msg = f"No microphone samples arrived within {config.audio_startup_timeout_s:.2f}s."
        if config.audio_required:
            raise RuntimeError(msg)
        print(f"[Audio] Warning: {msg} Continuing without audio.")
        return None

    print(
        "[Audio] Live microphone capture enabled (multiprocess): "
        f"{audio_buffer.sample_rate} Hz, {config.audio_channels} channel(s), "
        f"window={window_num_samples} samples, "
        f"buffer={buffer_num_samples} samples, "
        f"device={audio_buffer.input_device if audio_buffer.input_device is not None else 'default'}"
    )
    return audio_buffer


class LiveAudioConditioning:
    def __init__(self, config: AudioConditioningConfig):
        self._config = config
        self._audio_buffer: audio_tools.MultiprocessAudioBuffer | None = None
        self._instruction_audio_tracker: SileroVADInstructionTracker | None = None

    def start(self) -> None:
        self._audio_buffer = _start_audio_buffer(self._config)
        if self._audio_buffer is not None:
            self._instruction_audio_tracker = SileroVADInstructionTracker(self._config)
            print(
                "[Audio] Silero-VAD instruction tracker enabled: "
                f"{self._instruction_audio_tracker.model_info['model_loader']} "
                f"(version={self._instruction_audio_tracker.model_info['tool_version'] or 'unknown'})"
            )

    def stop(self) -> None:
        if self._audio_buffer is not None:
            self._audio_buffer.stop()
            self._audio_buffer = None
        self._instruction_audio_tracker = None

    def reset_instruction(self) -> None:
        if self._instruction_audio_tracker is not None:
            self._instruction_audio_tracker.reset()

    def build_request_fields(self) -> dict[str, Any]:
        if self._audio_buffer is None:
            return {}

        request_data: dict[str, Any] = {}
        audio_window, has_audio = self._audio_buffer.read_window_for_duration(
            target_num_samples=self._config.audio_num_samples,
            target_sample_rate=self._config.audio_sample_rate,
        )
        request_data["observation/audio"] = audio_window
        request_data["observation/audio_sample_rate"] = np.int32(self._audio_buffer.sample_rate)
        request_data["observation/audio_mask"] = np.bool_(has_audio)

        if self._instruction_audio_tracker is not None:
            vad_history_window, vad_has_audio = self._audio_buffer.read_window_for_duration(
                target_num_samples=self._instruction_audio_tracker.history_num_samples,
                target_sample_rate=self._config.audio_sample_rate,
            )
            instruction_audio, instruction_audio_sample_rate, has_instruction_audio = self._instruction_audio_tracker.update(
                vad_history_window,
                sample_rate=self._audio_buffer.sample_rate,
                has_audio=vad_has_audio,
            )
            request_data["observation/instruction_audio"] = instruction_audio
            request_data["observation/instruction_audio_sample_rate"] = instruction_audio_sample_rate
            request_data["observation/instruction_audio_mask"] = np.bool_(has_instruction_audio)

        return request_data

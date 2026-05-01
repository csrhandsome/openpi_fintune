from __future__ import annotations

import collections
import contextlib
import functools
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import time
import wave

import numpy as np

from openpi.shared import audio_util as _audio_util

_PCM16_MAX = float(np.iinfo(np.int16).max)
_WHISPER_HIDDEN_SIZES = {
    "openai/whisper-tiny": 384,
    "openai/whisper-tiny.en": 384,
    "openai/whisper-base": 512,
    "openai/whisper-base.en": 512,
    "openai/whisper-small": 768,
    "openai/whisper-small.en": 768,
    "openai/whisper-medium": 1024,
    "openai/whisper-medium.en": 1024,
    "openai/whisper-large-v1": 1280,
    "openai/whisper-large-v2": 1280,
    "openai/whisper-large-v3": 1280,
    "openai/whisper-large": 1280,
    "openai/whisper-turbo": 1280,
}


def infer_whisper_hidden_size(model_name: str) -> int:
    try:
        return _WHISPER_HIDDEN_SIZES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Whisper model '{model_name}'. "
            "Set whisper_output_dim explicitly or extend _WHISPER_HIDDEN_SIZES."
        ) from exc


def _resolve_whisper_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=8)
def _load_whisper_components(model_name: str, device: str):
    import torch
    from transformers import WhisperFeatureExtractor
    from transformers import WhisperModel

    resolved_device = _resolve_whisper_device(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)
    model.eval()
    model.to(resolved_device)
    if resolved_device.startswith("cuda"):
        model.to(dtype=torch.float16)
    return feature_extractor, model, resolved_device


def compute_whisper_audio_embedding(
    audio: np.ndarray,
    *,
    sample_rate: int,
    model_name: str,
    device: str = "cpu",
    pooling: str = "mean",
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.ndim != 1:
        raise ValueError(f"Expected 1D audio, got shape {audio.shape}")

    feature_extractor, model, resolved_device = _load_whisper_components(
        model_name, device
    )

    import torch

    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(resolved_device)
    if resolved_device.startswith("cuda"):
        input_features = input_features.to(dtype=torch.float16)

    with torch.inference_mode():
        hidden = model.encoder(input_features=input_features).last_hidden_state

    if pooling == "mean":
        pooled = hidden.mean(dim=1)  # [b, T, d] -> [b, d]
    elif pooling == "first":
        pooled = hidden[:, 0, :]  # [b, T, d] -> [b, d]
    else:
        raise ValueError(f"Unsupported Whisper pooling '{pooling}'")

    return pooled[0].to(dtype=torch.float32, device="cpu").numpy()


def compute_whisper_audio_sequence(
    audio: np.ndarray,
    *,
    sample_rate: int,
    model_name: str,
    device: str = "cpu",
) -> np.ndarray:
    """Extract Whisper encoder hidden states without collapsing the time dimension."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.ndim != 1:
        raise ValueError(f"Expected 1D audio, got shape {audio.shape}")

    feature_extractor, model, resolved_device = _load_whisper_components(
        model_name, device
    )

    import torch

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="do_not_pad",
    )
    input_features = inputs.input_features.to(resolved_device)
    if resolved_device.startswith("cuda"):
        input_features = input_features.to(dtype=torch.float16)

    with torch.inference_mode():
        hidden = model.encoder(input_features=input_features).last_hidden_state

    return hidden[0].to(dtype=torch.float32, device="cpu").numpy()


def whisper_num_frames_for_duration(
    num_samples: int,
    *,
    sample_rate: int,
) -> int:
    """Approximate Whisper encoder output length at 20 ms resolution."""
    if num_samples <= 0:
        return 1
    return max(1, int(round(float(num_samples) / float(sample_rate) * 50.0)))


def fit_audio_feature_sequence(
    features: np.ndarray,
    target_num_frames: int,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected audio feature sequence with shape [T, D], got {features.shape}")
    if target_num_frames <= 0:
        raise ValueError("target_num_frames must be > 0")

    if features.shape[0] >= target_num_frames:
        return np.ascontiguousarray(features[-target_num_frames:], dtype=np.float32)

    padded = np.zeros((target_num_frames, features.shape[1]), dtype=np.float32)
    padded[-features.shape[0] :, :] = features
    return padded


def episode_audio_stem(episode_index: int) -> str:
    return f"episode_{int(episode_index):06d}"


def resolve_episode_audio_paths(
    dataset_root: str | Path, episode_index: int
) -> tuple[Path, Path]:
    audio_dir = Path(dataset_root) / "audio"
    stem = episode_audio_stem(episode_index)
    return audio_dir / f"{stem}.wav", audio_dir / f"{stem}.sync.json"


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


@functools.lru_cache(maxsize=256)
def load_json(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=32)
def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    with wave.open(str(path), "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        sample_rate = int(wav_file.getframerate())
        sample_width = int(wav_file.getsampwidth())
        num_frames = int(wav_file.getnframes())
        raw = wav_file.readframes(num_frames)

    dtype_map = {
        1: np.uint8,
        2: np.int16,
        4: np.int32,
    }
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes ({path})")

    audio = np.frombuffer(raw, dtype=dtype_map[sample_width])
    audio = audio.reshape(-1, channels)
    if audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val

    audio = np.mean(audio, axis=1) if channels > 1 else audio[:, 0]
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def resolve_audio_path_from_sync(
    dataset_root: str | Path, sync_data: dict | None
) -> Path | None:
    if not sync_data:
        return None
    raw_audio_path = sync_data.get("audio_path")
    if not raw_audio_path:
        return None

    audio_path = Path(str(raw_audio_path))
    if audio_path.is_absolute():
        return audio_path
    return Path(dataset_root) / audio_path


def resolve_dataset_path(
    dataset_root: str | Path, raw_path: str | Path | None
) -> Path | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return Path(dataset_root) / path


def get_frame_record(frame_records: list[dict], frame_index: int) -> dict | None:
    if 0 <= frame_index < len(frame_records):
        record = frame_records[frame_index]
        if _safe_int(record.get("frame_index")) in (None, frame_index):
            return record

    for record in frame_records:
        if _safe_int(record.get("frame_index")) == frame_index:
            return record
    return None


def frame_range_to_audio_samples(
    *,
    sync_data: dict | None,
    start_frame_index: int,
    end_frame_index: int,
    sample_rate: int,
    audio_total_samples: int,
    fps: float,
) -> tuple[int, int]:
    fps_safe = max(float(fps), 1e-6)
    fallback_start = int(
        np.clip(
            round((start_frame_index / fps_safe) * sample_rate), 0, audio_total_samples
        )
    )
    fallback_end = int(
        np.clip(
            round((end_frame_index / fps_safe) * sample_rate), 0, audio_total_samples
        )
    )

    if not sync_data:
        return fallback_start, max(fallback_start, fallback_end)

    audio_start_ns = _safe_int(sync_data.get("audio_start_monotonic_ns"))
    frame_records = sync_data.get("frame_records")
    if (
        audio_start_ns is None
        or not isinstance(frame_records, list)
        or not frame_records
    ):
        return fallback_start, max(fallback_start, fallback_end)

    start_record = get_frame_record(frame_records, start_frame_index)
    end_record = get_frame_record(frame_records, end_frame_index - 1)
    next_record = get_frame_record(frame_records, end_frame_index)
    if start_record is None or end_record is None:
        return fallback_start, max(fallback_start, fallback_end)

    start_ns = _safe_int(start_record.get("host_frame_monotonic_ns"))
    end_ns = (
        _safe_int(next_record.get("host_frame_monotonic_ns"))
        if next_record is not None
        else None
    )
    if end_ns is None:
        last_ns = _safe_int(end_record.get("host_frame_monotonic_ns"))
        if last_ns is not None:
            end_ns = last_ns + round(1e9 / fps_safe)

    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return fallback_start, max(fallback_start, fallback_end)

    start_sample = int(
        np.clip(
            round(((start_ns - audio_start_ns) / 1e9) * sample_rate),
            0,
            audio_total_samples,
        )
    )
    end_sample = int(
        np.clip(
            round(((end_ns - audio_start_ns) / 1e9) * sample_rate),
            0,
            audio_total_samples,
        )
    )
    return start_sample, max(start_sample, end_sample)


def resample_audio(
    audio: np.ndarray, orig_sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")
    if orig_sample_rate == target_sample_rate or audio.size == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)

    target_len = max(round(audio.shape[0] * target_sample_rate / orig_sample_rate), 1)
    source_positions = np.arange(audio.shape[0], dtype=np.float32)
    target_positions = np.linspace(
        0, audio.shape[0] - 1, num=target_len, dtype=np.float32
    )
    return np.ascontiguousarray(
        np.interp(target_positions, source_positions, audio).astype(np.float32)
    )


def fit_audio_window(audio: np.ndarray, window_num_samples: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.shape[0] == 0:
        return np.zeros((window_num_samples,), dtype=np.float32)
    if audio.shape[0] >= window_num_samples:
        return np.ascontiguousarray(audio[-window_num_samples:], dtype=np.float32)

    padded = np.zeros((window_num_samples,), dtype=np.float32)
    padded[-audio.shape[0] :] = audio
    return padded


def find_latest_completed_vad_segment(
    sync_data: dict | None,
    *,
    current_time_sec: float,
    kind: str | None = "instruction",
) -> dict | None:
    if not sync_data:
        return None
    raw_segments = sync_data.get("vad_segments")
    if not isinstance(raw_segments, list):
        return None

    latest_segment = None
    latest_end_sec = -float("inf")
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        if kind is not None and segment.get("kind") not in (None, kind):
            continue
        end_sec = _safe_float(segment.get("end_sec"))
        if end_sec is None or end_sec > current_time_sec:
            continue
        if end_sec >= latest_end_sec:
            latest_segment = segment
            latest_end_sec = end_sec
    return latest_segment


def block_frames_for_sample_rate(
    sample_rate: int, *, block_duration_s: float = 0.05
) -> int:
    return max(256, int(round(int(sample_rate) * float(block_duration_s))))


def pcm_array_to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim != 2:
        raise ValueError(f"Expected mono/stereo PCM array, got shape {audio.shape}")

    if audio.dtype.kind == "u":
        if audio.dtype.itemsize != 1:
            raise ValueError(f"Unsupported unsigned PCM dtype: {audio.dtype}")
        audio_f32 = (audio.astype(np.float32) - 128.0) / 128.0
    elif audio.dtype.kind == "i":
        max_val = float(np.iinfo(audio.dtype).max)
        audio_f32 = audio.astype(np.float32) / max_val
    elif audio.dtype.kind == "f":
        audio_f32 = audio.astype(np.float32)
    else:
        raise ValueError(f"Unsupported PCM dtype: {audio.dtype}")

    mono = np.mean(audio_f32, axis=1) if audio_f32.shape[1] > 1 else audio_f32[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32)


def pcm16_bytes_to_mono_float32(raw: bytes, *, channels: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError("channels must be > 0")

    bytes_per_frame = channels * np.dtype(np.int16).itemsize
    complete_bytes = len(raw) - (len(raw) % bytes_per_frame)
    if complete_bytes <= 0:
        return np.zeros((0,), dtype=np.float32)

    pcm = np.frombuffer(raw[:complete_bytes], dtype=np.int16)
    if pcm.size == 0:
        return np.zeros((0,), dtype=np.float32)

    return pcm_array_to_mono_float32(pcm.reshape(-1, channels))


def _looks_like_usb_input_device(device_name: object) -> bool:
    name = str(device_name).strip().lower()
    if not name:
        return False
    usb_markers = ("usb", "mic", "microphone", "headset", "composite")
    return any(marker in name for marker in usb_markers)


def resolve_preferred_input_device() -> int | str | None:
    import sounddevice as sd

    try:
        default_input, _ = sd.default.device
    except Exception:
        default_input = None

    try:
        devices = sd.query_devices()
    except Exception:
        return default_input if default_input not in (-1, None) else None

    valid_default = default_input if default_input not in (-1, None) else None
    if isinstance(valid_default, int):
        try:
            default_info = devices[valid_default]
            if int(
                default_info.get("max_input_channels", 0)
            ) > 0 and _looks_like_usb_input_device(default_info.get("name")):
                return valid_default
        except Exception:
            pass

    for index, device_info in enumerate(devices):
        try:
            if int(device_info.get("max_input_channels", 0)) <= 0:
                continue
        except Exception:
            continue
        if _looks_like_usb_input_device(device_info.get("name")):
            return index

    return valid_default


def resolve_supported_input_sample_rate(
    *,
    input_device: int | str | None,
    requested_sample_rate: int,
    channels: int,
    dtype_name: str,
) -> int:
    import sounddevice as sd

    requested_rate = int(requested_sample_rate)
    try:
        sd.check_input_settings(
            device=input_device,
            samplerate=requested_rate,
            channels=channels,
            dtype=dtype_name,
        )
        return requested_rate
    except Exception:
        pass

    try:
        device_info = sd.query_devices(input_device)
        fallback_rate = int(round(float(device_info["default_samplerate"])))
    except Exception:
        sd.check_input_settings(
            device=input_device,
            samplerate=requested_rate,
            channels=channels,
            dtype=dtype_name,
        )
        return requested_rate

    if fallback_rate <= 0 or fallback_rate == requested_rate:
        sd.check_input_settings(
            device=input_device,
            samplerate=requested_rate,
            channels=channels,
            dtype=dtype_name,
        )
        return requested_rate

    sd.check_input_settings(
        device=input_device,
        samplerate=fallback_rate,
        channels=channels,
        dtype=dtype_name,
    )
    print(
        "[MicrophoneRecorder] "
        f"Requested sample rate {requested_rate} Hz unsupported on input device "
        f"{input_device}; using {fallback_rate} Hz instead."
    )
    return fallback_rate


def _normalize_input_device(input_device: int | str | None) -> int | str | None:
    if isinstance(input_device, str):
        stripped = input_device.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return input_device


def _live_microphone_buffer_worker_main(
    conn: Connection,
    sample_rate: int,
    channels: int,
    dtype_name: str,
    input_device: int | str | None,
    window_num_samples: int,
    buffer_num_samples: int,
    block_duration_s: float,
) -> None:
    dtype = np.dtype(dtype_name)
    requested_sample_rate = int(sample_rate)
    resolved_sample_rate = int(sample_rate)
    resolved_input_device = _normalize_input_device(input_device)
    block_frames = block_frames_for_sample_rate(
        resolved_sample_rate, block_duration_s=block_duration_s
    )
    default_window_num_samples = int(window_num_samples)
    effective_buffer_num_samples = int(buffer_num_samples)

    stream = None
    is_recording = False
    sample_count = 0
    runtime_error: str | None = None
    status_lines: collections.deque[str] = collections.deque(maxlen=20)
    chunks: collections.deque[np.ndarray] = collections.deque()
    total_samples = 0

    def _summary() -> dict:
        return {
            "sample_rate": int(resolved_sample_rate),
            "input_device": resolved_input_device,
            "num_samples": int(sample_count),
        }

    def _close_stream() -> None:
        nonlocal stream
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        stream = None

    def _reset_buffer() -> None:
        nonlocal sample_count
        nonlocal runtime_error
        nonlocal total_samples
        chunks.clear()
        status_lines.clear()
        sample_count = 0
        total_samples = 0
        runtime_error = None

    def _append_status(message: str) -> None:
        if message:
            status_lines.append(message)

    def _append_chunk(chunk: np.ndarray) -> None:
        nonlocal total_samples
        nonlocal sample_count
        if chunk.size == 0:
            return
        chunks.append(chunk)
        total_samples += int(chunk.shape[0])
        sample_count += int(chunk.shape[0])
        while total_samples > effective_buffer_num_samples and chunks:
            overflow = total_samples - effective_buffer_num_samples
            head = chunks[0]
            if head.shape[0] <= overflow:
                chunks.popleft()
                total_samples -= int(head.shape[0])
                continue
            chunks[0] = head[overflow:]
            total_samples -= int(overflow)

    def _requested_window_num_samples(command: dict) -> int | None:
        requested_window_num_samples = command.get("num_samples")
        if requested_window_num_samples is None:
            return None
        requested_window_num_samples = int(requested_window_num_samples)
        if requested_window_num_samples <= 0:
            raise ValueError("window num_samples must be > 0")
        return requested_window_num_samples

    def _current_window(
        window_num_samples_override: int | None = None,
    ) -> tuple[np.ndarray, bool]:
        current_window_num_samples = (
            default_window_num_samples
            if window_num_samples_override is None
            else int(window_num_samples_override)
        )
        if current_window_num_samples <= 0:
            raise ValueError("window num_samples must be > 0")
        if not chunks:
            return np.zeros((current_window_num_samples,), dtype=np.float32), False
        if len(chunks) == 1:
            audio = np.array(chunks[0], copy=True)
        else:
            audio = np.concatenate(tuple(chunks), axis=0)
        return fit_audio_window(audio, current_window_num_samples), True

    def _window_response(window_num_samples_override: int | None = None) -> dict:
        audio, has_audio = _current_window(window_num_samples_override)
        payload = {
            "ok": runtime_error is None,
            "audio": audio,
            "has_audio": has_audio,
            **_summary(),
        }
        if runtime_error is not None:
            payload["error"] = runtime_error
        if status_lines:
            payload["status"] = list(status_lines)
        return payload

    def _start_recording() -> dict:
        nonlocal resolved_sample_rate
        nonlocal resolved_input_device
        nonlocal block_frames
        nonlocal default_window_num_samples
        nonlocal effective_buffer_num_samples
        nonlocal stream
        nonlocal is_recording

        import sounddevice as sd

        _reset_buffer()
        resolved_input_device = (
            _normalize_input_device(input_device)
            if input_device is not None
            else resolve_preferred_input_device()
        )
        resolved_sample_rate = resolve_supported_input_sample_rate(
            input_device=resolved_input_device,
            requested_sample_rate=requested_sample_rate,
            channels=channels,
            dtype_name=dtype.name,
        )
        default_window_num_samples = _audio_util.window_num_samples_for_duration(
            window_num_samples,
            sample_rate=resolved_sample_rate,
            target_sample_rate=requested_sample_rate,
        )
        effective_buffer_num_samples = max(
            default_window_num_samples,
            _audio_util.window_num_samples_for_duration(
                buffer_num_samples,
                sample_rate=resolved_sample_rate,
                target_sample_rate=requested_sample_rate,
            ),
        )
        block_frames = block_frames_for_sample_rate(
            resolved_sample_rate, block_duration_s=block_duration_s
        )
        stream_obj = sd.InputStream(
            samplerate=resolved_sample_rate,
            channels=channels,
            dtype=dtype.name,
            device=resolved_input_device,
            blocksize=block_frames,
            latency="high",
        )
        stream_obj.start()
        stream = stream_obj
        is_recording = True
        return {"ok": True, **_summary()}

    try:
        while True:
            if is_recording:
                try:
                    assert stream is not None
                    chunk, overflowed = stream.read(block_frames)
                    if chunk.size > 0:
                        chunk_arr = np.asarray(chunk, dtype=dtype)
                        _append_chunk(pcm_array_to_mono_float32(chunk_arr))
                        if overflowed:
                            _append_status(f"overflow@{time.monotonic_ns()}")
                except Exception as exc:
                    runtime_error = str(exc)
                    is_recording = False
                    _close_stream()

                while conn.poll():
                    command = conn.recv()
                    action = command.get("cmd")
                    if action == "window":
                        try:
                            conn.send(
                                _window_response(_requested_window_num_samples(command))
                            )
                        except Exception as exc:
                            conn.send({"ok": False, "error": str(exc), **_summary()})
                    elif action == "reset":
                        is_recording = False
                        _close_stream()
                        _reset_buffer()
                        conn.send({"ok": True, **_summary()})
                    elif action == "shutdown":
                        is_recording = False
                        _close_stream()
                        conn.send({"ok": True})
                        return
                    elif action == "start":
                        conn.send(
                            {
                                "ok": False,
                                "error": "Recording is already active",
                                **_summary(),
                            }
                        )
                    else:
                        conn.send(
                            {
                                "ok": False,
                                "error": f"Unknown command: {action}",
                                **_summary(),
                            }
                        )
                continue

            try:
                command = conn.recv()
            except EOFError:
                _close_stream()
                return

            action = command.get("cmd")
            if action == "start":
                try:
                    conn.send(_start_recording())
                except Exception as exc:
                    runtime_error = str(exc)
                    conn.send({"ok": False, "error": runtime_error, **_summary()})
            elif action == "window":
                try:
                    conn.send(_window_response(_requested_window_num_samples(command)))
                except Exception as exc:
                    conn.send({"ok": False, "error": str(exc), **_summary()})
            elif action == "reset":
                _reset_buffer()
                conn.send({"ok": True, **_summary()})
            elif action == "shutdown":
                _close_stream()
                conn.send({"ok": True})
                return
            else:
                conn.send(
                    {"ok": False, "error": f"Unknown command: {action}", **_summary()}
                )
    finally:
        _close_stream()


class MultiprocessAudioBuffer:
    """Process-backed live microphone buffer aligned with the collection-side recorder design."""

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        window_num_samples: int,
        buffer_num_samples: int,
        block_duration_s: float,
        device: int | str | None = None,
        dtype: np.dtype | type[np.generic] = np.int16,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if window_num_samples <= 0:
            raise ValueError("window_num_samples must be > 0")
        if buffer_num_samples < window_num_samples:
            raise ValueError("buffer_num_samples must be >= window_num_samples")
        if block_duration_s <= 0.0:
            raise ValueError("block_duration_s must be > 0")

        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._dtype = np.dtype(dtype)
        if self._dtype.kind not in {"i", "u"}:
            raise ValueError(
                "Only integer PCM dtypes are supported for live microphone capture"
            )
        self._window_num_samples = int(window_num_samples)
        self._buffer_num_samples = int(buffer_num_samples)
        self._block_duration_s = float(block_duration_s)
        self._device = _normalize_input_device(device)

        self._ctx = mp.get_context("spawn")
        self._proc: mp.Process | None = None
        self._parent_conn: Connection | None = None
        self._child_conn: Connection | None = None
        self._is_recording = False
        self._spawn_worker()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def input_device(self) -> int | str | None:
        return self._device

    def _spawn_worker(self) -> None:
        self._parent_conn, self._child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_live_microphone_buffer_worker_main,
            args=(
                self._child_conn,
                self._sample_rate,
                self._channels,
                self._dtype.name,
                self._device,
                self._window_num_samples,
                self._buffer_num_samples,
                self._block_duration_s,
            ),
            daemon=True,
        )

    def _ensure_worker_started(self) -> None:
        if self._proc is None or (
            self._proc.exitcode is not None and not self._proc.is_alive()
        ):
            self._spawn_worker()
        if self._proc is None:
            raise RuntimeError("Microphone worker process is not initialized")
        if not self._proc.is_alive():
            self._proc.start()
            if self._child_conn is not None:
                self._child_conn.close()
                self._child_conn = None

    def _request(self, command: dict, *, timeout_s: float = 5.0) -> dict:
        self._ensure_worker_started()
        if self._parent_conn is None:
            raise RuntimeError("Microphone worker connection is not available")
        try:
            self._parent_conn.send(command)
            if not self._parent_conn.poll(timeout_s):
                raise RuntimeError(
                    f"Microphone worker did not respond to {command.get('cmd')}"
                )
            response = self._parent_conn.recv()
        except Exception:
            self.stop()
            raise

        sample_rate = response.get("sample_rate")
        if sample_rate is not None:
            self._sample_rate = int(sample_rate)
        if "input_device" in response:
            self._device = response.get("input_device")
        return response

    def start(self) -> None:
        response = self._request({"cmd": "start"}, timeout_s=10.0)
        if not response.get("ok", False):
            raise RuntimeError(
                str(response.get("error", "Failed to start microphone recording"))
            )
        self._is_recording = True

    def stop(self) -> None:
        proc = self._proc
        conn = self._parent_conn
        if proc is not None and proc.is_alive() and conn is not None:
            try:
                conn.send({"cmd": "shutdown"})
                if conn.poll(2.0):
                    conn.recv()
            except Exception:
                pass
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        if self._parent_conn is not None:
            with contextlib.suppress(Exception):
                self._parent_conn.close()
            self._parent_conn = None
        if self._child_conn is not None:
            with contextlib.suppress(Exception):
                self._child_conn.close()
            self._child_conn = None
        self._proc = None
        self._is_recording = False

    def wait_until_ready(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while time.monotonic() < deadline:
            _, has_audio = self.read_window()
            if has_audio:
                return True
            time.sleep(0.02)
        _, has_audio = self.read_window()
        return has_audio

    def read_window(
        self, window_num_samples: int | None = None
    ) -> tuple[np.ndarray, bool]:
        if not self._is_recording:
            raise RuntimeError("Microphone recording has not been started.")
        current_window_num_samples = (
            self._window_num_samples
            if window_num_samples is None
            else int(window_num_samples)
        )
        if current_window_num_samples <= 0:
            raise ValueError("window_num_samples must be > 0")
        response = self._request(
            {"cmd": "window", "num_samples": current_window_num_samples}, timeout_s=5.0
        )
        if not response.get("ok", False):
            raise RuntimeError(
                str(response.get("error", "Failed to read microphone window"))
            )
        raw_audio = response.get("audio")
        audio = (
            np.zeros((current_window_num_samples,), dtype=np.float32)
            if raw_audio is None
            else np.asarray(raw_audio, dtype=np.float32)
        )
        return fit_audio_window(audio, current_window_num_samples), bool(
            response.get("has_audio", False)
        )

    def read_window_for_duration(
        self,
        *,
        target_num_samples: int,
        target_sample_rate: int,
    ) -> tuple[np.ndarray, bool]:
        # 麦的采样率假如更高的话就录更多的样本，动态的调整需要的样本的数量，比如说麦的采样率16k->48k了，那么这里就根据这个函数把16000的样本数量到48000的样本数量了
        current_window_num_samples = _audio_util.window_num_samples_for_duration(
            target_num_samples,
            sample_rate=self._sample_rate,
            target_sample_rate=target_sample_rate,
        )
        return self.read_window(window_num_samples=current_window_num_samples)


def extract_episode_audio_window(
    *,
    dataset_root: str | Path,
    episode_index: int,
    frame_index: int,
    window_num_samples: int,
    target_sample_rate: int,
    fallback_fps: float,
) -> tuple[np.ndarray, bool]:
    default_audio_path, sync_path = resolve_episode_audio_paths(
        dataset_root, episode_index
    )
    sync_data = load_json(sync_path)
    audio_path = (
        resolve_audio_path_from_sync(dataset_root, sync_data) or default_audio_path
    )
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sample_rate = read_wav_mono(audio_path)
    fps = (
        float(sync_data.get("control_frequency", fallback_fps))
        if sync_data
        else float(fallback_fps)
    )
    _, end_sample = frame_range_to_audio_samples(
        sync_data=sync_data,
        start_frame_index=frame_index,
        end_frame_index=frame_index + 1,
        sample_rate=sample_rate,
        audio_total_samples=int(audio.shape[0]),
        fps=fps,
    )

    window_orig_samples = _audio_util.window_num_samples_for_duration(
        window_num_samples,
        sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
    )
    start_sample = max(0, end_sample - window_orig_samples)
    segment = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
    if sample_rate != target_sample_rate:
        segment = resample_audio(segment, sample_rate, target_sample_rate)

    has_audio = segment.size > 0
    return fit_audio_window(segment, window_num_samples), has_audio


def extract_episode_instruction_audio(
    *,
    dataset_root: str | Path,
    episode_index: int,
    frame_index: int,
    window_num_samples: int,
    target_sample_rate: int,
    fallback_fps: float,
) -> tuple[np.ndarray, bool]:
    default_audio_path, sync_path = resolve_episode_audio_paths(
        dataset_root, episode_index
    )
    sync_data = load_json(sync_path)
    if not sync_data:
        return np.zeros((window_num_samples,), dtype=np.float32), False

    audio_path = (
        resolve_audio_path_from_sync(dataset_root, sync_data) or default_audio_path
    )
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    full_audio, sample_rate = read_wav_mono(audio_path)
    fps = (
        float(sync_data.get("control_frequency", fallback_fps))
        if sync_data
        else float(fallback_fps)
    )
    _, current_end_sample = frame_range_to_audio_samples(
        sync_data=sync_data,
        start_frame_index=frame_index,
        end_frame_index=frame_index + 1,
        sample_rate=sample_rate,
        audio_total_samples=int(full_audio.shape[0]),
        fps=fps,
    )
    current_time_sec = float(current_end_sample) / float(sample_rate)
    segment = find_latest_completed_vad_segment(
        sync_data,
        current_time_sec=current_time_sec,
        kind="instruction",
    )
    if segment is None:
        return np.zeros((window_num_samples,), dtype=np.float32), False

    segment_audio = None
    segment_sample_rate = sample_rate
    segment_path = resolve_dataset_path(dataset_root, segment.get("audio_wav_path"))
    if segment_path is not None and segment_path.is_file():
        segment_audio, segment_sample_rate = read_wav_mono(segment_path)
    else:
        start_sample = _safe_int(segment.get("start_sample"))
        end_sample = _safe_int(segment.get("end_sample"))
        if (
            start_sample is not None
            and end_sample is not None
            and 0 <= start_sample < end_sample <= full_audio.shape[0]
        ):
            segment_audio = np.ascontiguousarray(
                full_audio[start_sample:end_sample],
                dtype=np.float32,
            )

    if segment_audio is None or segment_audio.size == 0:
        return np.zeros((window_num_samples,), dtype=np.float32), False

    if segment_sample_rate != target_sample_rate:
        segment_audio = resample_audio(
            segment_audio,
            segment_sample_rate,
            target_sample_rate,
        )

    return fit_audio_window(segment_audio, window_num_samples), True

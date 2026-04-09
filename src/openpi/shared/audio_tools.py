from __future__ import annotations

import functools
import json
from pathlib import Path
import wave

import numpy as np

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

    feature_extractor, model, resolved_device = _load_whisper_components(model_name, device)

    import torch

    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(resolved_device)
    if resolved_device.startswith("cuda"):
        input_features = input_features.to(dtype=torch.float16)

    with torch.inference_mode():
        hidden = model.encoder(input_features=input_features).last_hidden_state

    if pooling == "mean":
        pooled = hidden.mean(dim=1)
    elif pooling == "first":
        pooled = hidden[:, 0, :]
    else:
        raise ValueError(f"Unsupported Whisper pooling '{pooling}'")

    return pooled[0].to(dtype=torch.float32, device="cpu").numpy()


def episode_audio_stem(episode_index: int) -> str:
    return f"episode_{int(episode_index):06d}"


def resolve_episode_audio_paths(dataset_root: str | Path, episode_index: int) -> tuple[Path, Path]:
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


def resolve_audio_path_from_sync(dataset_root: str | Path, sync_data: dict | None) -> Path | None:
    if not sync_data:
        return None
    raw_audio_path = sync_data.get("audio_path")
    if not raw_audio_path:
        return None

    audio_path = Path(str(raw_audio_path))
    if audio_path.is_absolute():
        return audio_path
    return Path(dataset_root) / audio_path


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
    fallback_start = int(np.clip(round((start_frame_index / fps_safe) * sample_rate), 0, audio_total_samples))
    fallback_end = int(np.clip(round((end_frame_index / fps_safe) * sample_rate), 0, audio_total_samples))

    if not sync_data:
        return fallback_start, max(fallback_start, fallback_end)

    audio_start_ns = _safe_int(sync_data.get("audio_start_monotonic_ns"))
    frame_records = sync_data.get("frame_records")
    if audio_start_ns is None or not isinstance(frame_records, list) or not frame_records:
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

    start_sample = int(np.clip(round(((start_ns - audio_start_ns) / 1e9) * sample_rate), 0, audio_total_samples))
    end_sample = int(np.clip(round(((end_ns - audio_start_ns) / 1e9) * sample_rate), 0, audio_total_samples))
    return start_sample, max(start_sample, end_sample)


def resample_audio(audio: np.ndarray, orig_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")
    if orig_sample_rate == target_sample_rate or audio.size == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)

    target_len = max(round(audio.shape[0] * target_sample_rate / orig_sample_rate), 1)
    source_positions = np.arange(audio.shape[0], dtype=np.float32)
    target_positions = np.linspace(0, audio.shape[0] - 1, num=target_len, dtype=np.float32)
    return np.ascontiguousarray(np.interp(target_positions, source_positions, audio).astype(np.float32))


def fit_audio_window(audio: np.ndarray, window_num_samples: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.shape[0] >= window_num_samples:
        return np.ascontiguousarray(audio[-window_num_samples:], dtype=np.float32)

    padded = np.zeros((window_num_samples,), dtype=np.float32)
    padded[-audio.shape[0] :] = audio
    return padded


def extract_episode_audio_window(
    *,
    dataset_root: str | Path,
    episode_index: int,
    frame_index: int,
    window_num_samples: int,
    target_sample_rate: int,
    fallback_fps: float,
) -> tuple[np.ndarray, bool]:
    default_audio_path, sync_path = resolve_episode_audio_paths(dataset_root, episode_index)
    sync_data = load_json(sync_path)
    audio_path = resolve_audio_path_from_sync(dataset_root, sync_data) or default_audio_path
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sample_rate = read_wav_mono(audio_path)
    fps = float(sync_data.get("control_frequency", fallback_fps)) if sync_data else float(fallback_fps)
    _, end_sample = frame_range_to_audio_samples(
        sync_data=sync_data,
        start_frame_index=frame_index,
        end_frame_index=frame_index + 1,
        sample_rate=sample_rate,
        audio_total_samples=int(audio.shape[0]),
        fps=fps,
    )

    duration_s = window_num_samples / float(target_sample_rate)
    window_orig_samples = max(round(duration_s * sample_rate), 1)
    start_sample = max(0, end_sample - window_orig_samples)
    segment = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
    if sample_rate != target_sample_rate:
        segment = resample_audio(segment, sample_rate, target_sample_rate)

    has_audio = segment.size > 0
    return fit_audio_window(segment, window_num_samples), has_audio

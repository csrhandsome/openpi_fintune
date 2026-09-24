"""
Examples:
  # Check local model status; does not run ASR.
  uv run python scripts/api/asr_api.py info

  # Save one randomly corrupted audio file; does not run ASR.
  uv run python scripts/api/asr_api.py degrade --audio benchmark/data/libero_audio/direct/libero_object/libero_object_16.wav --output /tmp/asr_failed.wav --seed 0

  # Run ASR on randomly corrupted audio; this is the wrong-transcript path.
  uv run python scripts/api/asr_api.py transcribe --audio benchmark/data/libero_audio/direct/libero_object/libero_object_16.wav --language English --dump-degraded /tmp/asr_baseline_input.wav --seed 0
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

try:
    from audio_util import AudioDegradationConfig
    from audio_util import degrade_audio_for_asr_baseline
except ImportError:
    from scripts.api.audio_util import AudioDegradationConfig
    from scripts.api.audio_util import degrade_audio_for_asr_baseline

DEFAULT_MODEL_REPO = "Qwen/Qwen3-ASR-0.6B"
KNOWN_MODEL_REPOS = ("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B")
TARGET_SAMPLE_RATE = 16_000


@dataclass
class Qwen3ASRApi:
    """Small deployment and inference wrapper for one local Qwen3-ASR model."""

    model_name: str = DEFAULT_MODEL_REPO
    model_root: str | Path | None = None
    device_map: str = "cuda:0"
    dtype: str | None = "bfloat16"
    attn_implementation: str | None = "auto"
    local_files_only: bool = True
    max_inference_batch_size: int = 4
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.qwen_asr_dir = self.project_root / "third_party" / "Qwen3-ASR"
        default_model_root = self.qwen_asr_dir / "models"
        self.model_root = Path(self.model_root or default_model_root).expanduser().resolve()
        self._model: Any | None = None

    def repo_name(self, model_name: str | None = None) -> str:
        return (model_name or self.model_name).rstrip("/").split("/")[-1]

    def model_dir(self, model_name: str | None = None) -> Path:
        candidate = Path(model_name or self.model_name).expanduser()
        if candidate.exists():
            return candidate.resolve()
        return self.model_root / self.repo_name(model_name)

    def _cached_snapshot_dir(self, model_name: str) -> Path | None:
        try:
            from transformers.utils.hub import cached_file

            config_path = cached_file(model_name, "config.json", local_files_only=True)
        except Exception:
            return None
        if config_path is None:
            return None
        return Path(config_path).parent

    def available_models(self) -> list[dict[str, str]]:
        models: list[dict[str, str]] = []
        for model_name in KNOWN_MODEL_REPOS:
            local_dir = self.model_dir(model_name)
            if (local_dir / "config.json").exists():
                models.append({"repo": model_name, "source": "model_root", "path": str(local_dir)})
                continue
            cached_dir = self._cached_snapshot_dir(model_name)
            if cached_dir is not None:
                models.append({"repo": model_name, "source": "huggingface_cache", "path": str(cached_dir)})
        return models

    def is_downloaded(self) -> bool:
        return self._resolve_model_path(allow_repo_fallback=False) is not None

    def download_model(self, *, force: bool = False) -> Path:
        """Download the configured checkpoint with the ModelScope CLI."""

        local_dir = self.model_dir()
        if (local_dir / "config.json").exists() and not force:
            return local_dir

        modelscope = shutil.which("modelscope")
        if modelscope is None:
            raise RuntimeError(
                "ModelScope CLI is not available. Install it in the active uv environment with "
                "`uv sync` after keeping `modelscope` in pyproject dependencies."
            )

        local_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [modelscope, "download", "--model", self.model_name, "--local_dir", str(local_dir)]
        subprocess.run(cmd, check=True)

        if not (local_dir / "config.json").exists():
            raise RuntimeError(f"ModelScope download finished but config.json was not found under {local_dir}.")
        return local_dir

    def _resolve_model_path(self, *, allow_repo_fallback: bool) -> Path | str | None:
        requested = Path(self.model_name).expanduser()
        if requested.exists():
            requested = requested.resolve()
            if (requested / "config.json").exists():
                return requested
            return None

        local_dir = self.model_dir()
        if (local_dir / "config.json").exists():
            return local_dir

        cached_dir = self._cached_snapshot_dir(self.model_name)
        if cached_dir is not None:
            return cached_dir

        if self.model_name == DEFAULT_MODEL_REPO:
            for fallback in KNOWN_MODEL_REPOS:
                cached_dir = self._cached_snapshot_dir(fallback)
                if cached_dir is not None:
                    self.model_name = fallback
                    return cached_dir

        if allow_repo_fallback and not self.local_files_only:
            return self.model_name
        return None

    def ensure_model(self, *, download: bool = False) -> Path | str:
        if download:
            return self.download_model()

        model_path = self._resolve_model_path(allow_repo_fallback=True)
        if model_path is not None:
            return model_path

        raise FileNotFoundError(
            f"Qwen3-ASR weights are not ready for {self.model_name!r}. "
            "Run: uv run python scripts/api/asr_api.py info to inspect local cache, or "
            "uv run python scripts/api/asr_api.py download"
        )

    def _import_runtime(self) -> tuple[Any, Any, Any, int, list[str]]:
        if str(self.qwen_asr_dir) not in sys.path:
            sys.path.insert(0, str(self.qwen_asr_dir))
        from qwen_asr import Qwen3ASRModel
        from qwen_asr.inference.utils import SAMPLE_RATE
        from qwen_asr.inference.utils import SUPPORTED_LANGUAGES
        from qwen_asr.inference.utils import normalize_audio_input
        import torch

        return torch, Qwen3ASRModel, normalize_audio_input, int(SAMPLE_RATE), list(SUPPORTED_LANGUAGES)

    def _resolve_dtype(self, torch: Any) -> Any | None:
        if self.dtype is None:
            return None
        dtype = self.dtype.lower()
        if dtype in {"", "none"}:
            return None
        if dtype == "auto":
            return "auto"
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype {self.dtype!r}. Use bfloat16, float16, float32, auto, or none.")
        return mapping[dtype]

    def _resolve_attn_implementation(self) -> str | None:
        if self.attn_implementation is None:
            return None
        value = self.attn_implementation.lower()
        if value in {"", "none"}:
            return None
        if value != "auto":
            return self.attn_implementation
        if importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        return "sdpa"

    def load(self, *, download: bool = False) -> Any:
        """Load the local Qwen3-ASR model, downloading first when requested."""

        if self._model is not None:
            return self._model

        model_path = self.ensure_model(download=download)
        torch, qwen_model_cls, _, _, _ = self._import_runtime()

        load_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
        }
        dtype = self._resolve_dtype(torch)
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        attn_implementation = self._resolve_attn_implementation()
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model_path, Path):
            load_kwargs["local_files_only"] = self.local_files_only

        self._model = qwen_model_cls.from_pretrained(str(model_path), **load_kwargs)
        return self._model

    def supported_languages(self) -> list[str]:
        _, _, _, _, languages = self._import_runtime()
        return languages

    def _normalize_audio(self, audio: Any) -> np.ndarray:
        _, _, normalize_audio_input, sample_rate, _ = self._import_runtime()
        if sample_rate != TARGET_SAMPLE_RATE:
            raise RuntimeError(f"Expected Qwen3-ASR sample rate {TARGET_SAMPLE_RATE}, got {sample_rate}")
        if isinstance(audio, Path):
            audio = str(audio)
        return np.asarray(normalize_audio_input(audio), dtype=np.float32).reshape(-1)

    def prepare_audios(
        self,
        audio: Any | list[Any],
        *,
        degradation: AudioDegradationConfig | None = None,
        seed: int | None = None,
    ) -> tuple[list[tuple[np.ndarray, int]], list[dict[str, Any]]]:
        items = audio if isinstance(audio, list) else [audio]
        rng = np.random.default_rng(seed)
        prepared: list[tuple[np.ndarray, int]] = []
        reports: list[dict[str, Any]] = []
        degradation = degradation or AudioDegradationConfig()

        for item in items:
            wav = self._normalize_audio(item)
            degraded_wav, report = degrade_audio_for_asr_baseline(
                wav,
                sample_rate=TARGET_SAMPLE_RATE,
                config=degradation,
                rng=rng,
            )
            prepared.append((degraded_wav, TARGET_SAMPLE_RATE))
            reports.append(asdict(report))

        return prepared, reports

    def transcribe_prepared(
        self,
        prepared_audio: list[tuple[np.ndarray, int]],
        *,
        context: str | list[str] = "",
        language: str | list[str] | None = None,
        return_time_stamps: bool = False,
        download: bool = False,
    ) -> list[Any]:
        model = self.load(download=download)
        return model.transcribe(
            audio=prepared_audio,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )

    def transcribe(
        self,
        audio: Any | list[Any],
        *,
        context: str | list[str] = "",
        language: str | list[str] | None = None,
        return_time_stamps: bool = False,
        degradation: AudioDegradationConfig | None = None,
        seed: int | None = None,
        download: bool = False,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        prepared, reports = self.prepare_audios(audio, degradation=degradation, seed=seed)
        results = self.transcribe_prepared(
            prepared,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
            download=download,
        )
        return results, reports

    @staticmethod
    def save_wavs(wavs: list[tuple[np.ndarray, int]], output: str | Path) -> list[Path]:
        import soundfile as sf

        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(wavs) == 1:
            wav, sr = wavs[0]
            sf.write(output_path, wav, sr)
            return [output_path]

        suffix = output_path.suffix or ".wav"
        stem = output_path.with_suffix("")
        paths: list[Path] = []
        for index, (wav, sr) in enumerate(wavs):
            path = stem.parent / f"{stem.name}_{index:03d}{suffix}"
            sf.write(path, wav, sr)
            paths.append(path)
        return paths


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL_REPO, help="Qwen3-ASR checkpoint repo id or local directory.")
    parser.add_argument("--model-root", default=None, help="Directory that stores ModelScope-downloaded weights.")
    parser.add_argument("--device-map", default="cuda:0", help="Device map passed to Qwen3ASRModel.from_pretrained.")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16, float16, float32, auto, or none.")
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help="auto chooses flash_attention_2 when installed, otherwise sdpa.",
    )
    parser.add_argument(
        "--local-files-only",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use local model files only. Disable only when an online download is intended.",
    )
    parser.add_argument("--max-inference-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)


def _add_degradation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-degrade", action="store_true", help="Disable ASR-baseline audio corruption.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible random corruption sampling.")
    parser.add_argument(
        "--degrade-method",
        choices=("random", "vad_trim", "repeat_speech"),
        default="random",
        help="random selects one verified-failing error type: VAD trim or repeated speech.",
    )
    parser.add_argument("--trim-ms", type=int, default=None, help="Override both-side crop for vad_trim; otherwise sample a verified failing case.")
    parser.add_argument("--trim-start-ms", type=int, default=None, help="Override start crop for vad_trim.")
    parser.add_argument("--trim-end-ms", type=int, default=None, help="Override end crop for vad_trim.")
    parser.add_argument("--repeat-mode", choices=("tail", "twice"), default=None, help="Override repeat mode for repeat_speech.")
    parser.add_argument("--repeat-start-ms", type=int, default=None, help="Override repeat start offset for tail repeat.")
    parser.add_argument("--repeat-gap-ms", type=int, default=None, help="Override silence gap before repeated audio.")


def _api_from_args(args: argparse.Namespace) -> Qwen3ASRApi:
    return Qwen3ASRApi(
        model_name=args.model,
        model_root=args.model_root,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )


def _degradation_from_args(args: argparse.Namespace) -> AudioDegradationConfig:
    trim_start_ms = args.trim_start_ms if args.trim_start_ms is not None else args.trim_ms
    trim_end_ms = args.trim_end_ms if args.trim_end_ms is not None else args.trim_ms
    return AudioDegradationConfig(
        enabled=not args.no_degrade,
        method=args.degrade_method,
        trim_start_ms=trim_start_ms,
        trim_end_ms=trim_end_ms,
        repeat_mode=args.repeat_mode,
        repeat_start_after_ms=args.repeat_start_ms,
        repeat_gap_ms=args.repeat_gap_ms,
    )


def _result_to_dict(result: Any, degradation: dict[str, Any]) -> dict[str, Any]:
    item = {
        "language": getattr(result, "language", ""),
        "text": getattr(result, "text", ""),
        "degradation": degradation,
    }
    time_stamps = getattr(result, "time_stamps", None)
    if time_stamps is not None:
        item["time_stamps"] = str(time_stamps)
    return item


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy and call local Qwen3-ASR with simple ASR-failure audio variants.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download the configured checkpoint with ModelScope.")
    _add_common_args(download_parser)
    download_parser.add_argument("--force", action="store_true", help="Run ModelScope download even if files exist.")

    info_parser = subparsers.add_parser("info", help="Print local model availability and supported languages.")
    _add_common_args(info_parser)
    info_parser.add_argument("--load", action="store_true", help="Also load the ASR model.")

    degrade_parser = subparsers.add_parser("degrade", help="Write the corrupted ASR baseline audio without ASR.")
    _add_common_args(degrade_parser)
    _add_degradation_args(degrade_parser)
    degrade_parser.add_argument("--audio", required=True)
    degrade_parser.add_argument("--output", required=True)

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio after VAD-trim or repeated-speech corruption.")
    _add_common_args(transcribe_parser)
    _add_degradation_args(transcribe_parser)
    transcribe_parser.add_argument("--audio", required=True)
    transcribe_parser.add_argument("--language", default=None, help="Optional forced language, e.g. Chinese or English.")
    transcribe_parser.add_argument("--context", default="")
    transcribe_parser.add_argument("--download", action="store_true")
    transcribe_parser.add_argument("--return-time-stamps", action="store_true")
    transcribe_parser.add_argument("--dump-degraded", default=None, help="Optional path for the corrupted audio.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        api = _api_from_args(args)
        model_path = api.download_model(force=args.force)
        print(json.dumps({"model": str(model_path), "repo": api.model_name}, ensure_ascii=False, indent=2))
        return

    if args.command == "info":
        api = _api_from_args(args)
        resolved = api.ensure_model() if api.is_downloaded() else None
        if args.load:
            api.load()
        print(
            json.dumps(
                {
                    "configured_model": api.model_name,
                    "resolved_model": str(resolved) if resolved is not None else None,
                    "available_models": api.available_models(),
                    "supported_languages": api.supported_languages(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "degrade":
        api = _api_from_args(args)
        prepared, reports = api.prepare_audios(args.audio, degradation=_degradation_from_args(args), seed=args.seed)
        paths = api.save_wavs(prepared, args.output)
        print(
            json.dumps(
                {"outputs": [str(path) for path in paths], "degradation": reports},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "transcribe":
        api = _api_from_args(args)
        prepared, reports = api.prepare_audios(args.audio, degradation=_degradation_from_args(args), seed=args.seed)
        results = api.transcribe_prepared(
            prepared,
            context=args.context,
            language=args.language,
            return_time_stamps=args.return_time_stamps,
            download=args.download,
        )

        degraded_paths: list[str] = []
        if args.dump_degraded:
            degraded_paths = [str(path) for path in api.save_wavs(prepared, args.dump_degraded)]
        print(
            json.dumps(
                {
                    "model": str(api.ensure_model(download=args.download)),
                    "degraded_audio": degraded_paths,
                    "results": [_result_to_dict(result, report) for result, report in zip(results, reports, strict=True)],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

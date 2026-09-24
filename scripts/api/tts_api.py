"""
uv run python scripts/api/tts_api.py download
uv run python scripts/api/tts_api.py speak --text "Hello, this is a local Qwen TTS test." --language English --speaker Ryan --output /tmp/out.wav
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

DEFAULT_MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


@dataclass
class Qwen3TTSApi:
    """Small deployment and inference wrapper for one local Qwen3-TTS model.

    This deploys the lightweight CustomVoice checkpoint from the upstream README:
    ``Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice``. We download it with ModelScope
    and load it locally through ``Qwen3TTSModel.from_pretrained``.
    """

    model_root: str | Path | None = None
    device_map: str = "cuda:0"
    dtype: str | None = "bfloat16"
    attn_implementation: str | None = "auto"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.qwen_tts_dir = self.project_root / "third_party" / "Qwen3-TTS"
        default_model_root = self.qwen_tts_dir / "models"
        self.model_root = Path(self.model_root or default_model_root).expanduser().resolve()
        self._model: Any | None = None

    @staticmethod
    def repo_name() -> str:
        return DEFAULT_MODEL_REPO.rstrip("/").split("/")[-1]

    def model_dir(self) -> Path:
        return self.model_root / self.repo_name()

    def is_downloaded(self) -> bool:
        model_dir = self.model_dir()
        return (model_dir / "config.json").exists() and (model_dir / "speech_tokenizer" / "config.json").exists()

    def download_model(self, *, force: bool = False) -> Path:
        """Download the configured checkpoint with the ModelScope CLI."""

        local_dir = self.model_dir()
        if self.is_downloaded() and not force:
            return local_dir

        modelscope = shutil.which("modelscope")
        if modelscope is None:
            raise RuntimeError(
                "ModelScope CLI is not available. Install it in the active uv environment with "
                "`uv sync` after keeping `modelscope` in pyproject dependencies."
            )

        local_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [modelscope, "download", "--model", DEFAULT_MODEL_REPO, "--local_dir", str(local_dir)]
        subprocess.run(cmd, check=True)

        if not self.is_downloaded():
            raise RuntimeError(
                f"ModelScope download finished but expected files were not found under {local_dir}. "
                "The local directory should contain config.json and speech_tokenizer/config.json."
            )
        return local_dir

    def ensure_model(self, *, download: bool = False) -> Path:
        if download:
            return self.download_model()
        local_dir = self.model_dir()
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"Qwen3-TTS weights are not ready at {local_dir}. Run: uv run python scripts/api/tts_api.py download"
            )
        return local_dir

    def _import_runtime(self) -> tuple[Any, Any]:
        if str(self.qwen_tts_dir) not in sys.path:
            sys.path.insert(0, str(self.qwen_tts_dir))
        import torch
        from qwen_tts import Qwen3TTSModel

        return torch, Qwen3TTSModel

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
        """Load the local CustomVoice model, downloading first when requested."""

        if self._model is not None:
            return self._model

        model_dir = self.ensure_model(download=download)
        torch, qwen_model_cls = self._import_runtime()

        load_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "local_files_only": self.local_files_only,
        }
        dtype = self._resolve_dtype(torch)
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        attn_implementation = self._resolve_attn_implementation()
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        self._model = qwen_model_cls.from_pretrained(str(model_dir), **load_kwargs)
        return self._model

    def supported_speakers(self) -> list[str] | None:
        return self.load().get_supported_speakers()

    def supported_languages(self) -> list[str] | None:
        return self.load().get_supported_languages()

    def generate(
        self,
        text: str | list[str],
        *,
        language: str | list[str] = "Auto",
        speaker: str | list[str] = "Vivian",
        instruct: str | list[str] | None = None,
        download: bool = False,
        **generate_kwargs: Any,
    ) -> tuple[list[Any], int]:
        """Generate speech with the deployed CustomVoice model."""

        model = self.load(download=download)
        return model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            **generate_kwargs,
        )

    @staticmethod
    def save_wavs(wavs: list[Any], sample_rate: int, output: str | Path) -> list[Path]:
        import soundfile as sf

        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(wavs) == 1:
            sf.write(output_path, wavs[0], sample_rate)
            return [output_path]

        suffix = output_path.suffix or ".wav"
        stem = output_path.with_suffix("")
        paths: list[Path] = []
        for index, wav in enumerate(wavs):
            path = stem.parent / f"{stem.name}_{index:03d}{suffix}"
            sf.write(path, wav, sample_rate)
            paths.append(path)
        return paths


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-root", default=None, help="Directory that stores ModelScope-downloaded weights.")
    parser.add_argument("--device-map", default="cuda:0", help="Device map passed to Qwen3TTSModel.from_pretrained.")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16, float16, float32, auto, or none.")
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help="auto chooses flash_attention_2 when installed, otherwise sdpa.",
    )


def _api_from_args(args: argparse.Namespace) -> Qwen3TTSApi:
    return Qwen3TTSApi(
        model_root=args.model_root,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )


def _generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if getattr(args, "max_new_tokens", None) is not None:
        kwargs["max_new_tokens"] = args.max_new_tokens
    return kwargs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy and call local Qwen3-TTS CustomVoice with ModelScope weights.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download", help="Download the single configured checkpoint with ModelScope."
    )
    _add_common_args(download_parser)
    download_parser.add_argument("--force", action="store_true", help="Run ModelScope download even if files exist.")

    speak_parser = subparsers.add_parser("speak", help="Generate speech with the deployed CustomVoice model.")
    _add_common_args(speak_parser)
    speak_parser.add_argument("--text", required=True)
    speak_parser.add_argument("--output", required=True)
    speak_parser.add_argument("--language", default="Auto")
    speak_parser.add_argument("--speaker", default="Vivian")
    speak_parser.add_argument("--instruct", default="")
    speak_parser.add_argument("--download", action="store_true")
    speak_parser.add_argument("--max-new-tokens", type=int, default=None)

    info_parser = subparsers.add_parser("info", help="Print supported languages and speakers for the local model.")
    _add_common_args(info_parser)
    info_parser.add_argument("--download", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        api = _api_from_args(args)
        model_path = api.download_model(force=args.force)
        print(json.dumps({"model": str(model_path), "repo": DEFAULT_MODEL_REPO}, ensure_ascii=True, indent=2))
        return

    if args.command == "speak":
        api = _api_from_args(args)
        wavs, sr = api.generate(
            args.text,
            language=args.language,
            speaker=args.speaker,
            instruct=args.instruct,
            download=args.download,
            **_generation_kwargs(args),
        )
        print(
            json.dumps({"outputs": [str(p) for p in api.save_wavs(wavs, sr, args.output)], "sample_rate": sr}, indent=2)
        )
        return

    if args.command == "info":
        api = _api_from_args(args)
        api.load(download=args.download)
        print(
            json.dumps(
                {"languages": api.supported_languages(), "speakers": api.supported_speakers()},
                ensure_ascii=True,
                indent=2,
            )
        )
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

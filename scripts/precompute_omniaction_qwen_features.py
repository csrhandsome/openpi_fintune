"""Precompute static instruction-audio Qwen features per episode.

Supports both:
  - Franka realworld ASR sidecars (instruction often mid-episode)
  - OmniAction-LIBERO converted sidecars (static dialogue for the whole episode)

Default output:
  <dataset_root>/qwen_instruction_features/episode_XXXXXX.npy

Examples:
    # Franka (writes under data/realworld/franka_lerobot_7_6_audio/qwen_instruction_features)
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
      .venv/bin/python scripts/precompute_omniaction_qwen_features.py \
      pi0_base_audio_joint_lora --qwen --workers 4 --threads-per-worker 4

    # OmniAction-LIBERO (writes under data/omniaction_libero/qwen_features/<repo_id>)
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
      .venv/bin/python scripts/precompute_omniaction_qwen_features.py \
      pi0_omniaction_libero_10 --workers 4 --threads-per-worker 4
"""

from __future__ import annotations

import argparse
from concurrent import futures
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import time

import numpy as np
import tqdm

from openpi.shared import audio_tools

_EPISODE_PATTERN = re.compile(r"episode_(\d+)\.sync\.json$")
_WORKER_CONFIG: dict[str, object] = {}


def qwen_feature_cache_dir(dataset_root: str | Path) -> Path:
    """Default cache root for per-episode Qwen instruction features."""
    return audio_tools.qwen_instruction_feature_cache_dir(dataset_root)


def _resolve_dataset_root(repo_id: str) -> Path:
    repo_path = Path(repo_id)
    if repo_path.is_dir():
        return repo_path
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / repo_id


def _instruction_frame_index(
    audio_sidecar_dir: Path,
    episode_index: int,
    fallback_fps: float,
) -> int | None:
    """Pick a frame where static instruction audio is active.

    Compatibility:
      - Franka ASR: ``instruction_audio_window`` usually starts ~9s in; use mid-window.
      - OmniAction: window is ``[0, episode_end]`` (or absent); mid-window / frame 0 both work.
      - Missing/invalid window: fall back to frame 0 and let extraction decide validity.
    """
    sync_path = audio_sidecar_dir / "audio" / f"episode_{episode_index:06d}.sync.json"
    if not sync_path.is_file():
        return None
    sync_data = json.loads(sync_path.read_text())
    fps = float(sync_data.get("control_frequency", fallback_fps))
    video_frames = sync_data.get("video_frames")
    max_frame = max(0, int(video_frames) - 1) if video_frames is not None else None

    window = sync_data.get("instruction_audio_window")
    if isinstance(window, dict):
        if not bool(window.get("audio_valid", True)):
            return None
        start_sec = window.get("start_sec")
        end_sec = window.get("end_sec")
        if start_sec is not None and end_sec is not None and float(end_sec) >= float(start_sec):
            mid_sec = 0.5 * (float(start_sec) + float(end_sec))
            frame_index = max(0, int(mid_sec * fps))
            if max_frame is not None:
                frame_index = min(frame_index, max_frame)
            return frame_index

    # OmniAction / legacy sidecars without an explicit window: try the episode start.
    return 0


def _init_worker(
    model_name: str,
    device: str,
    worker_count: int,
    threads_per_worker: int,
    audio_sidecar_dir: str,
    output_dir: str,
    sample_rate: int,
    instruction_num_samples: int,
    expected_num_frames: int,
    expected_dim: int,
    output_dtype: str,
    overwrite: int,
    fallback_fps: float,
) -> None:
    import torch

    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed_cpus = sorted(os.sched_getaffinity(0))
        # Prefer one hardware thread per physical core on this host before using SMT siblings.
        physical_cpus = [cpu for cpu in allowed_cpus if cpu < len(allowed_cpus) // 2]
        candidate_cpus = physical_cpus if len(physical_cpus) >= worker_count * threads_per_worker else allowed_cpus
        process_identity = multiprocessing.current_process()._identity  # noqa: SLF001
        worker_index = ((process_identity[0] - 1) if process_identity else 0) % worker_count
        start = worker_index * threads_per_worker
        worker_cpus = candidate_cpus[start : start + threads_per_worker]
        if len(worker_cpus) == threads_per_worker:
            os.sched_setaffinity(0, worker_cpus)

    torch.set_num_threads(threads_per_worker)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)

    _WORKER_CONFIG.update(
        {
            "model_name": model_name,
            "device": device,
            "audio_sidecar_dir": audio_sidecar_dir,
            "output_dir": output_dir,
            "sample_rate": sample_rate,
            "instruction_num_samples": instruction_num_samples,
            "expected_num_frames": expected_num_frames,
            "expected_dim": expected_dim,
            "output_dtype": output_dtype,
            "overwrite": bool(overwrite),
            "fallback_fps": float(fallback_fps),
        }
    )

    # Load one model per worker up front; subsequent episodes reuse the process-local LRU cache.
    audio_tools._load_qwen_asr_components(model_name, device)  # noqa: SLF001


def _precompute_episode(episode_index: int) -> tuple[int, str, float]:
    start = time.perf_counter()
    output_dir = Path(str(_WORKER_CONFIG["output_dir"]))
    output_path = output_dir / f"episode_{episode_index:06d}.npy"
    if output_path.is_file() and not bool(_WORKER_CONFIG["overwrite"]):
        return episode_index, "skipped", time.perf_counter() - start

    sample_rate = int(_WORKER_CONFIG["sample_rate"])
    instruction_num_samples = int(_WORKER_CONFIG["instruction_num_samples"])
    expected_num_frames = int(_WORKER_CONFIG["expected_num_frames"])
    expected_dim = int(_WORKER_CONFIG["expected_dim"])
    fallback_fps = float(_WORKER_CONFIG["fallback_fps"])
    audio_sidecar_dir = Path(str(_WORKER_CONFIG["audio_sidecar_dir"]))

    frame_index = _instruction_frame_index(audio_sidecar_dir, episode_index, fallback_fps)
    if frame_index is None:
        return episode_index, "invalid", time.perf_counter() - start

    audio, valid = audio_tools.extract_episode_instruction_audio(
        dataset_root=str(audio_sidecar_dir),
        episode_index=episode_index,
        frame_index=frame_index,
        window_num_samples=instruction_num_samples,
        target_sample_rate=sample_rate,
        fallback_fps=fallback_fps,
    )
    if not valid:
        return episode_index, "invalid", time.perf_counter() - start

    feature = audio_tools.compute_qwen_asr_audio_sequence(
        audio,
        sample_rate=sample_rate,
        model_name=str(_WORKER_CONFIG["model_name"]),
        device=str(_WORKER_CONFIG["device"]),
    )
    feature = audio_tools.fit_audio_feature_sequence(feature, expected_num_frames)
    if feature.shape != (expected_num_frames, expected_dim):
        raise ValueError(
            f"Episode {episode_index} feature shape mismatch: "
            f"expected {(expected_num_frames, expected_dim)}, got {feature.shape}."
        )

    feature = np.asarray(feature, dtype=np.dtype(str(_WORKER_CONFIG["output_dtype"])))
    temporary_path = output_dir / f".episode_{episode_index:06d}.{os.getpid()}.tmp"
    with temporary_path.open("wb") as file:
        np.save(file, feature, allow_pickle=False)
    os.replace(temporary_path, output_path)
    return episode_index, "written", time.perf_counter() - start


def _episode_indices(audio_sidecar_dir: Path) -> list[int]:
    indices = []
    for sync_path in sorted((audio_sidecar_dir / "audio").glob("episode_*.sync.json")):
        match = _EPISODE_PATTERN.search(sync_path.name)
        if match is not None:
            indices.append(int(match.group(1)))
    if not indices:
        raise FileNotFoundError(f"No episode sync files found under {audio_sidecar_dir / 'audio'}")
    return indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_name",
        help="Audio train config, e.g. pi0_base_audio_joint_lora or pi0_audio_omniaction_libero_object",
    )
    parser.add_argument("--audio-sidecar-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--qwen",
        action="store_true",
        help="Enable Qwen encoder on the config (default audio configs use Whisper).",
    )
    parser.add_argument("--qwen-model-name", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--qwen-device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.threads_per_worker <= 0:
        raise ValueError("--threads-per-worker must be > 0")

    # Some OpenMP runtimes narrow the parent affinity as soon as they are imported. Restore the
    # container's full CPU set before spawning workers, then assign disjoint cores in _init_worker.
    if hasattr(os, "sched_setaffinity"):
        with contextlib.suppress(OSError):
            os.sched_setaffinity(0, range(os.cpu_count() or 1))

    # Importing the full config module is intentionally kept out of worker module import.
    from openpi.models import pi0_audio
    from openpi.training import config as training_config

    audio_data_config_types = (
        training_config.LeRobotLiberoAudioDataConfig,
        training_config.LeRobotFrankaJointAudioDataConfig,
        training_config.LeRobotOmniActionLiberoDataConfig,
    )

    config = training_config.get_config(args.config_name)
    if args.qwen or not (isinstance(config.model, pi0_audio.Pi0AudioConfig) and config.model.use_qwen):
        config = training_config.enable_qwen_encoder(
            config,
            qwen_model_name=args.qwen_model_name,
            qwen_device=args.qwen_device,
        )
    if not isinstance(config.data, audio_data_config_types):
        raise TypeError(
            f"{args.config_name} is not a supported audio LeRobot config "
            f"(expected Libero/OmniAction or Franka joint audio)."
        )
    if not isinstance(config.model, pi0_audio.Pi0AudioConfig) or not config.model.use_qwen:
        raise TypeError(f"{args.config_name} does not use the Qwen audio encoder.")
    if config.model.audio_tokenizer_type != "sequence_resampler":
        raise ValueError("Precomputation currently requires audio_tokenizer_type='sequence_resampler'.")

    repo_id = config.data.repo_id
    if args.audio_sidecar_dir is not None:
        audio_sidecar_dir = args.audio_sidecar_dir
    elif getattr(config.data, "audio_sidecar_dir", None):
        audio_sidecar_dir = Path(str(config.data.audio_sidecar_dir))
    else:
        audio_sidecar_dir = _resolve_dataset_root(str(repo_id))
    audio_sidecar_dir = audio_sidecar_dir.expanduser().resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif isinstance(config.data, training_config.LeRobotOmniActionLiberoDataConfig):
        output_dir = training_config.omniaction_qwen_feature_cache_dir(str(repo_id))
    else:
        output_dir = qwen_feature_cache_dir(audio_sidecar_dir)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_indices = _episode_indices(audio_sidecar_dir)
    if args.limit is not None:
        episode_indices = episode_indices[: args.limit]

    sample_rate = config.model.audio_sample_rate
    instruction_num_samples = config.model.instruction_audio_num_samples
    expected_num_frames = audio_tools.qwen_asr_num_frames_for_duration(
        instruction_num_samples,
        sample_rate=sample_rate,
    )
    expected_dim = config.model.audio_feature_dim
    fallback_fps = float(config.data.audio_fallback_fps)

    for env_name, value in {
        "OMP_NUM_THREADS": args.threads_per_worker,
        "MKL_NUM_THREADS": args.threads_per_worker,
        "OPENBLAS_NUM_THREADS": 1,
        "NUMEXPR_NUM_THREADS": min(args.threads_per_worker, 4),
        "TOKENIZERS_PARALLELISM": "false",
        "MALLOC_ARENA_MAX": 4,
    }.items():
        os.environ[env_name] = str(value)

    metadata = {
        "format_version": 1,
        "config_name": args.config_name,
        "repo_id": repo_id,
        "audio_sidecar_dir": str(audio_sidecar_dir),
        "model_name": config.model.qwen_model_name,
        "sample_rate": sample_rate,
        "instruction_num_samples": instruction_num_samples,
        "feature_shape": [expected_num_frames, expected_dim],
        "dtype": args.dtype,
        "fallback_fps": fallback_fps,
        "episodes": len(episode_indices),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    written = 0
    skipped = 0
    invalid = 0
    context = multiprocessing.get_context("spawn")
    with futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(
            config.model.qwen_model_name,
            args.device,
            args.workers,
            args.threads_per_worker,
            str(audio_sidecar_dir),
            str(output_dir),
            sample_rate,
            instruction_num_samples,
            expected_num_frames,
            expected_dim,
            args.dtype,
            int(args.overwrite),
            fallback_fps,
        ),
    ) as executor:
        pending = [executor.submit(_precompute_episode, episode_index) for episode_index in episode_indices]
        with tqdm.tqdm(total=len(pending), desc="Precomputing Qwen features", dynamic_ncols=True) as progress:
            for future in futures.as_completed(pending):
                episode_index, status, elapsed = future.result()
                written += status == "written"
                skipped += status == "skipped"
                invalid += status == "invalid"
                progress.set_postfix(
                    episode=episode_index,
                    status=status,
                    seconds=f"{elapsed:.2f}",
                    written=written,
                    skipped=skipped,
                    invalid=invalid,
                )
                progress.update()

    print(f"Done. output_dir={output_dir} written={written} skipped={skipped} invalid={invalid}")


if __name__ == "__main__":
    main()

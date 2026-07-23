"""Convert OmniAction-LIBERO RLDS (+ speech wavs) to LeRobot with audio sidecars.

RLDS stores dialogue references in ``speech_conv`` (``[UNK]``-joined relative paths).
This script resolves those paths under ``--audio_root``, concatenates turns into one
16 kHz mono wav per episode, and writes openpi sidecar files:

  <HF_LEROBOT_HOME>/<repo_id>/audio/episode_XXXXXX.wav
  <HF_LEROBOT_HOME>/<repo_id>/audio/episode_XXXXXX.sync.json

Usage:
  export HF_ENDPOINT=https://hf-mirror.com   # optional mirror
  uv run examples/omniaction_libero/convert_omniaction_to_lerobot.py \
    --data_dir data/omniaction_libero/rlds \
    --audio_root data/omniaction_libero \
    --suite object

Requires: ``uv pip install tensorflow tensorflow_datasets``
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import shutil
import wave

import numpy as np
import tensorflow_datasets as tfds
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

TARGET_SR = 16_000
SPEECH_SEP = "[UNK]"
FPS = 10

SUITE_RAW_DATASETS: dict[str, list[str]] = {
    "spatial": [
        "libero_speech_spatial_events",
        "libero_speech_spatial_sentiment",
        "libero_speech_spatial_overlap",
        "libero_speech_spatial_identity",
        "libero_speech_spatial_three",
        "libero_speech_spatial_two",
    ],
    "object": [
        "libero_speech_object_events",
        "libero_speech_object_sentiment",
        "libero_speech_object_overlap",
        "libero_speech_object_identity",
        "libero_speech_object_three",
        "libero_speech_object_two",
    ],
    "goal": [
        "libero_speech_goal_events",
        "libero_speech_goal_sentiment",
        "libero_speech_goal_overlap",
        "libero_speech_goal_identity",
        "libero_speech_goal_three",
        "libero_speech_goal_two",
    ],
    "10": [
        "libero_speech10_events",
        "libero_speech10_sentiment",
        "libero_speech10_overlap",
        "libero_speech10_identity",
        "libero_speech10_three",
        "libero_speech10_two",
    ],
}


@dataclasses.dataclass
class Args:
    data_dir: str = "data/omniaction_libero/rlds"
    audio_root: str = "data/omniaction_libero"
    suite: str = "object"  # spatial | object | goal | 10
    repo_id: str | None = None  # default: omniaction_libero_{suite} (suite=10 -> omniaction_libero10)
    push_to_hub: bool = False
    overwrite: bool = True
    max_episodes: int | None = None  # smoke-test limit across all raw datasets


def _read_wav_mono(path: pathlib.Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data / float(np.iinfo(dtype).max), sr


def _write_wav_mono(path: pathlib.Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio
    duration = audio.shape[0] / float(orig_sr)
    n_target = int(round(duration * target_sr))
    if n_target <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _resolve_wav(audio_root: pathlib.Path, entry: str) -> pathlib.Path | None:
    """Resolve one speech_conv reference against ``audio_root``.

    Handles flat refs and nested refs. Mirrors eval_omniaction_libero._resolve_wav
    so training and eval feed the same audio files.
    """
    rel = pathlib.PurePosixPath(entry.lstrip("./"))
    if rel.parts and rel.parts[0] in ("libero_audio", "audio"):
        rel = pathlib.PurePosixPath(*rel.parts[1:])
    candidates = [
        audio_root / rel.name,
        audio_root / rel.as_posix(),
        audio_root / "speech" / rel.as_posix(),
        audio_root / "speech" / rel.name,
    ]
    if len(rel.parts) > 1:
        candidates.append(audio_root / pathlib.PurePosixPath(*rel.parts[1:]).as_posix())
        candidates.append(audio_root / "speech" / pathlib.PurePosixPath(*rel.parts[1:]).as_posix())
        candidates.append(audio_root.parent / rel.as_posix())
    for c in candidates:
        if c.is_file():
            return c
    for search_root in (audio_root, audio_root / "speech"):
        if search_root.is_dir():
            hits = list(search_root.rglob(rel.name))
            if hits:
                return hits[0]
    return None


def _parse_speech_conv(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8", errors="replace")
    elif isinstance(raw, str):
        text = raw
    else:
        # already a sequence of path strings / bytes
        try:
            items = list(raw)
        except TypeError:
            return []
        out = []
        for item in items:
            if isinstance(item, (bytes, bytearray)):
                s = item.decode("utf-8", errors="replace").strip()
            else:
                s = str(item).strip()
            if s:
                out.append(s)
        return out
    text = text.strip()
    if not text:
        return []
    return [p for p in text.split(SPEECH_SEP) if p]


def _load_concat_audio(audio_root: pathlib.Path, speech_conv: list[str]) -> np.ndarray:
    segments = []
    for entry in speech_conv:
        path = _resolve_wav(audio_root, entry)
        if path is None:
            logging.warning("missing wav '%s' under %s", entry, audio_root)
            continue
        data, sr = _read_wav_mono(path)
        segments.append(_resample(data, sr, TARGET_SR))
    if not segments:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(segments, axis=0)


def _write_episode_sidecar(
    dataset_root: pathlib.Path,
    episode_index: int,
    audio: np.ndarray,
    num_frames: int,
) -> None:
    audio_dir = dataset_root / "audio"
    stem = f"episode_{int(episode_index):06d}"
    wav_path = audio_dir / f"{stem}.wav"
    sync_path = audio_dir / f"{stem}.sync.json"

    if audio.size == 0:
        # Keep a 1s silence placeholder so sidecar extraction stays well-defined.
        audio = np.zeros((TARGET_SR,), dtype=np.float32)

    _write_wav_mono(wav_path, audio, TARGET_SR)
    audio_dur = float(audio.shape[0]) / float(TARGET_SR)
    episode_dur = float(num_frames) / float(FPS)
    # Keep instruction audio active for the whole episode (RoboOmni static dialogue).
    end_sec = max(audio_dur, episode_dur) + 1.0
    sync = {
        "control_frequency": float(FPS),
        "sample_rate": TARGET_SR,
        "audio_path": f"audio/{stem}.wav",
        "instruction_audio_window": {
            "start_sec": 0.0,
            "end_sec": end_sec,
            "audio_valid": True,
        },
    }
    sync_path.write_text(json.dumps(sync, indent=2), encoding="utf-8")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.suite not in SUITE_RAW_DATASETS:
        raise ValueError(f"Unknown suite={args.suite!r}; expected one of {sorted(SUITE_RAW_DATASETS)}")

    if args.repo_id:
        repo_id = args.repo_id
    elif args.suite == "10":
        repo_id = "omniaction_libero10"
    else:
        repo_id = f"omniaction_libero_{args.suite}"
    output_path = HF_LEROBOT_HOME / repo_id
    audio_root = pathlib.Path(args.audio_root)
    data_dir = pathlib.Path(args.data_dir)

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace")
        logging.info("Removing existing dataset at %s", output_path)
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    written = 0
    for raw_name in SUITE_RAW_DATASETS[args.suite]:
        logging.info("Loading RLDS dataset %s from %s", raw_name, data_dir)
        try:
            raw_dataset = tfds.load(raw_name, data_dir=str(data_dir), split="train")
        except Exception as e:
            logging.error("Failed to load %s: %s", raw_name, e)
            continue

        for episode in tqdm.tqdm(raw_dataset, desc=raw_name):
            if args.max_episodes is not None and written >= args.max_episodes:
                break

            steps = list(episode["steps"].as_numpy_iterator())
            if not steps:
                continue

            # speech_conv is constant across steps in OmniAction-LIBERO.
            speech_conv = _parse_speech_conv(steps[0].get("speech_conv"))
            audio = _load_concat_audio(audio_root, speech_conv)

            for step in steps:
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"].astype(np.float32),
                        "actions": step["action"].astype(np.float32),
                        # RoboOmni: intent from audio; keep text prompt empty.
                        "task": "",
                    }
                )
            dataset.save_episode()
            _write_episode_sidecar(output_path, written, audio, num_frames=len(steps))
            written += 1

        if args.max_episodes is not None and written >= args.max_episodes:
            break

    logging.info("Wrote %d episodes to %s", written, output_path)

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["omniaction", "libero", "panda", "audio"],
            private=False,
            push_videos=False,
            license="cc-by-nc-4.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

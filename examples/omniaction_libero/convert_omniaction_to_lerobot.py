"""Convert OmniAction-LIBERO (RLDS) to a LeRobot dataset for pi0_audio.

RoboOmni/OmniAction-LIBERO stores, per episode:
  - image / wrist_image (256x256x3), state (8D), action (7D)
  - language_instruction: EMPTY (intent must be inferred from the spoken dialogue)
  - speech_conv: audio file paths, e.g. "./libero_overlap/39-1.wav[UNK]...[UNK]39-3.wav"

The dialogue audio is STATIC across the whole episode (RoboOmni is a semantic task,
not a time-triggered one). So we:
  1. Write the standard LeRobot frames (image/wrist/state/action/task).
  2. For each episode, concatenate the referenced wavs, resample to 16 kHz, and write a
     pi0_audio audio sidecar at  <audio_out_dir>/audio/episode_XXXXXX.wav  plus a
     sync.json whose `instruction_audio_window` covers the full clip -> every frame maps
     to the same whole-dialogue instruction_audio slot.

The eval wavs come from OpenMOSS-Team/OmniAction-LIBERO-eval, laid out by speech type,
e.g. data/libero_audio/overlap/39-1.wav .

Run through this repository's pinned ``uv`` environment (it contains the
patched LeRobot version used by training):

  .venv/bin/python examples/omniaction_libero/convert_omniaction_to_lerobot.py \
    --rlds_dir data/omniaction_libero/libero_speech_object_overlap/libero_speech_object_overlap/1.0.0 \
    --audio_root data/libero_audio/overlap \
    --repo_id omniaction_libero_object_overlap \
    --audio_out_dir data/omniaction_libero/sidecars/object_overlap

The LeRobot dataset is written to $HF_LEROBOT_HOME/<repo_id>; audio sidecars go to
<audio_out_dir>/audio/ (this path is what the DataConfig passes as `dataset_root`).
"""

import argparse
import dataclasses
import glob
import json
import pathlib
import shutil
import wave

import numpy as np
import tensorflow as tf

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

TARGET_SR = 16_000
SPEECH_SEP = "[UNK]"


@dataclasses.dataclass
class Args:
    rlds_dir: str
    """Path to the RLDS 1.0.0 dir containing *.tfrecord* + features.json."""
    audio_root: str
    """Dir with the eval wavs for this speech type, e.g. data/libero_audio/overlap."""
    repo_id: str
    """Output LeRobot dataset name (written under HF_LEROBOT_HOME)."""
    audio_out_dir: str
    """Dir where audio sidecars (audio/episode_XXXXXX.wav + .sync.json) are written.
    This is the `dataset_root` the training DataConfig will point at."""
    fps: int = 20
    prompt: str = ""
    """Text prompt per frame. RoboOmni infers intent from audio, so default empty."""
    max_episodes: int = -1
    """Limit episodes for a quick smoke test; -1 = all."""
    push_to_hub: bool = False


def build_talker_prompt(previous_robot_turns=()) -> str:
    lines = ["User audio is provided. Respond as the robot."]
    history = [" ".join(str(turn).strip().split()) for turn in previous_robot_turns if str(turn).strip()]
    if history:
        lines.append("Previous robot turns:")
        lines.extend(history)
    lines.append("Robot:")
    return "\n".join(lines)


def build_action_prompt(robot_turns=()) -> str:
    lines = ["User audio is provided. Execute the confirmed request."]
    history = [" ".join(str(turn).strip().split()) for turn in robot_turns if str(turn).strip()]
    if history:
        lines.append("Robot dialogue:")
        lines.extend(history)
    return "\n".join(lines)


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
    # normalize to [-1, 1]
    max_val = float(np.iinfo(dtype).max)
    data = data / max_val
    return data, sr


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


def _write_wav_mono(path: pathlib.Path, audio: np.ndarray, sr: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _parse_speech_conv(speech_conv: str) -> list[str]:
    """Return the referenced wav paths, RELATIVE (not just basename).

    Flat speech types (overlap/sentiment/events) reference a bare filename, e.g. ``39-1.wav``.
    Nested ones (identity) reference a subdir, e.g. ``dialogue_101/user_turn_0.wav`` -- basename
    alone would collide across dialogues, so we keep the relative path and let ``_resolve_wav``
    try several roots. Leading ``./`` is stripped.
    """
    parts = [p.strip() for p in speech_conv.split(SPEECH_SEP)]
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.startswith("./"):
            p = p[2:]
        out.append(pathlib.PurePosixPath(p).as_posix())
    return out


def _resolve_wav(audio_root: pathlib.Path, rel: str) -> pathlib.Path | None:
    """Resolve one speech_conv reference against ``audio_root`` by trying several layouts.

    Handles: flat (audio_root/name.wav), nested (audio_root/dialogue_X/turn.wav), a leading
    speech-type dir component (audio_root.parent/<type>/...), and basename fallback.
    """
    rel_path = pathlib.PurePosixPath(rel)
    candidates = [
        audio_root / rel_path.as_posix(),  # audio_root/dialogue_101/user_turn_0.wav  (nested)
        audio_root / rel_path.name,  # audio_root/39-1.wav  (flat)
    ]
    if len(rel_path.parts) > 1:
        # reference already includes the speech-type dir (e.g. "identity/dialogue_101/x.wav")
        candidates.append(audio_root.parent / rel_path.as_posix())
        candidates.append(audio_root / pathlib.PurePosixPath(*rel_path.parts[1:]).as_posix())
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_episode_audio(audio_root: pathlib.Path, wav_names: list[str]) -> tuple[np.ndarray, int]:
    """Concatenate the resolved wavs (resampled to 16 kHz). Returns (audio, n_missing)."""
    segments = []
    n_missing = 0
    for name in wav_names:
        p = _resolve_wav(audio_root, name)
        if p is None:
            print(f"  WARNING: missing wav '{name}' under {audio_root}")
            n_missing += 1
            continue
        data, sr = _read_wav_mono(p)
        segments.append(_resample(data, sr, TARGET_SR))
    if not segments:
        return np.zeros((0,), dtype=np.float32), n_missing
    return np.concatenate(segments, axis=0), n_missing


def _load_episode_audio_with_segments(
    audio_root: pathlib.Path,
    wav_names: list[str],
) -> tuple[np.ndarray, int, list[dict]]:
    """Load a concatenated dialogue waveform plus resampled per-turn boundaries."""
    segments = []
    metadata = []
    n_missing = 0
    cursor = 0
    for index, name in enumerate(wav_names):
        p = _resolve_wav(audio_root, name)
        if p is None:
            print(f"  WARNING: missing wav '{name}' under {audio_root}")
            n_missing += 1
            continue
        data, sr = _read_wav_mono(p)
        data = _resample(data, sr, TARGET_SR)
        start_sample = cursor
        cursor += int(data.shape[0])
        segments.append(data)
        metadata.append(
            {
                "audio_turn_index": index,
                "wav_name": name,
                "start_sec": float(start_sample) / TARGET_SR,
                "end_sec": float(cursor) / TARGET_SR,
            }
        )
    if not segments:
        return np.zeros((0,), dtype=np.float32), n_missing, metadata
    return np.concatenate(segments, axis=0), n_missing, metadata


def _decode_first(values: list[bytes]) -> str:
    return values[0].decode("utf-8") if values else ""


def _parse_conversation_turns(conversation: str) -> list[dict]:
    """Parse RoboOmni's ``[USER]... [ASSISTANT]...`` serialized turns."""
    turns = []
    for raw_turn in conversation.split(SPEECH_SEP):
        if "[ASSISTANT]" not in raw_turn:
            continue
        user_part, response = raw_turn.split("[ASSISTANT]", 1)
        user = user_part.split("[USER]", 1)[-1].strip()
        response = response.strip()
        if response:
            turns.append({"user_text": user, "robot_response": response})
    return turns


def _build_dialogue_metadata(
    feature_map,
    *,
    audio_segments: list[dict],
) -> tuple[list[dict], list[int], str]:
    """Return turn metadata, source-frame locations, and executor text prompt."""

    def _bytes(key: str) -> list[bytes]:
        return feature_map[key].bytes_list.value if key in feature_map else []

    conversation = _decode_first(_bytes("steps/conversation"))
    parsed_turns = _parse_conversation_turns(conversation)
    responses = [value.decode("utf-8").strip() for value in _bytes("steps/response")]
    response_frame_indices = [index for index, response in enumerate(responses) if response]

    # Some serialized records omit/garble the conversation string, but response
    # is authoritative at the turn frames.
    if not parsed_turns:
        parsed_turns = [{"user_text": "", "robot_response": response} for response in responses if response]

    turn_count = min(len(parsed_turns), len(response_frame_indices))
    turns = []
    history = []
    for turn_index in range(turn_count):
        response = responses[response_frame_indices[turn_index]] or parsed_turns[turn_index]["robot_response"]
        audio_end_sec = (
            float(audio_segments[min(turn_index, len(audio_segments) - 1)]["end_sec"]) if audio_segments else 0.0
        )
        turns.append(
            {
                "turn_index": turn_index,
                "frame_index": response_frame_indices[turn_index],
                "user_text": parsed_turns[turn_index].get("user_text", ""),
                "robot_response": response,
                "text_prompt": build_talker_prompt(history),
                # The Talker sees all user speech available up to this turn,
                # never a future confirmation turn.
                "context_audio_start_sec": 0.0,
                "context_audio_end_sec": audio_end_sec,
            }
        )
        history.append(response)

    return turns, response_frame_indices[:turn_count], build_action_prompt(history)


# LeRobot feature schema shared by the single-subset and merged converters. All 10 OmniAction-
# LIBERO subsets are LIBERO/panda with identical dims (256x256 rgb, 8D state, 7D action).
_FEATURES = {
    "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def create_lerobot_dataset(repo_id: str, fps: int = 20) -> LeRobotDataset:
    """Create a fresh LeRobot dataset under HF_LEROBOT_HOME/<repo_id> (wiping any existing one)."""
    out_path = HF_LEROBOT_HOME / repo_id
    print(f"  lerobot home: {HF_LEROBOT_HOME}  (dataset -> {out_path})")
    if out_path.exists():
        shutil.rmtree(out_path)
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features=_FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )


def write_episode(
    dataset: LeRobotDataset,
    audio_out: pathlib.Path,
    ex: "tf.train.Example",
    ep_idx: int,
    audio_root: pathlib.Path,
    prompt: str,
    fps: int,
) -> tuple[bool, int, dict]:
    """Write one RLDS episode's frames + audio sidecar under global index ``ep_idx``.

    Returns ``(written, n_missing_wavs, sync)``. ``written`` is False for 0-step episodes (skipped);
    ``sync`` is the sidecar dict written to ``episode_{ep_idx:06d}.sync.json``.
    """
    f = ex.features.feature

    def _bytes(key):
        return f[key].bytes_list.value if key in f else []

    images = _bytes("steps/observation/image")
    wrist_images = _bytes("steps/observation/wrist_image")
    states_flat = np.array(f["steps/observation/state"].float_list.value, dtype=np.float32)
    actions_flat = np.array(f["steps/action"].float_list.value, dtype=np.float32)
    speech_convs = _bytes("steps/speech_conv")
    n_steps = len(images)
    if n_steps == 0:
        return False, 0, {}
    states = states_flat.reshape(n_steps, -1)
    actions = actions_flat.reshape(n_steps, -1)

    # audio: static per episode -> take the first frame's speech_conv
    wav_names = _parse_speech_conv(speech_convs[0].decode("utf-8")) if speech_convs else []
    episode_audio, n_missing, audio_segments = _load_episode_audio_with_segments(audio_root, wav_names)
    dialogue_turns, dialogue_turn_frame_indices, action_prompt = _build_dialogue_metadata(
        f,
        audio_segments=audio_segments,
    )

    for t in range(n_steps):
        img = tf.io.decode_image(images[t], channels=3).numpy()
        wimg = tf.io.decode_image(wrist_images[t], channels=3).numpy()
        dataset.add_frame(
            {
                "image": img,
                "wrist_image": wimg,
                "state": states[t].astype(np.float32),
                "actions": actions[t].astype(np.float32),
                "task": prompt,
            }
        )
    dataset.save_episode()

    # write audio sidecar for this (global) episode index
    stem = f"episode_{ep_idx:06d}"
    _write_wav_mono(audio_out / f"{stem}.wav", episode_audio, TARGET_SR)
    duration_sec = float(episode_audio.shape[0]) / TARGET_SR if episode_audio.size else 0.0
    sync = {
        "episode_index": ep_idx,
        "control_frequency": float(fps),
        "audio_path": f"audio/{stem}.wav",
        "video_frames": n_steps,
        # whole-dialogue static instruction slot: covers the full episode timeline
        "instruction_audio_window": {
            "start_sec": 0.0,
            "end_sec": 1e9,
            "audio_valid": bool(episode_audio.size > 0),
        },
        "audio_duration_sec": duration_sec,
        "wav_names": wav_names,
        # New turn metadata is ignored by the legacy full-dialogue path. It is
        # consumed only by the Talker text-loss transform / interactive eval.
        "dialogue": dialogue_turns,
        "dialogue_turn_frame_indices": dialogue_turn_frame_indices,
        "action_prompt": action_prompt,
    }
    with (audio_out / f"{stem}.sync.json").open("w", encoding="utf-8") as fp:
        json.dump(sync, fp, ensure_ascii=False, indent=2)
    return True, n_missing, sync


def main(args: Args) -> None:
    audio_root = pathlib.Path(args.audio_root)
    audio_out = pathlib.Path(args.audio_out_dir) / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    dataset = create_lerobot_dataset(args.repo_id, fps=args.fps)

    files = sorted(glob.glob(str(pathlib.Path(args.rlds_dir) / "*.tfrecord*")))
    if not files:
        raise FileNotFoundError(f"No tfrecords in {args.rlds_dir}")
    raw_ds = tf.data.TFRecordDataset(files)

    ep_idx = 0
    total_missing = 0
    for raw in raw_ds:
        ex = tf.train.Example.FromString(raw.numpy())
        written, n_missing, _ = write_episode(dataset, audio_out, ex, ep_idx, audio_root, args.prompt, args.fps)
        if not written:
            continue
        total_missing += n_missing
        ep_idx += 1
        if ep_idx % 20 == 0:
            print(f"  converted {ep_idx} episodes...")
        if args.max_episodes > 0 and ep_idx >= args.max_episodes:
            break

    print(f"DONE: {ep_idx} episodes -> {HF_LEROBOT_HOME / args.repo_id}  (missing wavs: {total_missing})")
    print(f"Audio sidecars -> {audio_out}")
    if args.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds_dir", required=True)
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--audio_out_dir", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--push_to_hub", action="store_true")
    ns = parser.parse_args()
    main(Args(**vars(ns)))

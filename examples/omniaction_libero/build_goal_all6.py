"""Build a standalone *goal-only* OmniAction-LIBERO LeRobot dataset covering ALL 6 speech types,
so we can single-suite train a goal model (RoboOmni-style) with full speech coverage.

Why: the merged dataset only had goal x {events, identity, sentiment}. goal x {overlap, two, three}
were never trained (no tfrecord on HF). Diagnosis (identity trained -> ~1.0 once the eval silence
bug was fixed) showed the encoder/policy CAN learn multi-party dialogue -> action; the failing
types just need training data.

Approach (CPU only, no GPU / no LIBERO sim):
  Phase A  events/identity/sentiment  -> convert real RLDS as-is (real per-episode audio).
  Phase B  overlap/two/three          -> REUSE the goal/events trajectories and attach the
           matching speech audio by instruction:
             events episode -> video_id (from its speech_conv filename)
                            -> instruction (libero_events.json[video_id])
                            -> target speech_conv (libero_<speech>.json[instruction])
           The base robot trajectory is identical across speech types for the same task, so this
           is a valid audio->action pairing (10 unique clips/task, same as the working types).

Run (from repo root, matches the training env; force CPU so it never touches the training GPUs):
  CUDA_VISIBLE_DEVICES="" HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD/data/lerobot}" \
      .venv/bin/python examples/omniaction_libero/build_goal_all6.py

Outputs:
  $HF_LEROBOT_HOME/omniaction_libero_goal/            (LeRobot dataset, all 6 goal speech types)
  data/omniaction_libero/sidecars/goal/audio/episode_XXXXXX.{wav,sync.json}
  data/omniaction_libero/sidecars/goal/manifest.json
"""

import argparse
import glob
import json
import os
import pathlib
import re
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_omniaction_to_lerobot as conv  # noqa: E402

DATA = pathlib.Path("third_party/RoboOmni/data")
AUDIO_BASE = pathlib.Path("data/libero_audio")
RLDS_ROOT = pathlib.Path("data/omniaction_libero")

# real subsets with tfrecord on disk (converted as-is with their own audio)
REAL = [
    ("events", "libero_speech_goal_events"),
    ("identity", "libero_speech_goal_identity"),
    ("sentiment", "libero_speech_goal_sentiment"),
]
# synthetic subsets: reuse events trajectories, swap in this speech's audio (matched by instruction)
SYNTH = ["overlap", "two", "three"]
JSON_STEM = {"overlap": "libero_overlap", "two": "libero_two", "three": "libero_three", "events": "libero_events"}


def _norm(instr: str) -> str:
    return (instr or "").strip().lower().rstrip(".")


def _find_rlds_dir(hf_subset: str) -> pathlib.Path:
    hits = sorted(glob.glob(str(RLDS_ROOT / hf_subset / "**" / "1.0.0"), recursive=True))
    for h in hits:
        if pathlib.Path(h).is_dir():
            return pathlib.Path(h)
    raise FileNotFoundError(f"No 1.0.0 dir under {RLDS_ROOT / hf_subset}")


def _goal_entries(stem: str) -> list[dict]:
    d = json.load(open(DATA / f"{stem}.json", encoding="utf-8"))
    return [x for x in d if x.get("meta") in ("goal", "libero_goal")]


def _build_lookups():
    """video_id -> instruction; target speech -> (dialogue item, wav refs)."""
    vid2instr = {}
    for x in _goal_entries("libero_events"):
        vid = x.get("video_id")
        if vid is not None:
            vid2instr[int(vid)] = x["instruction"]
    instr2dialogue = {sp: {} for sp in SYNTH}
    for sp in SYNTH:
        for x in _goal_entries(JSON_STEM[sp]):
            wavs = [
                e.lstrip("./")[len("libero_audio/") :] if e.lstrip("./").startswith("libero_audio/") else e.lstrip("./")
                for e in (x.get("speech_conv") or [])
            ]
            instr2dialogue[sp][_norm(x["instruction"])] = (x, wavs)
    return vid2instr, instr2dialogue


_VID_RE = re.compile(r"libero_goal_(\d+)_")


def _events_video_id(ex) -> int | None:
    f = ex.features.feature
    sc = f["steps/speech_conv"].bytes_list.value if "steps/speech_conv" in f else []
    if not sc:
        return None
    first = conv._parse_speech_conv(sc[0].decode("utf-8"))
    for name in first:
        m = _VID_RE.search(name)
        if m:
            return int(m.group(1))
    return None


def _write_episode_with_audio(
    dataset,
    audio_out,
    ex,
    ep_idx,
    episode_audio,
    wav_names,
    fps,
    dialogue_item=None,
    audio_segments=None,
):
    """Like conv.write_episode but the audio is supplied (not read from the RLDS speech_conv)."""
    f = ex.features.feature
    images = f["steps/observation/image"].bytes_list.value
    wrist = f["steps/observation/wrist_image"].bytes_list.value
    states = np.array(f["steps/observation/state"].float_list.value, dtype=np.float32)
    actions = np.array(f["steps/action"].float_list.value, dtype=np.float32)
    n = len(images)
    if n == 0:
        return False, {}
    states = states.reshape(n, -1)
    actions = actions.reshape(n, -1)
    for t in range(n):
        dataset.add_frame(
            {
                "image": tf.io.decode_image(images[t], channels=3).numpy(),
                "wrist_image": tf.io.decode_image(wrist[t], channels=3).numpy(),
                "state": states[t].astype(np.float32),
                "actions": actions[t].astype(np.float32),
                "task": "",
            }
        )
    dataset.save_episode()
    stem = f"episode_{ep_idx:06d}"
    conv._write_wav_mono(audio_out / f"{stem}.wav", episode_audio, conv.TARGET_SR)
    dur = float(episode_audio.shape[0]) / conv.TARGET_SR if episode_audio.size else 0.0
    sync = {
        "episode_index": ep_idx,
        "control_frequency": float(fps),
        "audio_path": f"audio/{stem}.wav",
        "video_frames": n,
        "instruction_audio_window": {"start_sec": 0.0, "end_sec": 1e9, "audio_valid": bool(episode_audio.size > 0)},
        "audio_duration_sec": dur,
        "wav_names": wav_names,
    }
    if dialogue_item is not None:
        history = []
        turns = []
        for raw_turn in dialogue_item.get("conversation", []):
            response = str(raw_turn.get("response", "")).strip()
            if not response:
                continue
            turn_index = len(turns)
            segment = audio_segments[min(turn_index, len(audio_segments) - 1)] if audio_segments else {}
            turns.append(
                {
                    "turn_index": turn_index,
                    "frame_index": -1,
                    "user_text": str(raw_turn.get("user", "")),
                    "robot_response": response,
                    "text_prompt": conv.build_talker_prompt(history),
                    "context_audio_start_sec": 0.0,
                    "context_audio_end_sec": float(segment.get("end_sec", 0.0)),
                }
            )
            history.append(response)
        sync.update(
            {
                "dialogue": turns,
                "dialogue_turn_frame_indices": [],
                "action_prompt": conv.build_action_prompt(history),
            }
        )
    json.dump(sync, (audio_out / f"{stem}.sync.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return True, sync


def main(args):
    audio_out = pathlib.Path(args.audio_out_dir) / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)
    dataset = conv.create_lerobot_dataset(args.repo_id, fps=args.fps)
    vid2instr, instr2dialogue = _build_lookups()

    ep_idx = 0
    manifest = []
    counts = []

    # ---- Phase A: real subsets (events/identity/sentiment) with their own audio ----
    for speech, hf_subset in REAL:
        rlds_dir = _find_rlds_dir(hf_subset)
        audio_root = AUDIO_BASE / ({"sentiment": "libero_sentiment"}.get(speech, speech))
        files = sorted(glob.glob(str(rlds_dir / "*.tfrecord*")))
        print(f"==> [A] goal/{speech}  {rlds_dir}  (audio_root={audio_root})")
        start = ep_idx
        for raw in tf.data.TFRecordDataset(files):
            ex = tf.train.Example.FromString(raw.numpy())
            written, _n, sync = conv.write_episode(dataset, audio_out, ex, ep_idx, audio_root, "", args.fps)
            if not written:
                continue
            manifest.append(
                {
                    "episode_index": ep_idx,
                    "suite": "goal",
                    "speech": speech,
                    "hf_subset": hf_subset,
                    "source_episode": ep_idx - start,
                    "audio_valid": sync["instruction_audio_window"]["audio_valid"],
                    "audio_duration_sec": sync["audio_duration_sec"],
                }
            )
            ep_idx += 1
        counts.append((f"goal/{speech}", ep_idx - start))
        print(f"    -> {ep_idx - start} episodes")

    # ---- Phase B: synth subsets (overlap/two/three) reusing events trajectories ----
    events_dir = _find_rlds_dir("libero_speech_goal_events")
    events_files = sorted(glob.glob(str(events_dir / "*.tfrecord*")))
    for speech in SYNTH:
        print(f"==> [B] goal/{speech}  (reuse events trajectories, audio by instruction)")
        start = ep_idx
        n_skip = 0
        for raw in tf.data.TFRecordDataset(events_files):
            ex = tf.train.Example.FromString(raw.numpy())
            vid = _events_video_id(ex)
            instr = vid2instr.get(vid) if vid is not None else None
            pair = instr2dialogue[speech].get(_norm(instr)) if instr else None
            if pair is None:
                n_skip += 1
                continue
            dialogue_item, wavs = pair
            audio, _n_missing, segments = conv._load_episode_audio_with_segments(AUDIO_BASE, wavs)
            if audio.size == 0:
                n_skip += 1
                continue
            written, sync = _write_episode_with_audio(
                dataset,
                audio_out,
                ex,
                ep_idx,
                audio,
                wavs,
                args.fps,
                dialogue_item,
                segments,
            )
            if not written:
                continue
            manifest.append(
                {
                    "episode_index": ep_idx,
                    "suite": "goal",
                    "speech": speech,
                    "hf_subset": f"synth_goal_{speech}",
                    "source_episode": ep_idx - start,
                    "audio_valid": sync["instruction_audio_window"]["audio_valid"],
                    "audio_duration_sec": sync["audio_duration_sec"],
                    "source_video_id": vid,
                }
            )
            ep_idx += 1
        counts.append((f"goal/{speech}", ep_idx - start))
        print(f"    -> {ep_idx - start} episodes (skipped {n_skip} unmatched)")

    mpath = pathlib.Path(args.audio_out_dir) / "manifest.json"
    json.dump(
        {"total_episodes": ep_idx, "rows": manifest}, mpath.open("w", encoding="utf-8"), ensure_ascii=False, indent=2
    )
    print("\n=== GOAL-ONLY BUILD SUMMARY ===")
    for name, n in counts:
        print(f"  {name:16s} {n:5d} eps")
    print(f"TOTAL: {ep_idx} episodes -> {conv.HF_LEROBOT_HOME / args.repo_id}")
    print(f"Audio sidecars -> {audio_out}\nManifest -> {mpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="omniaction_libero_goal")
    p.add_argument("--audio_out_dir", default="data/omniaction_libero/sidecars/goal")
    p.add_argument("--fps", type=int, default=20)
    main(p.parse_args())

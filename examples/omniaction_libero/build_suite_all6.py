"""Build a standalone single-suite OmniAction-LIBERO LeRobot dataset with ALL 6 speech types,
for object / spatial / 10 (goal is built by build_goal_all6.py). RoboOmni-style per-suite model.

Real subsets (tfrecord on disk) are converted as-is with their own audio. The missing speech
types are synthesized by reusing the *sentiment* trajectories (flat, uniquely-named audio -> clean
instruction recovery) and attaching the target speech's audio matched by instruction. The base
robot trajectory is speech-type invariant, so this is a valid audio->action pairing.

Instruction recovery: libero_sentiment.json[suite] gives basename(speech_conv) -> instruction; each
sentiment RLDS episode's speech_conv basename looks up its instruction; then
libero_<target>.json[suite][instruction] gives the target audio.

Run (repo root; CPU-only, never touches the training GPUs):
  CUDA_VISIBLE_DEVICES="" HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD/data/lerobot}" \
      .venv/bin/python examples/omniaction_libero/build_suite_all6.py --suite object
  ...                                                                 --suite spatial
  ...                                                                 --suite 10

Outputs (per suite): $HF_LEROBOT_HOME/omniaction_libero_<suite>/,
  data/omniaction_libero/sidecars/<suite>/audio/*, data/omniaction_libero/sidecars/<suite>/manifest.json
"""

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_omniaction_to_lerobot as conv  # noqa: E402

DATA = pathlib.Path("third_party/RoboOmni/data")
AUDIO_BASE = pathlib.Path("data/libero_audio")

# per suite: real subsets present on disk (speech -> hf_subset dir) + speech types to synthesize.
SUITE_SPEC = {
    "object": {
        "real": {
            "overlap": "libero_speech_object_overlap",
            "identity": "libero_speech_object_identity",
            "sentiment": "libero_speech_object_sentiment",
        },
        "synth": ["events", "two", "three"],
    },
    "spatial": {
        "real": {"identity": "libero_speech_spatial_identity", "sentiment": "libero_speech_spatial_sentiment"},
        "synth": ["events", "overlap", "two", "three"],
    },
    "10": {
        "real": {"identity": "libero_speech10_identity", "sentiment": "libero_speech10_sentiment"},
        "synth": ["events", "overlap", "two", "three"],
    },
}
JSON_STEM = {
    "overlap": "libero_overlap",
    "two": "libero_two",
    "three": "libero_three",
    "events": "libero_events",
    "identity": "libero_identity",
    "sentiment": "libero_sentiment",
}
AUDIO_SUBDIR = {
    "overlap": "overlap",
    "identity": "identity",
    "sentiment": "libero_sentiment",
    "events": "events",
    "two": "two",
    "three": "three",
}
META_ALIAS = {
    "object": ("object", "libero_object"),
    "spatial": ("spatial", "libero_spatial"),
    "10": ("10", "libero_10"),
}


def _norm(s):
    return (s or "").strip().lower().rstrip(".")


def _find_rlds_dir(hf_subset):
    hits = sorted(glob.glob(str(pathlib.Path("data/omniaction_libero") / hf_subset / "**" / "1.0.0"), recursive=True))
    for h in hits:
        if pathlib.Path(h).is_dir():
            return pathlib.Path(h)
    raise FileNotFoundError(f"No 1.0.0 dir under data/omniaction_libero/{hf_subset}")


def _entries(stem, suite):
    d = json.load(open(DATA / f"{stem}.json", encoding="utf-8"))
    return [x for x in d if x.get("meta") in META_ALIAS[suite]]


def _strip(entry):
    rel = entry.lstrip("./")
    return rel[len("libero_audio/") :] if rel.startswith("libero_audio/") else rel


def _build_lookups(suite, synth):
    """basename -> instruction; target speech -> (dialogue item, wav refs)."""
    base2instr = {}
    for x in _entries("libero_sentiment", suite):
        for e in x.get("speech_conv") or []:
            base2instr[pathlib.PurePosixPath(e).name] = x["instruction"]
    instr2dialogue = {sp: {} for sp in synth}
    for sp in synth:
        for x in _entries(JSON_STEM[sp], suite):
            wavs = [_strip(e) for e in (x.get("speech_conv") or [])]
            instr2dialogue[sp][_norm(x["instruction"])] = (x, wavs)
    return base2instr, instr2dialogue


def _episode_instr(ex, base2instr):
    f = ex.features.feature
    sc = f["steps/speech_conv"].bytes_list.value if "steps/speech_conv" in f else []
    if not sc:
        return None
    for name in conv._parse_speech_conv(sc[0].decode("utf-8")):
        instr = base2instr.get(pathlib.PurePosixPath(name).name)
        if instr:
            return instr
    return None


def _write_with_audio(dataset, audio_out, ex, ep_idx, audio, wav_names, fps, dialogue_item=None, audio_segments=None):
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
    conv._write_wav_mono(audio_out / f"{stem}.wav", audio, conv.TARGET_SR)
    dur = float(audio.shape[0]) / conv.TARGET_SR if audio.size else 0.0
    sync = {
        "episode_index": ep_idx,
        "control_frequency": float(fps),
        "audio_path": f"audio/{stem}.wav",
        "video_frames": n,
        "instruction_audio_window": {"start_sec": 0.0, "end_sec": 1e9, "audio_valid": bool(audio.size > 0)},
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
    suite = args.suite
    spec = SUITE_SPEC[suite]
    repo_id = args.repo_id or f"omniaction_libero_{suite}"
    audio_out_dir = args.audio_out_dir or f"data/omniaction_libero/sidecars/{suite}"
    audio_out = pathlib.Path(audio_out_dir) / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)
    dataset = conv.create_lerobot_dataset(repo_id, fps=args.fps)
    base2instr, instr2dialogue = _build_lookups(suite, spec["synth"])

    ep_idx = 0
    manifest = []
    counts = []

    # ---- Phase A: real subsets with their own audio ----
    for speech, hf_subset in spec["real"].items():
        rlds_dir = _find_rlds_dir(hf_subset)
        audio_root = AUDIO_BASE / AUDIO_SUBDIR[speech]
        files = sorted(glob.glob(str(rlds_dir / "*.tfrecord*")))
        print(f"==> [A] {suite}/{speech}  {rlds_dir}  (audio_root={audio_root})")
        start = ep_idx
        for raw in tf.data.TFRecordDataset(files):
            ex = tf.train.Example.FromString(raw.numpy())
            written, _n, sync = conv.write_episode(dataset, audio_out, ex, ep_idx, audio_root, "", args.fps)
            if not written:
                continue
            manifest.append(
                {
                    "episode_index": ep_idx,
                    "suite": suite,
                    "speech": speech,
                    "hf_subset": hf_subset,
                    "source_episode": ep_idx - start,
                    "audio_valid": sync["instruction_audio_window"]["audio_valid"],
                    "audio_duration_sec": sync["audio_duration_sec"],
                }
            )
            ep_idx += 1
        counts.append((f"{suite}/{speech}", ep_idx - start))
        print(f"    -> {ep_idx - start} episodes")

    # ---- Phase B: synth subsets, reuse SENTIMENT trajectories, audio by instruction ----
    bank_dir = _find_rlds_dir(spec["real"]["sentiment"])
    bank_files = sorted(glob.glob(str(bank_dir / "*.tfrecord*")))
    for speech in spec["synth"]:
        print(f"==> [B] {suite}/{speech}  (reuse sentiment trajectories, audio by instruction)")
        start = ep_idx
        n_skip = 0
        for raw in tf.data.TFRecordDataset(bank_files):
            ex = tf.train.Example.FromString(raw.numpy())
            instr = _episode_instr(ex, base2instr)
            pair = instr2dialogue[speech].get(_norm(instr)) if instr else None
            if pair is None:
                n_skip += 1
                continue
            dialogue_item, wavs = pair
            audio_root = AUDIO_BASE / AUDIO_SUBDIR[speech]
            audio, _nmiss, segments = conv._load_episode_audio_with_segments(audio_root, wavs)
            if audio.size == 0:
                n_skip += 1
                continue
            written, sync = _write_with_audio(
                dataset, audio_out, ex, ep_idx, audio, wavs, args.fps, dialogue_item, segments
            )
            if not written:
                continue
            manifest.append(
                {
                    "episode_index": ep_idx,
                    "suite": suite,
                    "speech": speech,
                    "hf_subset": f"synth_{suite}_{speech}",
                    "source_episode": ep_idx - start,
                    "audio_valid": sync["instruction_audio_window"]["audio_valid"],
                    "audio_duration_sec": sync["audio_duration_sec"],
                }
            )
            ep_idx += 1
        counts.append((f"{suite}/{speech}", ep_idx - start))
        print(f"    -> {ep_idx - start} episodes (skipped {n_skip} unmatched)")

    mpath = pathlib.Path(audio_out_dir) / "manifest.json"
    json.dump(
        {"total_episodes": ep_idx, "rows": manifest}, mpath.open("w", encoding="utf-8"), ensure_ascii=False, indent=2
    )
    print(f"\n=== {suite.upper()} BUILD SUMMARY ===")
    for name, n in counts:
        print(f"  {name:16s} {n:5d} eps")
    print(f"TOTAL: {ep_idx} -> {conv.HF_LEROBOT_HOME / repo_id}\nsidecars -> {audio_out}\nmanifest -> {mpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, choices=["object", "spatial", "10"])
    p.add_argument("--repo_id", default="")
    p.add_argument("--audio_out_dir", default="")
    p.add_argument("--fps", type=int, default=20)
    main(p.parse_args())

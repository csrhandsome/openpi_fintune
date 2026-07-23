"""OmniAction-LIBERO (RoboOmni) eval harness for pi0_audio.

Runs the LIBERO simulator and feeds the per-task spoken dialogue audio into the pi0_audio
policy server via the `observation/instruction_audio` slot (raw waveform; the server-side
AudiovlaInputs -> ExtractInstructionAudioWindow resamples/fits it). RoboOmni is a semantic
intent-inference task, so the dialogue audio is loaded once per task and kept static for the
whole episode.

Task -> dialogue-wav mapping comes from RoboOmni's data/libero_<speech>.json files, which map
each LIBERO task instruction string to a `speech_conv` list of wavs under the eval audio set.

Usage (client side; a pi0_audio policy server must be serving separately):

  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=8
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  python examples/omniaction_libero/eval_omniaction_libero.py \
    --task_suite_name libero_object \
    --speech overlap \
    --speech_json third_party/RoboOmni/data/libero_overlap.json \
    --audio_root data/libero_audio/overlap \
    --host 0.0.0.0 --port 8000
"""

from __future__ import annotations  # PEP 585 generic aliases (tuple[...], list[...]) need py3.9+ at runtime without this

import collections
import dataclasses
import json
import logging
import math
import pathlib
import wave

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
TARGET_SR = 16_000
SPEECH_SEP = "[UNK]"


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_object"
    speech: str = "overlap"  # overlap | identity | libero_sentiment | events | two | three
    speech_json: str = "third_party/RoboOmni/data/libero_overlap.json"
    audio_root: str = "data/libero_audio/overlap"

    # How to feed the time-series ambient_audio slot (active in full_avla training, 1s/1-token):
    #   shared  -> send the same dialogue waveform (server fits the last 1s window). No server change.
    #   silent  -> omit ambient slot; requires the serving config to set audio_strict=False so the
    #              slot is zero-filled + masked off (matches the "ambient disabled" design intent).
    ambient_mode: str = "shared"

    num_steps_wait: int = 10
    num_trials_per_task: int = 10

    video_out_path: str = "data/omniaction_libero/videos"
    render_gpu_device_id: int = 0
    seed: int = 7


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

    Handles flat refs (``39-1.wav`` -> ``audio_root/39-1.wav``) AND nested refs
    (``identity/dialogue_130/user_turn_0.wav`` -> ``audio_root/dialogue_130/user_turn_0.wav``),
    where basename alone collides across dialogues. Mirrors the training-time resolver in
    ``convert_omniaction_to_lerobot.py`` so eval feeds the SAME audio the model trained on.
    """
    rel = pathlib.PurePosixPath(entry.lstrip("./"))
    if rel.parts and rel.parts[0] == "libero_audio":
        rel = pathlib.PurePosixPath(*rel.parts[1:])
    candidates = [
        audio_root / rel.name,  # flat: audio_root/39-1.wav
        audio_root / rel.as_posix(),  # audio_root/identity/dialogue_X/turn.wav
    ]
    if len(rel.parts) > 1:
        # strip a leading speech-type dir component (e.g. "identity/dialogue_X/turn.wav")
        candidates.append(audio_root / pathlib.PurePosixPath(*rel.parts[1:]).as_posix())
        candidates.append(audio_root.parent / rel.as_posix())
    for c in candidates:
        if c.is_file():
            return c
    # last resort: unique basename anywhere under audio_root (handles dialogue_X nesting)
    hits = list(audio_root.rglob(rel.name))
    return hits[0] if hits else None


def _load_task_audio(audio_root: pathlib.Path, speech_conv: list[str]) -> np.ndarray:
    segments = []
    for entry in speech_conv:
        p = _resolve_wav(audio_root, entry)
        if p is None:
            logging.warning(f"missing wav '{entry}' under {audio_root}")
            continue
        data, sr = _read_wav_mono(p)
        segments.append(_resample(data, sr, TARGET_SR))
    if not segments:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(segments, axis=0)


def _build_inst2audio(speech_json: str, audio_root: pathlib.Path) -> dict[str, np.ndarray]:
    with open(speech_json, encoding="utf-8") as f:
        data = json.load(f)
    inst2audio = {}
    for item in data:
        inst2audio[item["instruction"]] = _load_task_audio(audio_root, item.get("speech_conv", []))
    return inst2audio


def eval_omniaction_libero(args: Args) -> None:
    np.random.seed(args.seed)
    audio_root = pathlib.Path(args.audio_root)
    inst2audio = _build_inst2audio(args.speech_json, audio_root)
    logging.info(f"Loaded audio for {len(inst2audio)} task instructions.")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[args.task_suite_name]

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.render_gpu_device_id)

        task_audio = inst2audio.get(task_description)
        if task_audio is None:
            logging.warning(f"No dialogue audio for task '{task_description}'; using silence.")
            task_audio = np.zeros((TARGET_SR,), dtype=np.float32)
        has_audio = bool(task_audio.size > 0)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []
            done = False
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            # RoboOmni: intent inferred from audio; prompt kept empty like training.
                            "prompt": "",
                            # Whole-dialogue static instruction audio (raw waveform).
                            "observation/instruction_audio": task_audio,
                            "observation/instruction_audio_sample_rate": np.int32(TARGET_SR),
                            "observation/instruction_audio_mask": np.bool_(has_audio),
                        }
                        if args.ambient_mode == "shared":
                            # Ambient slot is also active under full_avla training (1s/1-token); feed the
                            # same dialogue waveform so the server-side window extraction matches training
                            # instead of a zeroed/masked-off ambient slot the model never saw.
                            element["observation/ambient_audio"] = task_audio
                            element["observation/ambient_audio_sample_rate"] = np.int32(TARGET_SR)
                            element["observation/ambient_audio_mask"] = np.bool_(has_audio)
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:60]
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"{args.speech}_{task_segment}_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            logging.info(
                f"[{args.task_suite_name}/{args.speech}] success={done} "
                f"total={total_successes}/{total_episodes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(f"Task {task_id} success rate: {task_successes / max(task_episodes, 1):.3f}")

    logging.info(
        f"TOTAL [{args.task_suite_name}/{args.speech}] success rate: "
        f"{total_successes / max(total_episodes, 1):.3f} over {total_episodes} episodes"
    )


def _get_libero_env(task, resolution, seed, render_gpu_device_id):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "render_gpu_device_id": render_gpu_device_id,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_omniaction_libero(args)

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import os
import signal
import time

from franka_robot_env import FrankaCameraIds
from franka_robot_env import FrankaRobotEnv
from moviepy import ImageSequenceClip
import numpy as np
from openpi_client import websocket_client_policy
from panda_py import controllers
from panda_py import libfranka
import pandas as pd
from PIL import Image
from plot_topk import plot_topk_xy
import tyro

from openpi.shared import audio_tools

DEFAULT_CONTROL_FREQUENCY = 30.0
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0
FRANKA_JOINT_LIMITS = np.array(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float64,
)


def _center_crop_and_resize(rgb: np.ndarray, out_hw: int = 224, crop_scale: float = 0.9) -> np.ndarray:
    """Center crop + resize to match training data preprocessing."""
    h, w = rgb.shape[:2]
    frac = float(np.sqrt(crop_scale))
    crop_h = max(1, min(h, round(h * frac)))
    crop_w = max(1, min(w, round(w * frac)))
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    cropped = rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
    img = Image.fromarray(cropped, mode="RGB").resize((out_hw, out_hw), resample=Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


@dataclasses.dataclass
class Args:
    # Camera naming (must match keys produced by FrankaRobotEnv)
    left_camera_id: str = "external"
    right_camera_id: str = "right"
    wrist_camera_id: str = "wrist"

    # Which external camera should be fed to the policy, choose from ['left', 'right']
    external_camera: str | None = "left"
    instruction: str = "pick up the yellow cup"

    # Rollout parameters
    max_timesteps: int = 2000
    # How many actions to execute from a predicted action chunk before querying policy server again
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "192.168.5.89"
    remote_port: int = 8000

    # Franka parameters
    franka_hostname: str = "192.168.1.100"
    franka_username: str = "your_username"
    franka_password: str = "your_password"

    # RealSense parameters
    external_camera_serial: str | None = None
    wrist_camera_serial: str | None = None
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    color_only: bool = False
    use_external_for_right: bool = True

    # Joint-position control parameters
    control_frequency_hz: float = DEFAULT_CONTROL_FREQUENCY
    joint_velocity_limit: float = 0.0
    max_joint_delta: float = 0.0
    action_epsilon: float = 1e-6
    controller_settle_s: float = 0.5
    gripper_open_threshold: float = 0.5

    # Top-k clustering count (set to >1 to enable; None/<=1 disables)
    topk_samples: int | None = None
    # Number of stochastic policy draws used before clustering.
    # If None, defaults to max(4 * topk_samples, topk_samples + 1).
    topk_draws: int | None = None

    # Audio conditioning parameters for pi0_base_audio_joint_lora.
    audio_required: bool = True
    audio_device: str | None = None
    audio_sample_rate: int = 16_000
    audio_channels: int = 1
    audio_num_samples: int = 16_000
    audio_buffer_num_samples: int = 32_000
    audio_block_duration_s: float = 0.05
    audio_startup_timeout_s: float = 1.0


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _kmeans_topk(samples: np.ndarray, k: int = 3, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2:
        raise ValueError(f"Expected samples shape (N, D), got {samples.shape}")
    if samples.shape[0] == 0:
        raise ValueError("No samples provided for top-k clustering.")
    k = min(int(k), samples.shape[0])
    rng = np.random.default_rng(0)
    centers = samples[rng.choice(samples.shape[0], size=k, replace=False)]
    for _ in range(int(iters)):
        d2 = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d2.argmin(-1)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = samples[mask].mean(0)
    probs = np.bincount(labels, minlength=k).astype(np.float32)
    probs = probs / np.clip(probs.sum(), 1.0, None)
    order = np.argsort(-probs)
    return centers[order], probs[order]


def _normalize_action_chunk(actions: np.ndarray) -> np.ndarray:
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[1] < 8:
        raise ValueError(f"Unexpected action chunk shape: {chunk.shape}")
    return chunk[:, :8]


def _hold_current_joint_position(env: FrankaRobotEnv, controller) -> np.ndarray:
    q_curr = np.asarray(env.robot.panda.get_state().q, dtype=np.float64).reshape(7)
    controller.set_control(q_curr, np.zeros(7, dtype=np.float64))
    return q_curr


def _start_joint_position_controller(env: FrankaRobotEnv, *, settle_s: float = 0.0):
    controller = controllers.JointPosition()
    env.robot.panda.start_controller(controller)
    if settle_s > 0.0:
        time.sleep(settle_s)
    _hold_current_joint_position(env, controller)
    return controller


def _sanitize_joint_target(args: Args, q_curr: np.ndarray, joint_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_curr = np.asarray(q_curr, dtype=np.float64).reshape(7)
    target = np.asarray(joint_target, dtype=np.float64).reshape(7)
    target = np.clip(target, FRANKA_JOINT_LIMITS[:, 0], FRANKA_JOINT_LIMITS[:, 1])

    joint_delta = target - q_curr
    if args.max_joint_delta > 0.0:
        joint_delta = np.clip(joint_delta, -args.max_joint_delta, args.max_joint_delta)
        target = np.clip(q_curr + joint_delta, FRANKA_JOINT_LIMITS[:, 0], FRANKA_JOINT_LIMITS[:, 1])
        joint_delta = target - q_curr

    if args.action_epsilon > 0.0:
        small_mask = np.abs(joint_delta) <= args.action_epsilon
        if np.any(small_mask):
            target = target.copy()
            joint_delta = joint_delta.copy()
            target[small_mask] = q_curr[small_mask]
            joint_delta[small_mask] = 0.0

    return target, joint_delta


def _extract_observation(
    args: Args,
    env: FrankaRobotEnv,
    gripper_state: float,
    *,
    save_to_disk: bool = False,
):
    last_observation = getattr(env, "_joint_main_last_observation", None)

    external_ok, wrist_ok = env.cameras.update(timeout_ms=env.camera_timeout_ms)
    external_img = wrist_img = None
    if external_ok and wrist_ok:
        external_img, wrist_img = env.cameras.get_images()

    if external_img is None or wrist_img is None:
        if last_observation is None:
            raise RuntimeError("Camera update failed and no previous observation is available.")
        left_image = last_observation["left_image"]
        right_image = last_observation["right_image"]
        wrist_image = last_observation["wrist_image"]
    else:
        left_image = np.asarray(external_img, dtype=np.uint8).copy()
        wrist_image = np.asarray(wrist_img, dtype=np.uint8).copy()
        right_image = left_image.copy() if args.use_external_for_right else wrist_image.copy()

    robot_state = env.robot.panda.get_state()
    ee_pose = np.asarray(robot_state.O_T_EE, dtype=np.float32).reshape(4, 4).T
    joint_position = np.asarray(robot_state.q, dtype=np.float32).reshape(7)
    gripper_position = np.asarray([np.float32(gripper_state)], dtype=np.float32)

    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        Image.fromarray(combined_image).save("robot_camera_views.png")

    observation = {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": ee_pose[:3, 3],
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }
    env._joint_main_last_observation = observation
    return observation


def _start_audio_buffer(args: Args) -> audio_tools.MultiprocessAudioBuffer | None:
    audio_buffer = audio_tools.MultiprocessAudioBuffer(
        sample_rate=args.audio_sample_rate,
        channels=args.audio_channels,
        window_num_samples=args.audio_num_samples,
        buffer_num_samples=args.audio_buffer_num_samples,
        block_duration_s=args.audio_block_duration_s,
        device=args.audio_device,
    )
    try:
        audio_buffer.start()
        ready = audio_buffer.wait_until_ready(args.audio_startup_timeout_s)
    except Exception as exc:
        audio_buffer.stop()
        if args.audio_required:
            raise RuntimeError(f"Failed to start microphone capture: {exc}") from exc
        print(f"[Audio] Warning: microphone capture unavailable, continuing without audio: {exc}")
        return None

    if not ready:
        audio_buffer.stop()
        msg = f"No microphone samples arrived within {args.audio_startup_timeout_s:.2f}s."
        if args.audio_required:
            raise RuntimeError(msg)
        print(f"[Audio] Warning: {msg} Continuing without audio.")
        return None

    print(
        "[Audio] Live microphone capture enabled (multiprocess): "
        f"{audio_buffer.sample_rate} Hz, {args.audio_channels} channel(s), "
        f"window={args.audio_num_samples} samples, "
        f"device={audio_buffer.input_device if audio_buffer.input_device is not None else 'default'}"
    )
    return audio_buffer


def _apply_gripper_if_needed(
    env: FrankaRobotEnv,
    controller,
    current_gripper_state: float,
    next_gripper_state: float,
):
    if next_gripper_state == current_gripper_state:
        return controller, current_gripper_state

    _hold_current_joint_position(env, controller)
    env.robot.panda.stop_controller()
    if next_gripper_state > 0.5:
        env.robot.gripper_open()
    else:
        env.robot.gripper_close()
    controller = _start_joint_position_controller(env, settle_s=0.0)
    return controller, next_gripper_state


def _reset_robot(env: FrankaRobotEnv, controller, *, settle_s: float) -> tuple[object, float]:
    with contextlib.suppress(Exception):
        env.robot.stop_velocity_streaming()
    with contextlib.suppress(Exception):
        _hold_current_joint_position(env, controller)
    with contextlib.suppress(Exception):
        env.robot.panda.stop_controller()
    env.robot.gripper_open()
    auto_move_to_start = getattr(env.robot, "auto_move_to_start", True)
    env.robot.auto_move_to_start = True
    try:
        env.robot.move_to_start()
    finally:
        env.robot.auto_move_to_start = auto_move_to_start
    controller = _start_joint_position_controller(env, settle_s=settle_s)
    return controller, GRIPPER_OPEN


def main(args: Args):
    if args.external_camera not in ["left", "right"]:
        raise ValueError("Please specify an external camera to use for the policy, choose from ['left', 'right']")
    if args.control_frequency_hz <= 0.0:
        raise ValueError("--control-frequency-hz must be > 0")
    if args.open_loop_horizon <= 0:
        raise ValueError("--open-loop-horizon must be > 0")
    if args.max_joint_delta < 0.0:
        raise ValueError("--max-joint-delta must be >= 0")
    if args.joint_velocity_limit < 0.0:
        raise ValueError("--joint-velocity-limit must be >= 0")
    if args.audio_sample_rate <= 0:
        raise ValueError("--audio-sample-rate must be > 0")
    if args.audio_channels <= 0:
        raise ValueError("--audio-channels must be > 0")
    if args.audio_num_samples <= 0:
        raise ValueError("--audio-num-samples must be > 0")
    if args.audio_buffer_num_samples < args.audio_num_samples:
        raise ValueError("--audio-buffer-num-samples must be >= --audio-num-samples")
    if args.audio_block_duration_s <= 0.0:
        raise ValueError("--audio-block-duration-s must be > 0")
    if args.audio_startup_timeout_s < 0.0:
        raise ValueError("--audio-startup-timeout-s must be >= 0")

    env = FrankaRobotEnv(
        action_space="joint_velocity",
        gripper_action_space="position",
        franka_hostname=args.franka_hostname,
        franka_username=args.franka_username,
        franka_password=args.franka_password,
        camera_ids=FrankaCameraIds(
            external=args.left_camera_id,
            right=args.right_camera_id,
            wrist=args.wrist_camera_id,
        ),
        external_camera_serial=args.external_camera_serial,
        wrist_camera_serial=args.wrist_camera_serial,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        color_only=args.color_only,
        use_external_for_right=args.use_external_for_right,
        control_frequency_hz=args.control_frequency_hz,
        joint_velocity_scale=1.0,
        auto_move_to_start=False,
    )

    controller = None
    audio_buffer = None
    try:
        controller, gripper_state = _reset_robot(env, controller, settle_s=args.controller_settle_s)
        audio_buffer = _start_audio_buffer(args)

        franka_model = None
        franka_frame = None
        if args.topk_samples is not None and args.topk_samples > 1:
            franka_model = env.robot.panda.get_robot().load_model()
            franka_frame = libfranka.Frame.kEndEffector
        plot_xlim: tuple[float, float] | None = None
        plot_ylim: tuple[float, float] | None = None

        policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
        df = pd.DataFrame(columns=["success", "duration", "video_filename"])

        while True:
            instruction = args.instruction.strip() or input("Enter instruction: ").strip()
            if not instruction:
                raise ValueError("Instruction must not be empty.")

            plot_xlim = None
            plot_ylim = None
            actions_from_chunk_completed = 0
            pred_action_chunk = None
            pred_action_horizon = 0

            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")  # noqa: DTZ005
            video = []
            steps_completed = 0
            print("Running rollout... press Ctrl+C to stop early.")

            while steps_completed < args.max_timesteps:
                start_time = time.time()
                try:
                    curr_obs = _extract_observation(
                        args,
                        env,
                        gripper_state,
                        save_to_disk=steps_completed == 0,
                    )

                    external_frame = curr_obs[f"{args.external_camera}_image"]
                    wrist_frame = curr_obs["wrist_image"]
                    video.append(np.concatenate([external_frame, wrist_frame], axis=1))

                    effective_horizon = min(
                        args.open_loop_horizon,
                        pred_action_horizon or args.open_loop_horizon,
                    )
                    if pred_action_chunk is None or actions_from_chunk_completed >= effective_horizon:
                        actions_from_chunk_completed = 0
                        request_data = {
                            "observation/exterior_image_1_left": _center_crop_and_resize(
                                curr_obs[f"{args.external_camera}_image"]
                            ),
                            "observation/wrist_image_left": _center_crop_and_resize(curr_obs["wrist_image"]),
                            "observation/joint_position": curr_obs["joint_position"],
                            "observation/gripper_position": curr_obs["gripper_position"],
                            "prompt": instruction,
                        }
                        if audio_buffer is not None:
                            # 获得前1s窗口的值
                            audio_window, has_audio = audio_buffer.read_window_for_duration(
                                target_num_samples=args.audio_num_samples,
                                target_sample_rate=args.audio_sample_rate,
                            )
                            request_data["observation/audio"] = audio_window
                            request_data["observation/audio_sample_rate"] = np.int32(audio_buffer.sample_rate)
                            request_data["observation/audio_mask"] = np.bool_(has_audio)
                        # args.topk_samples默认为0(应该)
                        topk_k = int(args.topk_samples) if args.topk_samples is not None else 0
                        with prevent_keyboard_interrupt():
                            if topk_k and topk_k > 1:
                                topk_draws = (
                                    int(args.topk_draws) if args.topk_draws is not None else max(4 * topk_k, topk_k + 1)
                                )
                                topk_draws = max(topk_draws, topk_k)
                                pred_action_chunk = _normalize_action_chunk(
                                    policy_client.infer(request_data)["actions"]
                                )
                                samples = [pred_action_chunk[0]]
                                for _ in range(topk_draws - 1):
                                    samples.append(
                                        _normalize_action_chunk(policy_client.infer(request_data)["actions"])[0]
                                    )
                                centers, probs = _kmeans_topk(np.asarray(samples, dtype=np.float32), k=topk_k)

                                robot_state = env.robot.panda.get_state()
                                q_curr = np.asarray(robot_state.q, dtype=np.float64).reshape(7)
                                F_T_EE = np.asarray(robot_state.F_T_EE, dtype=np.float32).reshape(16).tolist()
                                EE_T_K = np.asarray(robot_state.EE_T_K, dtype=np.float32).reshape(16).tolist()

                                topk_xyz = []
                                for center in centers:
                                    q_next, _ = _sanitize_joint_target(args, q_curr, center[:7])
                                    pose = franka_model.pose(franka_frame, q_next.tolist(), F_T_EE, EE_T_K)
                                    pose = np.asarray(pose, dtype=np.float32).reshape(4, 4).T
                                    topk_xyz.append(pose[:3, 3])
                                topk_xyz = np.asarray(topk_xyz, dtype=np.float32)
                                curr_pose = np.asarray(robot_state.O_T_EE, dtype=np.float32).reshape(4, 4).T
                                curr_ee_xyz = curr_pose[:3, 3]
                                if plot_xlim is None or plot_ylim is None:
                                    xy = np.concatenate(
                                        [
                                            topk_xyz[:, :2],
                                            curr_ee_xyz[:2].reshape(1, 2),
                                        ],
                                        axis=0,
                                    )
                                    xy_min = xy.min(axis=0)
                                    xy_max = xy.max(axis=0)
                                    span = np.maximum(
                                        xy_max - xy_min,
                                        np.array([0.15, 0.15], dtype=np.float32),
                                    )
                                    pad = 0.25 * span
                                    plot_xlim = (
                                        float(xy_min[0] - pad[0]),
                                        float(xy_max[0] + pad[0]),
                                    )
                                    plot_ylim = (
                                        float(xy_min[1] - pad[1]),
                                        float(xy_max[1] + pad[1]),
                                    )
                                plot_path = plot_topk_xy(
                                    topk_xyz,
                                    probs,
                                    current_xyz=curr_ee_xyz,
                                    xlim=plot_xlim,
                                    ylim=plot_ylim,
                                )
                                print(f"  [TopK] plot saved: {plot_path}")
                            else:
                                pred_action_chunk = _normalize_action_chunk(
                                    policy_client.infer(request_data)["actions"]
                                )

                        pred_action_horizon = pred_action_chunk.shape[0]

                    action = np.asarray(
                        pred_action_chunk[actions_from_chunk_completed],
                        dtype=np.float32,
                    ).reshape(8)
                    actions_from_chunk_completed += 1

                    next_gripper_state = (
                        GRIPPER_OPEN if float(action[7]) > args.gripper_open_threshold else GRIPPER_CLOSED
                    )
                    controller, gripper_state = _apply_gripper_if_needed(
                        env,
                        controller,
                        gripper_state,
                        next_gripper_state,
                    )

                    q_curr = np.asarray(env.robot.panda.get_state().q, dtype=np.float64).reshape(7)
                    q_target, joint_delta = _sanitize_joint_target(args, q_curr, action[:7])
                    if args.joint_velocity_limit > 0.0:
                        qvel_ff = np.clip(
                            joint_delta * float(args.control_frequency_hz),
                            -float(args.joint_velocity_limit),
                            float(args.joint_velocity_limit),
                        )
                    else:
                        qvel_ff = np.zeros(7, dtype=np.float64)
                    controller.set_control(q_target, qvel_ff)

                    steps_completed += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 1.0 / args.control_frequency_hz:
                        time.sleep(1.0 / args.control_frequency_hz - elapsed_time)
                except KeyboardInterrupt:
                    break
                except RuntimeError as exc:
                    msg = str(exc)
                    if "motion aborted by reflex" in msg or "cartesian_reflex" in msg:
                        print(f"[Error] Franka reflex triggered; stopping rollout safely: {msg}")
                        break
                    raise

            with contextlib.suppress(Exception):
                _hold_current_joint_position(env, controller)

            if video:
                save_filename = "video_" + timestamp
                ImageSequenceClip(list(np.stack(video)), fps=int(round(args.control_frequency_hz))).write_videofile(
                    save_filename + ".mp4",
                    codec="libx264",
                )
            else:
                save_filename = "video_" + timestamp
                print("No frames captured; skipping video export.")

            success = None
            env.robot.gripper_open()
            while success is None:
                raw = (
                    input(
                        "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec: "
                    )
                    .strip()
                    .lower()
                )
                if raw == "y":
                    success = 1.0
                    break
                if raw == "n":
                    success = 0.0
                    break

                try:
                    numeric = float(raw)
                except ValueError:
                    print(f"Please enter y, n, or a numeric value in [0, 100], but got: {raw!r}")
                    continue

                if 0.0 <= numeric <= 100.0:
                    success = numeric / 100.0
                else:
                    print(f"Success must be a number in [0, 100] but got: {numeric}")

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "success": success,
                                "duration": steps_completed,
                                "video_filename": save_filename,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            if input("Do one more eval? (enter y or n) ").lower() != "y":
                break
            controller, gripper_state = _reset_robot(env, controller, settle_s=args.controller_settle_s)

        os.makedirs("results", exist_ok=True)
        controller, gripper_state = _reset_robot(env, controller, settle_s=0.0)
    finally:
        with contextlib.suppress(Exception):
            if audio_buffer is not None:
                audio_buffer.stop()
        with contextlib.suppress(Exception):
            if controller is not None:
                _hold_current_joint_position(env, controller)
        with contextlib.suppress(Exception):
            env.robot.panda.stop_controller()
        with contextlib.suppress(Exception):
            env.close()


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

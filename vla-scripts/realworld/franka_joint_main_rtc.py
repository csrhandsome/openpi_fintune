from __future__ import annotations

import concurrent.futures
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
from tqdm import tqdm
import tyro

from openpi.serving.reactive_desk_vla_client import ReactiveDeskVlaClient

DEFAULT_CONTROL_FREQUENCY = 30.0
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0
CUSTOM_START_JOINT_POSITION: tuple[float, ...] = (
    0.00388,
    -0.45916,
    -0.15237,
    -2.47266,
    -0.06778,
    1.96442,
    0.70214,
)


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
    # Number of overlapping previous-chunk actions to condition on for training-time RTC inference.
    rtc_inference_delay: int = 0

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

    # Reactive Desk visualization websocket.
    reactive_desk_enabled: bool = True
    reactive_desk_host: str = "127.0.0.1"
    reactive_desk_port: int = 8000
    reactive_desk_path: str = "/ws/VlaIngest"


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


def _rtc_prefix_from_chunk(
    previous_chunk: np.ndarray | None,
    prefix_start: int,
    delay: int,
) -> np.ndarray | None:
    if previous_chunk is None or delay <= 0:
        return None
    chunk = np.asarray(previous_chunk, dtype=np.float32)
    if chunk.ndim != 2:
        return None
    prefix = chunk[int(prefix_start) : int(prefix_start) + int(delay)]
    if prefix.shape[0] == 0:
        return None
    return prefix


def _rtc_skip_for_chunk(
    previous_chunk: np.ndarray | None,
    prefix_start: int,
    delay: int,
) -> int:
    prefix = _rtc_prefix_from_chunk(previous_chunk, prefix_start, delay)
    return 0 if prefix is None else int(prefix.shape[0])


def _attach_rtc_prefix(
    request_data: dict,
    previous_chunk: np.ndarray | None,
    prefix_start: int,
    delay: int,
) -> int:
    prefix = _rtc_prefix_from_chunk(previous_chunk, prefix_start, delay)
    if prefix is None:
        return 0
    padded_prefix = np.zeros_like(np.asarray(previous_chunk, dtype=np.float32))
    padded_prefix[: prefix.shape[0]] = prefix
    request_data["action_prefix"] = padded_prefix
    request_data["rtc_delay"] = np.asarray(prefix.shape[0], dtype=np.int32)
    return int(prefix.shape[0])


def _normalize_action_chunk(actions: np.ndarray) -> np.ndarray:
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[1] < 8:
        raise ValueError(f"Unexpected action chunk shape: {chunk.shape}")
    return chunk[:, :8]


@dataclasses.dataclass(frozen=True)
class _PolicyChunkResult:
    action_chunk: np.ndarray
    rtc_skip: int
    topk_centers: np.ndarray | None = None
    topk_probs: np.ndarray | None = None
    # Wall-clock time spent inside policy_client.infer for this request (sum over draws for top-k).
    infer_time_s: float = 0.0
    # Number of policy_client.infer calls issued for this request (>1 only for top-k).
    infer_calls: int = 0
    # Wall-clock time spent preparing request fields before policy_client.infer.
    request_build_time_s: float = 0.0


def _run_policy_request(
    policy_client: websocket_client_policy.WebsocketClientPolicy,
    request_data: dict,
    *,
    topk_k: int,
    topk_draws: int | None,
    rtc_skip: int,
) -> _PolicyChunkResult:
    if topk_k and topk_k > 1:
        sample_count = topk_draws if topk_draws is not None else max(4 * topk_k, topk_k + 1)
        sample_count = max(int(sample_count), topk_k)
        infer_start = time.monotonic()
        action_chunk = _normalize_action_chunk(policy_client.infer(request_data)["actions"])
        sample_index = min(int(rtc_skip), action_chunk.shape[0] - 1)
        samples = [action_chunk[sample_index]]
        for _ in range(sample_count - 1):
            sample_chunk = _normalize_action_chunk(policy_client.infer(request_data)["actions"])
            samples.append(sample_chunk[min(int(rtc_skip), sample_chunk.shape[0] - 1)])
        infer_time_s = time.monotonic() - infer_start
        centers, probs = _kmeans_topk(np.asarray(samples, dtype=np.float32), k=topk_k)
        return _PolicyChunkResult(
            action_chunk=action_chunk,
            rtc_skip=int(rtc_skip),
            topk_centers=centers,
            topk_probs=probs,
            infer_time_s=infer_time_s,
            infer_calls=int(sample_count),
        )

    infer_start = time.monotonic()
    action_chunk = _normalize_action_chunk(policy_client.infer(request_data)["actions"])
    infer_time_s = time.monotonic() - infer_start
    return _PolicyChunkResult(
        action_chunk=action_chunk,
        rtc_skip=int(rtc_skip),
        infer_time_s=infer_time_s,
        infer_calls=1,
    )


def _build_request_data_from_observation(
    args: Args,
    instruction: str,
    curr_obs: dict,
) -> dict:
    return {
        "observation/exterior_image_1_left": _center_crop_and_resize(curr_obs[f"{args.external_camera}_image"]),
        "observation/wrist_image_left": _center_crop_and_resize(curr_obs["wrist_image"]),
        "observation/joint_position": curr_obs["joint_position"],
        "observation/gripper_position": curr_obs["gripper_position"],
        "prompt": instruction,
    }


def _run_policy_request_from_observation(
    policy_client: websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    instruction: str,
    curr_obs: dict,
    *,
    previous_chunk: np.ndarray | None,
    prefix_start: int,
    delay: int,
    topk_k: int,
    topk_draws: int | None,
) -> _PolicyChunkResult:
    build_start = time.monotonic()
    request_data = _build_request_data_from_observation(args, instruction, curr_obs)
    rtc_skip = _attach_rtc_prefix(request_data, previous_chunk, prefix_start, delay)
    request_build_time_s = time.monotonic() - build_start
    result = _run_policy_request(
        policy_client,
        request_data,
        topk_k=topk_k,
        topk_draws=topk_draws,
        rtc_skip=rtc_skip,
    )
    return dataclasses.replace(result, request_build_time_s=request_build_time_s)


def _finish_pending_policy_future(
    pending_policy_future: concurrent.futures.Future[_PolicyChunkResult] | None,
) -> None:
    if pending_policy_future is None:
        return
    if pending_policy_future.cancel():
        return
    start_time = time.monotonic()
    print("[RTC] waiting for in-flight policy request before eval prompt")
    with prevent_keyboard_interrupt():
        pending_policy_future.result()
    elapsed_s = time.monotonic() - start_time
    print(f"[RTC] in-flight policy request finished after {elapsed_s:.2f}s")


def _send_prediction_visualization(
    *,
    args: Args,
    env: FrankaRobotEnv,
    reactive_desk_client: ReactiveDeskVlaClient,
    instruction: str,
    result: _PolicyChunkResult,
    franka_model,
    franka_frame,
    plot_xlim: tuple[float, float] | None,
    plot_ylim: tuple[float, float] | None,
):
    robot_state = env.robot.panda.get_state()
    q_curr = np.asarray(robot_state.q, dtype=np.float64).reshape(7)
    F_T_EE = np.asarray(robot_state.F_T_EE, dtype=np.float32).reshape(16).tolist()
    EE_T_K = np.asarray(robot_state.EE_T_K, dtype=np.float32).reshape(16).tolist()
    if franka_model is None or franka_frame is None:
        franka_model = env.robot.panda.get_robot().load_model()
        franka_frame = libfranka.Frame.kEndEffector

    if result.topk_centers is not None and result.topk_probs is not None:
        topk_xyz = []
        for center in result.topk_centers:
            q_next, _ = _sanitize_joint_target(args, q_curr, center[:7])
            pose = franka_model.pose(franka_frame, q_next.tolist(), F_T_EE, EE_T_K)
            pose = np.asarray(pose, dtype=np.float32).reshape(4, 4).T
            topk_xyz.append(pose[:3, 3])
        topk_xyz = np.asarray(topk_xyz, dtype=np.float32)
        curr_pose = np.asarray(robot_state.O_T_EE, dtype=np.float32).reshape(4, 4).T
        curr_ee_xyz = curr_pose[:3, 3]
        if plot_xlim is None or plot_ylim is None:
            xy = np.concatenate([topk_xyz[:, :2], curr_ee_xyz[:2].reshape(1, 2)], axis=0)
            xy_min = xy.min(axis=0)
            xy_max = xy.max(axis=0)
            span = np.maximum(xy_max - xy_min, np.array([0.15, 0.15], dtype=np.float32))
            pad = 0.25 * span
            plot_xlim = (float(xy_min[0] - pad[0]), float(xy_max[0] + pad[0]))
            plot_ylim = (float(xy_min[1] - pad[1]), float(xy_max[1] + pad[1]))
        plot_path = plot_topk_xy(
            topk_xyz,
            result.topk_probs,
            current_xyz=curr_ee_xyz,
            xlim=plot_xlim,
            ylim=plot_ylim,
        )
        reactive_desk_client.send_predictions(
            topk_xyz,
            result.topk_probs,
            prompt=instruction,
            is_executing=True,
        )
        print(f"  [TopK] plot saved: {plot_path}")
        return franka_model, franka_frame, plot_xlim, plot_ylim

    chunk_for_viz = result.action_chunk[result.rtc_skip :]
    if chunk_for_viz.shape[0] == 0:
        chunk_for_viz = result.action_chunk[-1:]
    chunk_xyz = []
    for chunk_action in chunk_for_viz:
        q_next, _ = _sanitize_joint_target(args, q_curr, chunk_action[:7])
        pose = franka_model.pose(franka_frame, q_next.tolist(), F_T_EE, EE_T_K)
        pose = np.asarray(pose, dtype=np.float32).reshape(4, 4).T
        chunk_xyz.append(pose[:3, 3])
    chunk_xyz = np.asarray(chunk_xyz, dtype=np.float32)
    chunk_probs = np.full(chunk_xyz.shape[0], 1.0 / max(chunk_xyz.shape[0], 1), dtype=np.float32)
    tqdm.write("send prediction")
    reactive_desk_client.send_predictions(
        chunk_xyz,
        chunk_probs,
        prompt=instruction,
        is_executing=True,
    )
    return franka_model, franka_frame, plot_xlim, plot_ylim


def _format_joint_position(qpos: np.ndarray) -> str:
    return np.array2string(
        np.asarray(qpos, dtype=np.float64),
        precision=5,
        separator=", ",
        suppress_small=False,
    )


def _move_robot_to_joint_pose(env: FrankaRobotEnv, target_qpos: tuple[float, ...]) -> None:
    qpos = np.asarray(target_qpos, dtype=np.float64).flatten()
    if qpos.shape != (7,):
        raise ValueError(f"target_qpos must be 7D, got shape {qpos.shape}")
    if getattr(env.robot, "auto_set_default_behavior", False):
        env.robot.panda.set_default_behavior()
    print(f"[Control] Moving to custom joint pose: {_format_joint_position(qpos)}")
    env.robot.panda.move_to_joint_position(qpos, speed_factor=env.robot.joint_speed_factor)


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
    _move_robot_to_joint_pose(env, CUSTOM_START_JOINT_POSITION)
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
    reactive_desk_client = None
    policy_executor = None
    try:
        controller, gripper_state = _reset_robot(env, controller, settle_s=args.controller_settle_s)

        franka_model = None
        franka_frame = None
        if args.topk_samples is not None and args.topk_samples > 1:
            franka_model = env.robot.panda.get_robot().load_model()
            franka_frame = libfranka.Frame.kEndEffector
        plot_xlim: tuple[float, float] | None = None
        plot_ylim: tuple[float, float] | None = None

        policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
        policy_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-policy")
        reactive_desk_client = ReactiveDeskVlaClient(
            host=args.reactive_desk_host,
            port=args.reactive_desk_port,
            path=args.reactive_desk_path,
            enabled=args.reactive_desk_enabled,
        )
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
            pending_policy_future: concurrent.futures.Future[_PolicyChunkResult] | None = None
            pending_switch_step = 0

            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")  # noqa: DTZ005
            video = []
            steps_completed = 0
            total_infer_time_s = 0.0
            total_request_build_time_s = 0.0
            total_infer_calls = 0
            last_infer_time_s = 0.0
            last_request_build_time_s = 0.0
            last_rtc_skip = 0
            print("Running rollout... press Ctrl+C to stop early.")
            progress = tqdm(
                total=args.max_timesteps,
                desc="rollout",
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )

            while steps_completed < args.max_timesteps:
                start_time = time.time()
                try:
                    # panda-py controllers run asynchronously; surface controller aborts on this thread.
                    env.robot.panda.raise_error()
                    curr_obs = _extract_observation(
                        args,
                        env,
                        gripper_state,
                        save_to_disk=steps_completed == 0,
                    )

                    external_frame = curr_obs[f"{args.external_camera}_image"]
                    wrist_frame = curr_obs["wrist_image"]
                    video.append(np.concatenate([external_frame, wrist_frame], axis=1))

                    topk_k = int(args.topk_samples) if args.topk_samples is not None else 0
                    topk_draws = int(args.topk_draws) if args.topk_draws is not None else None

                    if pred_action_chunk is None:
                        pending_policy_future = policy_executor.submit(
                            _run_policy_request_from_observation,
                            policy_client,
                            args,
                            instruction,
                            curr_obs,
                            previous_chunk=None,
                            prefix_start=0,
                            delay=0,
                            topk_k=topk_k,
                            topk_draws=topk_draws,
                        )
                        with prevent_keyboard_interrupt():
                            policy_result = pending_policy_future.result()
                        pending_policy_future = None
                        total_infer_time_s += policy_result.infer_time_s
                        total_request_build_time_s += policy_result.request_build_time_s
                        total_infer_calls += policy_result.infer_calls
                        last_infer_time_s = policy_result.infer_time_s
                        last_request_build_time_s = policy_result.request_build_time_s
                        last_rtc_skip = policy_result.rtc_skip
                        pred_action_chunk = policy_result.action_chunk
                        pred_action_horizon = pred_action_chunk.shape[0]
                        actions_from_chunk_completed = 0
                        tqdm.write(f"[chunk] horizon={pred_action_horizon} rtc_skip={policy_result.rtc_skip}")
                        franka_model, franka_frame, plot_xlim, plot_ylim = _send_prediction_visualization(
                            args=args,
                            env=env,
                            reactive_desk_client=reactive_desk_client,
                            instruction=instruction,
                            result=policy_result,
                            franka_model=franka_model,
                            franka_frame=franka_frame,
                            plot_xlim=plot_xlim,
                            plot_ylim=plot_ylim,
                        )

                    effective_horizon = min(args.open_loop_horizon, pred_action_horizon or args.open_loop_horizon)
                    if pending_policy_future is None and actions_from_chunk_completed >= effective_horizon:
                        rtc_prefix_start = actions_from_chunk_completed
                        rtc_skip = _rtc_skip_for_chunk(
                            pred_action_chunk,
                            rtc_prefix_start,
                            args.rtc_inference_delay,
                        )
                        pending_policy_future = policy_executor.submit(
                            _run_policy_request_from_observation,
                            policy_client,
                            args,
                            instruction,
                            curr_obs,
                            previous_chunk=pred_action_chunk.copy(),
                            prefix_start=rtc_prefix_start,
                            delay=args.rtc_inference_delay,
                            topk_k=topk_k,
                            topk_draws=topk_draws,
                        )
                        pending_switch_step = rtc_prefix_start + rtc_skip

                    if pending_policy_future is not None:
                        switch_step = min(pending_switch_step, pred_action_horizon)
                        if actions_from_chunk_completed >= switch_step:
                            if not pending_policy_future.done():
                                tqdm.write("[RTC] waiting for pending policy result")
                            with prevent_keyboard_interrupt():
                                policy_result = pending_policy_future.result()
                            pending_policy_future = None
                            total_infer_time_s += policy_result.infer_time_s
                            total_request_build_time_s += policy_result.request_build_time_s
                            total_infer_calls += policy_result.infer_calls
                            last_infer_time_s = policy_result.infer_time_s
                            last_request_build_time_s = policy_result.request_build_time_s
                            last_rtc_skip = policy_result.rtc_skip
                            pred_action_chunk = policy_result.action_chunk
                            pred_action_horizon = pred_action_chunk.shape[0]
                            actions_from_chunk_completed = min(policy_result.rtc_skip, max(pred_action_horizon - 1, 0))
                            tqdm.write(f"[chunk] horizon={pred_action_horizon} rtc_skip={policy_result.rtc_skip}")
                            franka_model, franka_frame, plot_xlim, plot_ylim = _send_prediction_visualization(
                                args=args,
                                env=env,
                                reactive_desk_client=reactive_desk_client,
                                instruction=instruction,
                                result=policy_result,
                                franka_model=franka_model,
                                franka_frame=franka_frame,
                                plot_xlim=plot_xlim,
                                plot_ylim=plot_ylim,
                            )

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
                    env.robot.panda.raise_error()

                    steps_completed += 1
                    elapsed_time = time.time() - start_time
                    progress.update(1)
                    avg_infer_ms = (total_infer_time_s / total_infer_calls * 1e3) if total_infer_calls else 0.0
                    progress.set_postfix(
                        {
                            "step_ms": f"{elapsed_time * 1e3:.0f}",
                            "infer_ms": f"{last_infer_time_s * 1e3:.0f}",
                            "build_ms": f"{last_request_build_time_s * 1e3:.0f}",
                            "avg_infer_ms": f"{avg_infer_ms:.0f}",
                            "infers": total_infer_calls,
                            "horizon": pred_action_horizon,
                            "rtc_skip": last_rtc_skip,
                        },
                        refresh=False,
                    )
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

            _finish_pending_policy_future(pending_policy_future)
            pending_policy_future = None
            progress.close()
            if total_infer_calls:
                print(
                    f"[Timing] steps={steps_completed} infers={total_infer_calls} "
                    f"request_build_total={total_request_build_time_s:.2f}s "
                    f"avg_build={total_request_build_time_s / total_infer_calls * 1e3:.0f}ms "
                    f"infer_total={total_infer_time_s:.2f}s "
                    f"avg_infer={total_infer_time_s / total_infer_calls * 1e3:.0f}ms"
                )

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
            if controller is not None:
                _hold_current_joint_position(env, controller)
        with contextlib.suppress(Exception):
            env.robot.panda.stop_controller()
        with contextlib.suppress(Exception):
            env.close()
        with contextlib.suppress(Exception):
            if policy_executor is not None:
                policy_executor.shutdown(wait=False, cancel_futures=True)
        with contextlib.suppress(Exception):
            if reactive_desk_client is not None:
                reactive_desk_client.close()


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

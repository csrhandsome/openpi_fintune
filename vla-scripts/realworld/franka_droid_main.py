from __future__ import annotations

import contextlib
import dataclasses
import datetime
import os
import signal
import time

import numpy as np
import pandas as pd
import tyro
from moviepy import ImageSequenceClip
from openpi_client import websocket_client_policy
from panda_py import libfranka
from PIL import Image

from franka_robot_env import FrankaCameraIds, FrankaRobotEnv
from plot_topk import plot_topk_xy


def _center_crop_and_resize(rgb: np.ndarray, out_hw: int = 224, crop_scale: float = 0.9) -> np.ndarray:
    """Center crop + resize to match training data preprocessing."""
    h, w = rgb.shape[:2]
    frac = float(np.sqrt(crop_scale))
    crop_h = max(1, min(h, int(round(h * frac))))
    crop_w = max(1, min(w, int(round(w * frac))))
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    cropped = rgb[y0:y0 + crop_h, x0:x0 + crop_w]
    img = Image.fromarray(cropped, mode="RGB").resize((out_hw, out_hw), resample=Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Camera naming (must match keys produced by FrankaRobotEnv)
    left_camera_id: str = "external"
    right_camera_id: str = "right"
    wrist_camera_id: str = "wrist"

    # Which external camera should be fed to the policy, choose from ["left", "right"]
    external_camera: str | None = "left"

    # Rollout parameters
    max_timesteps: int = 800 
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

    # Control parameters
    joint_velocity_scale: float = 1.0  # 改为 1.0 匹配官方 DROID
    # Top-k clustering count (set to >1 to enable; None/<=1 disables)
    topk_samples: int | None = None
    # Number of stochastic policy draws used before clustering.
    # If None, defaults to max(4 * topk_samples, topk_samples + 1).
    topk_draws: int | None = None


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


def main(args: Args):
    assert args.external_camera in ["left", "right"], (
        "Please specify an external camera to use for the policy, choose from ['left', 'right']"
    )

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
        control_frequency_hz=DROID_CONTROL_FREQUENCY,
        joint_velocity_scale=args.joint_velocity_scale,
    )

    franka_model = None
    franka_frame = None
    if args.topk_samples is not None and args.topk_samples > 1:
        franka_model = env._robot.panda.get_robot().load_model()
        franka_frame = libfranka.Frame.kEndEffector
    plot_xlim: tuple[float, float] | None = None
    plot_ylim: tuple[float, float] | None = None

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        # instruction = input("Enter instruction: ")
        instruction = "pick up the yellow cup"
        # instruction = "Pick up the wide-mouth bottle"
        plot_xlim = None
        plot_ylim = None
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        pred_action_horizon = 0

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in range(args.max_timesteps):
            start_time = time.time()
            try:
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    save_to_disk=t_step == 0,
                )

                external_frame = curr_obs[f"{args.external_camera}_image"]
                wrist_frame = curr_obs["wrist_image"]
                video.append(np.concatenate([external_frame, wrist_frame], axis=1))

                effective_horizon = min(args.open_loop_horizon, pred_action_horizon or args.open_loop_horizon)
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

                    # 聚类的个数，有几个无法描述准确的物体就(核心创新点)
                    topk_k = int(args.topk_samples) if args.topk_samples is not None else 0
                    with prevent_keyboard_interrupt():
                        if topk_k and topk_k > 1:
                            topk_draws = int(args.topk_draws) if args.topk_draws is not None else max(4 * topk_k, topk_k + 1)
                            topk_draws = max(topk_draws, topk_k)
                            result = policy_client.infer(request_data)
                            pred_action_chunk = result["actions"]
                            # 创一个列表放一个action
                            samples = [pred_action_chunk[0]]
                            # 额外采样topk_draws次再聚类(核心创新点)
                            for _ in range(topk_draws - 1):
                                samples.append(policy_client.infer(request_data)["actions"][0])
                            centers, probs = _kmeans_topk(np.asarray(samples, dtype=np.float32), k=topk_k)

                            # 不阻塞的get_state()，然后用官方的api转换成末端的xyz
                            robot_state = env._robot.panda.get_state()
                            q_curr = np.asarray(robot_state.q, dtype=np.float32).reshape(7)
                            F_T_EE = np.asarray(robot_state.F_T_EE, dtype=np.float32).reshape(16).tolist()
                            EE_T_K = np.asarray(robot_state.EE_T_K, dtype=np.float32).reshape(16).tolist()

                            topk_xyz = []
                            for center in centers:
                                action = np.asarray(center, dtype=np.float32).flatten()
                                action_clipped = np.clip(action[:7], -1.0, 1.0)
                                joint_delta = action_clipped * env._max_joint_delta * env._joint_velocity_scale
                                q_next = q_curr + joint_delta
                                pose = franka_model.pose(franka_frame, q_next.tolist(), F_T_EE, EE_T_K)
                                pose = np.asarray(pose, dtype=np.float32).reshape(4, 4).T
                                topk_xyz.append(pose[:3, 3])
                            topk_xyz = np.asarray(topk_xyz, dtype=np.float32)
                            # 把末端的xy换成图片保存
                            curr_pose = np.asarray(robot_state.O_T_EE, dtype=np.float32).reshape(4, 4).T
                            curr_ee_xyz = curr_pose[:3, 3]
                            # Freeze XY axis on first plot to avoid per-frame auto-rescaling jitter.
                            if plot_xlim is None or plot_ylim is None:
                                xy = np.concatenate([topk_xyz[:, :2], curr_ee_xyz[:2].reshape(1,2)], axis=0)
                                # 确定坐标轴范围
                                xy_min = xy.min(axis=0)
                                xy_max = xy.max(axis=0)
                                span = np.maximum(xy_max - xy_min, np.array([0.15, 0.15], dtype=np.float32))
                                pad = 0.25 * span
                                plot_xlim = (float(xy_min[0] - pad[0]), float(xy_max[0] + pad[0]))
                                plot_ylim = (float(xy_min[1] - pad[1]), float(xy_max[1] + pad[1]))
                            plot_path = plot_topk_xy(
                                topk_xyz,
                                probs,
                                current_xyz=curr_ee_xyz,
                                xlim=plot_xlim,
                                ylim=plot_ylim,
                            )
                            print(f"  [TopK] plot saved: {plot_path}")
                        else:
                            pred_action_chunk = policy_client.infer(request_data)["actions"]

                    if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[1] < 8:
                        raise ValueError(f"Unexpected action chunk shape: {pred_action_chunk.shape}")
                    pred_action_chunk = pred_action_chunk[:, :8]
                    pred_action_horizon = pred_action_chunk.shape[0]

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                action = np.clip(action, -1, 1)
                env.step(action)

                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "success": success,
                            "duration": t_step,
                            "video_filename": save_filename,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    # timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    # csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    # df.to_csv(csv_filename, index=False)
    env._robot.move_to_start()
    env._robot.gripper_open()
    # print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    if left_image is None or wrist_image is None:
        raise RuntimeError("Missing required camera images (external/wrist).")
    if right_image is None:
        right_image = left_image

    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

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
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image

from franka_robot_env import FrankaCameraIds, FrankaRobotEnv

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
    max_timesteps: int = 600
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

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")

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
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    with prevent_keyboard_interrupt():
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
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


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

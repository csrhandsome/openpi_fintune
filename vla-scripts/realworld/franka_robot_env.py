from __future__ import annotations

from dataclasses import dataclass
from typing import Optional,TYPE_CHECKING

import numpy as np

from dual_camera_manager import DualCameraManager
from robotic_arm_controller import RoboticArmControler

if TYPE_CHECKING:
    from robotic_arm_controller import RoboticArmControler

@dataclass
class FrankaCameraIds:
    external: str = "external"
    right: str = "right"
    wrist: str = "wrist"


def _rgb_to_bgra(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got {rgb.shape}")
    bgr = rgb[..., ::-1]
    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([bgr, alpha], axis=-1)


class FrankaRobotEnv:
    """Franka adapter that matches the DROID RobotEnv interface."""

    def __init__(
        self,
        *,
        action_space: str = "joint_velocity",
        gripper_action_space: str = "position",
        franka_hostname: str = "192.168.1.100",
        franka_username: str = "your_username",
        franka_password: str = "your_password",
        camera_ids: FrankaCameraIds | None = None,
        external_camera_serial: Optional[str] = None,
        wrist_camera_serial: Optional[str] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        color_only: bool = False,
        use_external_for_right: bool = True,
        control_frequency_hz: float = 15.0,
        joint_velocity_scale: float = 0.5,
        gripper_open_threshold: float = 0.5,
        gripper_max_width_m: float = 0.08,
        camera_timeout_ms: int = 500,
        warmup_frames: int = 5,
        auto_move_to_start: bool = True,
    ) -> None:
        if action_space != "joint_velocity":
            raise ValueError("FrankaRobotEnv only supports action_space='joint_velocity'")
        if gripper_action_space != "position":
            raise ValueError("FrankaRobotEnv only supports gripper_action_space='position'")

        self._camera_ids = camera_ids or FrankaCameraIds()
        self._use_external_for_right = use_external_for_right
        self._camera_timeout_ms = int(camera_timeout_ms)
        self._control_frequency_hz = float(control_frequency_hz)
        # DROID 官方: max_joint_delta = 0.2 rad/step @ 15Hz
        # 转换为速度: 0.2 * 15 = 3.0 rad/s (当 action=1.0 时)
        self._max_joint_delta = 0.2  # rad per step, 匹配 DROID
        self._joint_velocity_scale = float(joint_velocity_scale)
        self._gripper_open_threshold = float(gripper_open_threshold)
        self._gripper_max_width_m = float(gripper_max_width_m)
        self._last_observation: dict | None = None
        self._last_gripper_open: Optional[bool] = None

        self._cameras = DualCameraManager(
            external_serial=external_camera_serial,
            wrist_serial=wrist_camera_serial,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            color_only=color_only,
        )
        self._cameras.connect()

        for _ in range(max(0, int(warmup_frames))):
            self._cameras.update(timeout_ms=self._camera_timeout_ms)

        self._robot: RoboticArmControler = RoboticArmControler(
            auto_move_to_start=auto_move_to_start,
        )
        if auto_move_to_start:
            self._robot.move_to_start()

        # 启动流式速度控制器
        # 注意：时间步 = 1 / control_frequency_hz，这决定了积分的步长
        self._robot.start_velocity_streaming(
            control_frequency=max(control_frequency_hz * 10, 200.0),
            time_step=1.0 / control_frequency_hz,
        )

    def _get_robot_state(self) -> dict:
        state = self._robot.state
        cartesian_position = np.asarray(state["cartesian_position"], dtype=np.float32).reshape(3)
        joint_positions = np.asarray(state["joint_positions"], dtype=np.float32)
        gripper_width = float(state["gripper_position"])
        gripper_position = np.clip(gripper_width / self._gripper_max_width_m, 0.0, 1.0)
        return {
            "cartesian_position": cartesian_position,
            "joint_positions": joint_positions,
            "gripper_position": gripper_position,
        }

    def get_observation(self) -> dict:
        external_ok, wrist_ok = self._cameras.update(timeout_ms=self._camera_timeout_ms)
        if not external_ok or not wrist_ok:
            if self._last_observation is None:
                raise RuntimeError("Camera update failed and no previous observation is available.")
            return self._last_observation

        external_img, wrist_img = self._cameras.get_images()
        if external_img is None or wrist_img is None:
            if self._last_observation is None:
                raise RuntimeError("Camera images missing and no previous observation is available.")
            return self._last_observation

        external_bgra = _rgb_to_bgra(external_img)
        wrist_bgra = _rgb_to_bgra(wrist_img)
        right_bgra = external_bgra if self._use_external_for_right else wrist_bgra

        images = {
            f"{self._camera_ids.external}_left": external_bgra,
            f"{self._camera_ids.right}_left": right_bgra,
            f"{self._camera_ids.wrist}_left": wrist_bgra,
        }

        observation = {
            "image": images,
            "robot_state": self._get_robot_state(),
        }
        self._last_observation = observation
        return observation

    def step(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape[0] != 8:
            raise ValueError(f"Expected action shape (8,), got {action.shape}")

        # DROID 官方转换: action [-1,1] -> joint_delta [-0.2, 0.2] rad
        # 然后转换为速度: joint_velocity = joint_delta * control_frequency
        action_clipped = np.clip(action[:7], -1.0, 1.0)
        joint_delta = action_clipped * self._max_joint_delta * self._joint_velocity_scale
        joint_velocity = joint_delta * self._control_frequency_hz  # 转换为 rad/s

        # 使用流式控制（无需频繁 start/stop 控制器）
        self._robot.apply_joint_velocity(joint_velocity, streaming=True)

        gripper_open = float(action[7]) > self._gripper_open_threshold
        if self._last_gripper_open is None or gripper_open != self._last_gripper_open:
            if gripper_open:
                self._robot.gripper_open()
            else:
                self._robot.gripper_close()
            self._last_gripper_open = gripper_open

    def reset(self) -> dict:
        # 停止流式控制器以避免控制冲突
        self._robot.stop_velocity_streaming()

        # 移动到起始位置
        self._robot.gripper_open()
        self._robot.move_to_start()

        # 重新启动流式速度控制器
        self._robot.start_velocity_streaming(
            control_frequency=max(self._control_frequency_hz * 10, 200.0),
            time_step=1.0 / self._control_frequency_hz,
        )

        return self.get_observation()

    def close(self) -> None:
        self._robot.stop_velocity_streaming()
        self._cameras.close()
        self._robot.cleanup()

    def __enter__(self) -> "FrankaRobotEnv":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self.close()
        return False

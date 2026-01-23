# 模块加载更快
from __future__ import annotations

import json
import os
import threading
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional, Sequence
import time
import h5py
import numpy as np
import panda_py
from spatialmath import SE3
from panda_py import libfranka

if TYPE_CHECKING:
    from realsense_connector import RealSenseConnector


class _LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._rgb: Optional[np.ndarray] = None

    def set(self, rgb: np.ndarray) -> None:
        with self._lock:
            self._seq += 1
            self._rgb = np.asarray(rgb, dtype=np.uint8)

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._rgb is None:
                return None
            return np.array(self._rgb, copy=True)


def _camera_capture_worker(
    *,
    camera: "RealSenseConnector",
    buf: _LatestFrameBuffer,
    stop_event: threading.Event,
    timeout_ms: int,
    crop_scale: float,
    out_hw: int,
) -> None:
    frame_count = 0
    error_count = 0
    while not stop_event.is_set():
        try:
            ok = camera.update(timeout=int(timeout_ms))
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"[Camera Thread] Update exception: {e}")
            ok = False
        if not ok:
            continue

        rgb = camera.img
        if rgb is None:
            continue
        try:
            rgb224 = _center_crop_and_resize_rgb_uint8(
                np.asarray(rgb, dtype=np.uint8), crop_scale=float(crop_scale), out_hw=int(out_hw)
            )
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"[Camera Thread] Crop/resize exception: {e}")
            continue
        buf.set(rgb224)
        frame_count += 1
        if frame_count == 1:
            print(f"[Camera Thread] First frame captured successfully!")
        elif frame_count % 100 == 0:
            print(f"[Camera Thread] {frame_count} frames captured")


def _center_crop_and_resize_rgb_uint8(rgb: np.ndarray, *, crop_scale: float = 0.9, out_hw: int = 224) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image of shape (H, W, 3), got {rgb.shape}")

    try:
        import tensorflow as tf

        # Match OpenVLAPolicy._crop_and_resize behavior (center crop by area scale, then resize to 224x224).
        image_tf = tf.convert_to_tensor(rgb)  # uint8 (H,W,3)
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)  # [0,1]
        image_tf = tf.expand_dims(image_tf, axis=0)  # (1,H,W,3)

        batch_size = 1
        scale = tf.reshape(tf.clip_by_value(tf.sqrt(float(crop_scale)), 0, 1), shape=(batch_size,))
        height_offsets = (1 - scale) / 2
        width_offsets = (1 - scale) / 2
        boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + scale,
                width_offsets + scale,
            ],
            axis=1,
        )

        image_tf = tf.image.crop_and_resize(image_tf, boxes, tf.range(batch_size), (int(out_hw), int(out_hw)))
        image_tf = image_tf[0]
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        out = image_tf.numpy()
    except Exception:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        frac = float(np.sqrt(float(crop_scale)))
        crop_h = max(1, min(h, int(round(h * frac))))
        crop_w = max(1, min(w, int(round(w * frac))))
        y0 = max(0, (h - crop_h) // 2)
        x0 = max(0, (w - crop_w) // 2)
        cropped = rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]

        from PIL import Image

        img = Image.fromarray(cropped, mode="RGB").resize((int(out_hw), int(out_hw)), resample=Image.BILINEAR)
        out = np.asarray(img, dtype=np.uint8)

    if out.shape != (int(out_hw), int(out_hw), 3):
        raise RuntimeError(f"Unexpected resized RGB shape: {out.shape}")
    return out.astype(np.uint8, copy=False)


class RoboticArmControler:
    """利用panda_py这个无敌的好库来帮助我控制机械臂"""

    def __init__(
        self,
        hostname: str = "192.168.1.100",
        username: str = "BionicDL-Franka",
        password: str = "Design@2018",
        cartesian_speed_factor: float = 0.2,
        joint_speed_factor: float = 0.21,
        default_height: Optional[float] = 0.118046,
        enforce_default_height: bool = False,
        stiffness: Optional[Sequence[float]] = (600, 600, 600, 600, 250, 150, 50),
        damping: Optional[Sequence[float]] = None,
        auto_set_default_behavior: bool = True,
        auto_move_to_start: bool = True,
    ) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            hostname = config.get("hostname", hostname)
            username = config.get("username", username)
            password = config.get("password", password)

        # 启动 FCI 模式
        self.desk = panda_py.Desk(hostname, username, password)
        self.desk.unlock()
        self.desk.activate_fci()
        print("FCI 模式已激活")

        # 启动机械臂
        self.panda = panda_py.Panda(hostname)
        self.gripper = libfranka.Gripper(hostname)
        self.auto_set_default_behavior = auto_set_default_behavior
        self.auto_move_to_start = auto_move_to_start

        # 初始化机械臂参数
        self.T_0 = SE3(self.panda.get_pose(), check=False)
        self.joint_speed_factor = joint_speed_factor
        self.cart_speed_factor = cartesian_speed_factor
        self.stiffness = list(stiffness) if stiffness is not None else None
        self.damping = list(damping) if damping is not None else None
        self.default_height = default_height
        self.enforce_default_height = enforce_default_height

        # 设置碰撞检测阈值
        collision_torque = [50.0, 50.0, 50.0, 50.0, 40.0, 30.0, 20.0]
        collision_force = [50.0, 50.0, 50.0, 40.0, 40.0, 40.0]
        self.panda.get_robot().set_collision_behavior(
            collision_torque,
            collision_torque,
            collision_force,
            collision_force,
        )

        self.log_dir = os.path.join(os.path.dirname(__file__), "../../data/hdf5")
        os.makedirs(self.log_dir, exist_ok=True)

        self.robot_state: dict[str, np.ndarray] = {}
        self.time: list[float] = []
        self._logging_started_at: Optional[datetime] = None
        self._last_log_filename: Optional[str] = None

        # 流式速度控制的持久化状态
        self._velocity_controller: Optional[object] = None
        self._velocity_ctx: Optional[object] = None
        self._velocity_control_thread: Optional[threading.Thread] = None
        self._velocity_control_stop = threading.Event()
        self._velocity_control_lock = threading.Lock()
        self._velocity_accel_time: float = 0.1  # 加速度时间
        self._velocity_control_freq: float = 200.0  # 控制频率
        self._velocity_time_step: float = 1.0 / 200.0  # 积分时间步
        self._last_velocity_target: Optional[np.ndarray] = None  # 上一次的速度目标
        self._velocity_ramp_start_time: Optional[float] = None  # 梯形斜坡开始时间

    def move_to_start(self) -> None:
        if self.auto_set_default_behavior:
            self.panda.set_default_behavior()
        if self.auto_move_to_start:
            self.panda.move_to_start()
        self.T_0 = SE3(self.panda.get_pose(), check=False)

    def move_to_pose(self, poses: list, cart_speed_factor=None, stiffness=None) -> None:
        """poses传的是世界坐标系的坐标,格式: [x, y, z, roll, pitch, yaw]"""
        initial_pose = SE3(self.panda.get_pose(), check=False)

        se3_poses = []
        for pose in poses:
            trans = SE3.Trans(pose[0], pose[1], pose[2])
            if len(pose) == 6:
                rot = SE3.RPY([pose[3], pose[4], pose[5]], order="xyz")
            else:
                rot = SE3.Rt(initial_pose.R, t=[0, 0, 0], check=False)
            se3_poses.append(trans * rot)

        if cart_speed_factor is None:
            cart_speed_factor = self.cart_speed_factor
        if stiffness is None:
            stiffness = self.stiffness
        self.panda.move_to_pose(se3_poses, speed_factor=cart_speed_factor)
        self.T_0 = SE3(self.panda.get_pose(), check=False)

    def move_to_joint_position(self, poses: list, joint_speed_factor=None, stiffness=None) -> None:
        """poses传的是世界坐标系的坐标,格式: [x, y, z, roll, pitch, yaw]"""
        initial_pose = SE3(self.panda.get_pose(), check=False)
        initial_rpy = initial_pose.rpy(order="xyz", unit="rad")

        def _normalize_angle(angle: float) -> float:
            return (angle + np.pi) % (2 * np.pi) - np.pi

        q_current = self.panda.q
        current_yaw = initial_rpy[2]

        qs = []
        for i, pose in enumerate(poses):
            x, y, z = pose[:3]
            trans = SE3.Trans(x, y, z)

            if len(pose) == 6:
                roll, pitch, yaw = pose[3:6]
                yaw_candidates = [yaw]
            else:
                roll, pitch = initial_rpy[0], initial_rpy[1]
                base_candidates = [current_yaw]
                offsets = [np.pi / 6, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
                for offset in offsets:
                    base_candidates.append(_normalize_angle(current_yaw + offset))
                    base_candidates.append(_normalize_angle(current_yaw - offset))
                yaw_candidates = []
                seen = set()
                for candidate in base_candidates:
                    key = round(candidate, 6)
                    if key not in seen:
                        seen.add(key)
                        yaw_candidates.append(candidate)

            best_q = None
            min_distance = float("inf")
            chosen_yaw = current_yaw

            for yaw_candidate in yaw_candidates:
                rot = SE3.RPY([roll, pitch, yaw_candidate], order="xyz")
                T = trans * rot
                all_solutions = panda_py.ik_full(T, q_init=q_current)
                for solution in all_solutions:
                    if np.any(np.isnan(solution)):
                        continue
                    distance = np.linalg.norm(solution - q_current)
                    if distance < min_distance:
                        min_distance = distance
                        best_q = solution
                        chosen_yaw = yaw_candidate

            if best_q is None:
                raise ValueError(
                    f"IK求解失败:目标位姿 {i} 无解或超出工作空间。\n"
                    f"目标位置: x={x:.3f}, y={y:.3f}, z={z:.3f}\n"
                    f"请检查目标位置是否在机械臂可达范围内。"
                )
            qs.append(best_q)
            q_current = best_q
            if len(pose) == 6:
                current_yaw = yaw
            else:
                current_yaw = chosen_yaw
        if joint_speed_factor is None:
            joint_speed_factor = self.joint_speed_factor
        if stiffness is None:
            stiffness = self.stiffness
        self.panda.move_to_joint_position(qs, speed_factor=joint_speed_factor)
        self.T_0 = SE3(self.panda.get_pose(), check=False)

    def gripper_grasp(
        self,
        width: float = 0.0,
        speed: float = 0.2,
        force: float = 10.0,
        epsilon_inner: float = 0.04,
        epsilon_outer: float = 0.04,
    ) -> bool:
        return self.gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)

    def gripper_move(self, width: float = 0.08, speed: float = 0.2) -> bool:
        return self.gripper.move(width, speed)

    def gripper_open(self, width: float = 0.05, speed: float = 0.2) -> bool:
        return self.gripper_move(width=width, speed=speed)

    def gripper_close(
        self,
        width: float = 0.0,
        speed: float = 0.2,
        force: float = 60.0,
        epsilon_inner: float = 0.04,
        epsilon_outer: float = 0.04,
    ) -> bool:
        return self.gripper_grasp(width, speed, force, epsilon_inner, epsilon_outer)

    def move_delta_pose(
        self,
        action: np.ndarray,
        joint_speed_factor: float = None,
        use_cartesian: bool = True,
        flip_axes: tuple = (False, False, False),  # (flip_x, flip_y, flip_z)
        delta_frame: Literal["ee", "base"] = "ee",
    ) -> None:
        """
        执行增量位姿移动（GA-DDPG 模型输出的 6D delta action）

        Args:
            action: 6D 增量向量 [dx, dy, dz, roll, pitch, yaw]，在末端坐标系下
            joint_speed_factor: 可选的速度因子
            use_cartesian: 是否使用笛卡尔空间移动（推荐True，避免关节大幅转动）
            flip_axes: 翻转哪些轴 (x, y, z)，用于坐标系对齐
            delta_frame: 增量动作所在坐标系；"ee" 表示末端坐标系右乘，"base" 表示基座/世界坐标系左乘
        """
        if any(flip_axes):
            action = action.copy()
            for i, flip in enumerate(flip_axes[:3]):
                if flip:
                    action[i] = -action[i]
            if flip_axes != (False, False, False):
                print(
                    f"[DEBUG] Flipped action: flip_axes={flip_axes}, result=[{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]"
                )

        current_pose = self.ee_pose_matrix
        delta_pose = self._unpack_action(action)
        # 计算目标位姿：
        # - 末端坐标系增量：target = current @ delta (右乘)
        # - 基座/世界坐标系增量：target = delta @ current (左乘)
        if delta_frame == "ee":
            target_pose = current_pose @ delta_pose
        elif delta_frame == "base":
            target_pose = delta_pose @ current_pose
        else:
            raise ValueError(f"delta_frame must be 'ee' or 'base', got {delta_frame!r}")

        if use_cartesian:
            cart_speed = joint_speed_factor if joint_speed_factor else self.cart_speed_factor
            self.panda.move_to_pose(target_pose, speed_factor=cart_speed)
            self.T_0 = SE3(self.panda.get_pose(), check=False)
        else:
            target_se3 = SE3(target_pose, check=False)
            position = target_se3.t
            rpy = target_se3.rpy(order="xyz", unit="rad")
            target_6d = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(rpy[0]),
                float(rpy[1]),
                float(rpy[2]),
            ]
            self.move_to_joint_position([target_6d], joint_speed_factor=joint_speed_factor)

    def _unpack_action(self, action: np.ndarray) -> np.ndarray:
        from transforms3d.euler import euler2mat

        action = np.asarray(action, dtype=np.float64).flatten()
        if action.shape[0] != 6:
            raise ValueError(f"action must be 6D, got shape {action.shape}")

        pose_delta = np.eye(4, dtype=np.float32)
        pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
        pose_delta[:3, 3] = action[:3]

        return pose_delta

    def apply_joint_velocity(
        self,
        joint_velocity: np.ndarray,
        duration: float = 0.01,
        acceleration_time: float = 0.01,
        *,
        control_frequency: float = 200.0,
        max_abs_velocity: float | None = None,
        streaming: bool = False,
    ) -> None:
        """
        关节速度控制 - 支持单次执行或流式持续控制

        Args:
            joint_velocity: 7维关节速度 (rad/s)
            duration: 执行时长（秒），仅在 streaming=False 时有效
            acceleration_time: 加速时间（秒）
            control_frequency: 控制频率 (Hz)，默认 200Hz
            max_abs_velocity: 最大速度限幅（可选）
            streaming:
                - False (默认): 单次执行，立即返回
                - True: 启动持久化控制器，后续调用只需更新速度命令

        使用方式:
            # 单次执行（原来的用法）
            arm.apply_joint_velocity(velocity, duration=0.066)

            # 流式控制（高频调用）
            arm.start_velocity_streaming(control_frequency=200.0)
            for _ in range(1000):
                arm.apply_joint_velocity(velocity, streaming=True)
            arm.stop_velocity_streaming()
        """
        joint_velocity = np.asarray(joint_velocity, dtype=np.float64).flatten()
        if joint_velocity.shape[0] != 7:
            raise ValueError(f"Expected 7-DoF joint velocity, got {joint_velocity.shape}")

        if max_abs_velocity is not None:
            joint_velocity = np.clip(joint_velocity, -max_abs_velocity, max_abs_velocity)

        if streaming:
            # 流式模式：只更新控制命令
            self._update_velocity_command(joint_velocity)
        else:
            # 单次执行模式
            self._execute_velocity_once(joint_velocity, duration, acceleration_time, control_frequency)

    def _execute_velocity_once(
        self,
        joint_velocity: np.ndarray,
        duration: float,
        acceleration_time: float,
        control_frequency: float,
    ) -> None:
        """单次执行速度控制（不持久化）"""
        from panda_py import controllers

        # 防止已有控制器冲突（不影响持久化的流式控制器）
        if self._velocity_controller is None:
            self.panda.stop_controller()

        ctrl = controllers.IntegratedVelocity()
        self.panda.start_controller(ctrl)

        duration = max(0.0, float(duration))
        accel = max(0.0, float(acceleration_time))

        try:
            with self.panda.create_context(frequency=control_frequency, max_runtime=duration) as ctx:
                while ctx.ok():
                    t = ctx.time
                    if accel > 0.0 and duration > 0.0:
                        # 梯形速度曲线：起始加速、末尾减速
                        ramp_up = min(1.0, t / accel)
                        ramp_down = min(1.0, max(0.0, (duration - t) / accel))
                        scale = min(ramp_up, ramp_down)
                    else:
                        scale = 1.0

                    ctrl.set_control(joint_velocity * scale)
        finally:
            # 结束时清零
            ctrl.set_control(np.zeros(7))
            self.panda.stop_controller()

    def start_velocity_streaming(
        self,
        control_frequency: float = 200.0,
        acceleration_time: float = 0.1,
        time_step: float | None = None,
    ) -> None:
        """
        启动持久化速度控制器（用于高频流式调用）

        Args:
            control_frequency: 控制循环频率 (Hz)
            acceleration_time: 加速/减速时间（秒），用于梯形速度曲线
            time_step: 积分时间步 (秒)。如果提供，每个速度命令会乘以这个值来匹配原来的积分行为
                      例如：原来 duration=0.066s，现在应该传 time_step=0.066

        Example:
            arm.start_velocity_streaming(control_frequency=200.0, time_step=0.066)
            try:
                for _ in range(1000):
                    arm.apply_joint_velocity(vel, streaming=True)
            finally:
                arm.stop_velocity_streaming()
        """
        if self._velocity_controller is not None:
            print("[Warning] Velocity streaming already started, stopping first...")
            self.stop_velocity_streaming()

        from panda_py import controllers

        self.panda.stop_controller()
        self._velocity_controller = controllers.IntegratedVelocity()
        self.panda.start_controller(self._velocity_controller)

        # 初始化为零速度
        self._velocity_controller.set_control(np.zeros(7))

        # 保存加速度参数和时间步
        self._velocity_accel_time = max(0.0, float(acceleration_time))
        self._velocity_control_freq = max(1.0, float(control_frequency))
        self._velocity_time_step = float(time_step) if time_step is not None else (1.0 / self._velocity_control_freq)
        self._last_velocity_target = np.zeros(7)
        self._velocity_ramp_start_time = None
        self._velocity_control_stop.clear()

        print(
            f"[Velocity] 已启动持续速度控制器 (freq={control_frequency:.0f}Hz, accel_time={acceleration_time:.3f}s, time_step={self._velocity_time_step:.4f}s)"
        )

    def _update_velocity_command(self, joint_velocity: np.ndarray) -> None:
        """在流式模式下更新速度命令"""
        if self._velocity_controller is None:
            raise RuntimeError(
                "Velocity controller not started. Call start_velocity_streaming() first or use streaming=False."
            )

        # 应用时间步缩放，保持原来的积分行为
        # 原来：target_q = current_q + joint_velocity * duration
        # 现在：速度 * time_step 相当于原来的位置增量
        scaled_velocity = joint_velocity * self._velocity_time_step

        with self._velocity_control_lock:
            try:
                self._velocity_controller.set_control(scaled_velocity)
            except Exception as e:
                print(f"[Error] Failed to set velocity command: {e}")
                raise

    def set_velocity_command(self, joint_velocity: np.ndarray) -> None:
        """
        快速设置速度命令（不加锁，性能更高，用于流式控制）

        需要先调用 start_velocity_streaming()
        """
        joint_velocity = np.asarray(joint_velocity, dtype=np.float64).flatten()
        if joint_velocity.shape[0] != 7:
            raise ValueError(f"Expected 7-DoF joint velocity, got {joint_velocity.shape}")

        if self._velocity_controller is None:
            raise RuntimeError("Velocity controller not started. Call start_velocity_streaming() first.")

        self._velocity_controller.set_control(joint_velocity)

    def stop_velocity_streaming(self) -> None:
        """停止持久化速度控制器"""
        with self._velocity_control_lock:
            if self._velocity_controller is None:
                return

            try:
                # 清零速度
                self._velocity_controller.set_control(np.zeros(7))
                self.panda.stop_controller()
                self._velocity_controller = None
                print("[Velocity] 已停止持续速度控制器")
            except Exception as e:
                print(f"[Error] Error stopping velocity controller: {e}")
                raise

    @property
    def state(self):
        """
        获取机器人状态。

        注意：当流式速度控制器运行时，使用 get_state() 返回缓存快照，
        而不是 read_once() 来避免阻塞冲突。
        """
        if self._velocity_controller is not None:
            # 流式控制运行中，用缓存状态
            robot_state = self.panda.get_state()
        else:
            # 无控制器运行，可以直接读取最新状态
            robot_state = self.panda.get_robot().read_once()

        return {
            "joint_positions": robot_state.q,
            "joint_velocities": robot_state.dq,
            "end_effector_pose": robot_state.O_T_EE,
            "joint_torques": robot_state.tau_J,
            "cartesian_position": robot_state.O_T_EE[:3, 3] if hasattr(robot_state.O_T_EE, 'shape') else np.zeros(3),
            "gripper_position": self.gripper.read_once().width if hasattr(self, 'gripper') else 0.0,
        }

    @property
    def ee_pose_matrix(self):
        robot_state = self.panda.get_robot().read_once()
        pose_matrix = np.array(robot_state.O_T_EE).reshape(4, 4).T
        return pose_matrix.astype(np.float32)

    @property
    def pose(self):
        robot_state = self.panda.get_robot().read_once()
        pose_matrix = np.array(robot_state.O_T_EE).reshape(4, 4).T
        T = SE3(pose_matrix, check=False)
        position = T.t
        rpy = T.rpy(order="xyz")
        return np.concatenate([position, rpy])

    def __del__(self):
        self.cleanup()

    def cleanup(self) -> None:
        with suppress(Exception):
            self.panda.disable_logging()
        with suppress(Exception):
            self.panda.stop_controller()
        with suppress(Exception):
            self.desk.stop_listen()
        with suppress(Exception):
            self.desk.deactivate_fci()
        with suppress(Exception):
            self.desk.release_control()
        with suppress(Exception):
            self.desk.logout()

    def manual_control(
        self,
        control_mode: Literal["keyboard", "gamepad", "pilot", "teleop"] = "keyboard",
        control_frequency: float = 100.0,
        translation_speed: float = 0.001,
        rotation_speed: float = 0.01,
        use_impedance_control: bool = True,
        enable_logging: bool = True,
        max_logging_duration: float = 3600.0,
        *,
        camera: Optional["RealSenseConnector"] = None,
        camera_timeout_ms: int = 1000,
        language_instruction: str = "",
    ) -> None:
        """
        手动控制机械臂（支持键盘、游戏手柄、Pilot按钮或远程遥操作）

        Args:
            control_mode: 控制模式 - "keyboard"（键盘）、"gamepad"（游戏手柄）、"pilot"（Pilot按钮）、"teleop"（远程遥操作）
            control_frequency: 控制循环频率（Hz）
            translation_speed: 平移速度（米/步）
            rotation_speed: 旋转速度（弧度/步）
            use_impedance_control: 是否使用阻抗控制（更平滑，推荐）
            enable_logging: 是否启用状态记录（用于数据采集）
            max_logging_duration: 最大记录时长（秒），防止内存溢出
            camera: RealSenseConnector 实例（可选，用于同步采集图像）
            camera_timeout_ms: 相机采集线程的 update 超时（毫秒）
            language_instruction: 本次演示/采集的语言指令（写入 HDF5 的 task.language_instruction）

        键盘控制说明:
            W/S: X轴前进/后退
            A/D: Y轴左右移动
            Q/E: Z轴上升/下降
            I/K: 绕X轴旋转（Roll）
            J/L: 绕Y轴旋转（Pitch）
            U/O: 绕Z轴旋转（Yaw）
            Space: 夹爪张开
            Enter: 夹爪闭合
            R: 回到起始位置
            ESC: 退出控制

        游戏手柄控制说明:
            左摇杆: X-Y平面移动
            右摇杆: Z轴移动 + Yaw旋转
            L1/R1: Roll旋转
            L2/R2: Pitch旋转
            X按钮: 夹爪张开
            O按钮: 夹爪闭合
            Start: 回到起始位置
            Select: 退出控制

        Pilot按钮控制说明:
            方向键: X-Y平面移动
            circle: 夹爪张开
            cross: 夹爪闭合
            check: 回到起始位置

        数据记录说明:
            如果启用logging，控制结束后可以通过 panda.get_log() 获取真实状态数据
        """
        print(f"\n{'='*60}")
        print(f"启动手动控制模式: {control_mode}")
        if use_impedance_control and enable_logging:
            print(f"阻抗控制 + 状态记录（推荐用于数据采集）")
        elif use_impedance_control:
            print(f"阻抗控制（平滑但不记录状态）")
        else:
            print(f"离散控制（实时读取状态但不平滑）")
        print(f"{'='*60}\n")

        if enable_logging and control_mode in ("keyboard", "gamepad", "teleop"):
            if camera is None:
                raise ValueError(
                    "enable_logging=True requires a RealSenseConnector instance (camera=...) for keyboard/gamepad/teleop."
                )
            if not str(language_instruction).strip():
                raise ValueError("enable_logging=True requires a non-empty language_instruction.")

        if control_mode == "keyboard":
            self._keyboard_control(
                control_frequency,
                translation_speed,
                rotation_speed,
                use_impedance_control,
                enable_logging,
                max_logging_duration,
                camera=camera,
                camera_timeout_ms=camera_timeout_ms,
                language_instruction=language_instruction,
            )
        elif control_mode == "gamepad":
            self._gamepad_control(
                control_frequency,
                translation_speed,
                rotation_speed,
                use_impedance_control,
                enable_logging,
                max_logging_duration,
                camera=camera,
                camera_timeout_ms=camera_timeout_ms,
                language_instruction=language_instruction,
            )
        elif control_mode == "pilot":
            self._pilot_control(control_frequency, translation_speed, rotation_speed)
        elif control_mode == "teleop":
            self._teleop_control(
                control_frequency,
                translation_speed,
                rotation_speed,
                use_impedance_control,
                enable_logging,
                max_logging_duration,
                camera=camera,
                camera_timeout_ms=camera_timeout_ms,
                language_instruction=language_instruction,
            )
        else:
            raise ValueError(f"Unsupported control mode: {control_mode}")


    def _teleop_control(
        self,
        control_frequency: float,
        translation_speed: float,
        rotation_speed: float,
        use_impedance_control: bool,
        enable_logging: bool,
        max_logging_duration: float,
        *,
        camera: Optional["RealSenseConnector"],
        camera_timeout_ms: int,
        language_instruction: str,
    ) -> None:
        """远程遥操作控制实现（基于 robot_teleoperator）"""
        import os

        try:
            from robot_teleoperator import RobotTeleoperator
        except ImportError:
            raise ImportError("请确保 robot_teleoperator.py 和 teleop.py 在正确的路径下")

        from panda_py import controllers

        # 初始化 teleop 服务器
        teleop = RobotTeleoperator()
        teleop.start()

        running = [True]  # 使用列表以便在回调中修改
        manual_stop = [False]  # 用于检测用户是否手动停止

        print("远程遥操作控制已启动！")
        print("控制说明：")
        print("  通过手机/平板进行遥操作")
        print("  按下手机端的 reservedButtonA：停止记录并回到起始位置")
        print("  或按 Ctrl+C 直接停止记录并退出")
        print("\n[Recording] 掐头去尾模式：检测到第一个移动开始记录，按 reservedButtonA 后结束\n")

        if use_impedance_control:
            # 使用实时阻抗控制
            from transforms3d.quaternions import mat2quat, quat2mat, qmult
            from transforms3d.euler import euler2mat, quat2euler
            from transforms3d.quaternions import qinverse
            import scipy.spatial.transform as stf

            ctrl = controllers.CartesianImpedance(filter_coeff=1.0)

            log_file = None
            h5_file = None
            image_ds = None
            action_ds = None
            instr_ds = None

            camera_stop = None
            camera_thread = None
            camera_buf = None

            if enable_logging and hasattr(self, "log_dir"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(self.log_dir, f"manual_control_{timestamp}.h5")
                h5_file = h5py.File(log_file, "w")
                self._last_log_filename = log_file

                h5_file.attrs["created_at"] = timestamp
                h5_file.attrs["control_mode"] = "teleop"
                h5_file.attrs["control_frequency_hz"] = float(control_frequency)
                h5_file.attrs["translation_speed"] = float(translation_speed)
                h5_file.attrs["rotation_speed"] = float(rotation_speed)
                h5_file.attrs["action_type"] = "EEF_POS"
                h5_file.attrs["image_center_crop_scale"] = 0.9
                h5_file.attrs["image_size_hw"] = 224

                obs_group = h5_file.require_group("observation")
                task_group = h5_file.require_group("task")
                image_ds = obs_group.create_dataset(
                    "image_primary",
                    shape=(0, 224, 224, 3),
                    maxshape=(None, 224, 224, 3),
                    chunks=(1, 224, 224, 3),
                    dtype=np.uint8,
                    compression="lzf",
                )
                action_ds = h5_file.create_dataset(
                    "action",
                    shape=(0, 7),
                    maxshape=(None, 7),
                    chunks=(256, 7),
                    dtype=np.float32,
                    compression="lzf",
                )
                instr_ds = task_group.create_dataset(
                    "language_instruction",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(256,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )

                if camera is None:
                    raise ValueError("Teleop control logging requires camera=RealSenseConnector().")

                camera_buf = _LatestFrameBuffer()
                camera_stop = threading.Event()
                camera_thread = threading.Thread(
                    target=_camera_capture_worker,
                    kwargs={
                        "camera": camera,
                        "buf": camera_buf,
                        "stop_event": camera_stop,
                        "timeout_ms": int(camera_timeout_ms),
                        "crop_scale": 0.9,
                        "out_hw": 224,
                    },
                    daemon=True,
                )
                camera_thread.start()

                # ✅ 等待相机线程采集到第一帧
                print("[Camera] Waiting for first frame...")
                max_wait_time = 5.0
                wait_start = time.time()
                while camera_buf.get_latest() is None:
                    if time.time() - wait_start > max_wait_time:
                        raise RuntimeError(
                            "Camera timeout: Failed to get first frame after 5 seconds. "
                            "Please check camera connection."
                        )
                    time.sleep(0.01)
                print("[Camera] First frame acquired, ready to record!")

            img_buf: list[np.ndarray] = []
            act_buf: list[np.ndarray] = []
            instr_buf: list[str] = []
            flush_every = 20

            # ✅ 记录控制：只有检测到移动才开始记录
            recording_started = False
            recording_stopped = False

            # ✅ 在启动控制器前读取初始位姿
            target_position = self.panda.get_position().astype(np.float64)
            target_orientation = self.panda.get_orientation().astype(np.float64)  # 四元数 [w, x, y, z]
            gripper_state = 1.0
            last_gripper_cmd = 1.0  # 记录上一次的夹爪命令

            # ✅ 零点偏移：记录手机姿态的基准点（当 move 从 False 变为 True 时记录）
            phone_base_pose = None  # 4x4 变换矩阵，作为手机的"零点"
            last_is_moving = False  # 记录上一次的 is_moving 状态

            # 保存初始位姿到 HDF5（用于回放验证）
            if h5_file is not None:
                h5_file.attrs["initial_position"] = target_position.tolist()
                h5_file.attrs["initial_orientation"] = target_orientation.tolist()
                print(f"[Data] Initial position: {target_position}")
                print(f"[Data] Initial orientation: {target_orientation}")

            self.panda.start_controller(ctrl)

            # 标志：是否发生了 reflex 错误（用于决定是否保存数据）
            reflex_error_occurred = False

            try:
                with self.panda.create_context(frequency=control_frequency) as ctx:
                    while ctx.ok() and running[0]:
                        # 从 teleop 获取最新动作
                        action = teleop.get_latest_action()

                        if action is None:
                            # 没有接收到动作，保持当前位置
                            continue

                        # 解析 teleop 动作
                        pose_matrix = action["pose"]  # 4x4 变换矩阵
                        gripper_cmd = float(action.get("gripper", 1.0))  # 0=close, 1=open
                        is_moving = action.get("moving", False)  # 是否正在移动
                        stop_recording = action.get("stop_recording", False)  # 是否停止记录

                        # ✅ 零点重置：当 move 从 False 变为 True 时，记录当前手机姿态作为基准
                        if is_moving and not last_is_moving:
                            phone_base_pose = pose_matrix.copy()
                            print(f"\n[Teleop] 检测到 move=True，已重置手机零点基准")
                            print(
                                f"  基准位置: [{phone_base_pose[0,3]:.3f}, {phone_base_pose[1,3]:.3f}, {phone_base_pose[2,3]:.3f}]"
                            )

                        last_is_moving = is_moving

                        # 检查是否收到停止信号
                        if stop_recording and recording_started:
                            print("\n[Recording] 收到停止信号，停止记录并回到起始位置...")
                            recording_stopped = True  # 停止记录
                            self.panda.stop_controller()
                            self.move_to_start()
                            target_position = self.panda.get_position().astype(np.float64)
                            target_orientation = self.panda.get_orientation().astype(np.float64)
                            self.panda.start_controller(ctrl)
                            running[0] = False  # 结束控制循环
                            continue

                        # ✅ 计算相对于基准点的增量（只有在 move=True 且已设置基准点时才计算）
                        if is_moving and phone_base_pose is not None:
                            # 计算手机的相对变换: delta_phone = phone_base^-1 * phone_current
                            phone_base_inv = np.linalg.inv(phone_base_pose)
                            delta_phone = phone_base_inv @ pose_matrix

                            # 将手机的相对变换应用到机械臂的当前位姿
                            target_position_new = target_position + delta_phone[:3, 3]
                            rotation_matrix = delta_phone[:3, :3]
                            delta_quat = mat2quat(rotation_matrix).astype(np.float64)
                            target_orientation_new = qmult(target_orientation, delta_quat)
                        else:
                            # 如果没有移动或者没有基准点，保持当前位姿
                            target_position_new = target_position
                            target_orientation_new = target_orientation

                        # 计算增量动作（用于记录）
                        delta = np.zeros(6, dtype=np.float64)
                        delta[:3] = target_position_new - target_position

                        # 计算旋转增量
                        delta_quat_for_record = qmult(target_orientation_new, qinverse(target_orientation))
                        delta[3:] = quat2euler(delta_quat_for_record, axes="sxyz")

                        # 处理夹爪命令
                        gripper_drift_delta = np.zeros(6, dtype=np.float64)
                        if gripper_cmd != last_gripper_cmd:
                            # 夹爪状态改变
                            pos_before = self.panda.get_position().astype(np.float64)
                            ori_before = self.panda.get_orientation().astype(np.float64)

                            self.panda.stop_controller()
                            if gripper_cmd > 0.5:  # 打开
                                self.gripper_open()
                                gripper_state = 1.0
                            else:  # 关闭
                                self.gripper_close()
                                gripper_state = 0.0

                            # 记录夹爪动作后的位置
                            target_position = self.panda.get_position().astype(np.float64)
                            target_orientation = self.panda.get_orientation().astype(np.float64)

                            # 计算位置偏移（用于记录）
                            gripper_drift_delta[:3] = target_position - pos_before
                            delta_quat_gripper = qmult(target_orientation, qinverse(ori_before))
                            gripper_drift_delta[3:] = quat2euler(delta_quat_gripper, axes="sxyz")

                            self.panda.start_controller(ctrl)
                            last_gripper_cmd = gripper_cmd

                        # 如果夹爪动作导致了位置偏移，将偏移量加到 delta 中
                        if np.any(gripper_drift_delta != 0):
                            delta = delta + gripper_drift_delta

                        # ✅ 先检查相机图像（如果需要记录）
                        skip_frame = False
                        if (
                            h5_file is not None
                            and camera_buf is not None
                            and image_ds is not None
                            and action_ds is not None
                            and instr_ds is not None
                        ):
                            rgb224 = camera_buf.get_latest()
                            if rgb224 is None:
                                print("[Warning] Camera frame is None, skipping this frame", end="\r")
                                skip_frame = True
                            elif rgb224.shape != (224, 224, 3) or rgb224.dtype != np.uint8:
                                print(f"[Warning] Invalid frame shape/dtype: {rgb224.shape}/{rgb224.dtype}", end="\r")
                                skip_frame = True

                        if skip_frame:
                            continue

                        # 应用增量动作 - 只有当 is_moving 为 True 时才移动
                        if is_moving:
                            # 更新目标位姿
                            target_position = target_position_new
                            target_orientation = target_orientation_new

                            # 设置控制目标
                            ctrl.set_control(target_position, target_orientation)

                        # 记录数据（此时已确保有图像）
                        # ✅ 掐头去尾：检测到移动开始记录，停止移动后结束
                        if (
                            h5_file is not None
                            and camera_buf is not None
                            and image_ds is not None
                            and action_ds is not None
                            and instr_ds is not None
                            and not recording_stopped
                        ):
                            # 检查是否有动作
                            has_action = np.any(delta != 0) or is_moving

                            # 第一次有动作时开始记录（掐头）
                            if has_action and not recording_started:
                                recording_started = True
                                print("[Recording] 检测到第一个动作，开始记录数据...")

                            # 开始记录后，保留所有帧
                            if recording_started:
                                action7 = np.concatenate(
                                    [delta.astype(np.float32, copy=False), [np.float32(gripper_state)]]
                                )
                                img_buf.append(rgb224)
                                act_buf.append(action7.astype(np.float32, copy=False))
                                instr_buf.append(str(language_instruction))

                            if len(img_buf) >= flush_every:
                                n0 = int(image_ds.shape[0])
                                batch_n = len(img_buf)
                                image_ds.resize((n0 + batch_n, 224, 224, 3))
                                action_ds.resize((n0 + batch_n, 7))
                                instr_ds.resize((n0 + batch_n,))
                                image_ds[n0 : n0 + batch_n] = np.stack(img_buf, axis=0)
                                action_ds[n0 : n0 + batch_n] = np.stack(act_buf, axis=0)
                                instr_ds[n0 : n0 + batch_n] = np.asarray(instr_buf, dtype=object)
                                img_buf.clear()
                                act_buf.clear()
                                instr_buf.clear()
                                print(f"[Recording] 已记录 {n0 + batch_n} 帧", end="\r")

            except RuntimeError as e:
                # 检查是否是 reflex 错误
                error_msg = str(e)
                if "cartesian_reflex" in error_msg or "motion aborted by reflex" in error_msg:
                    print(f"\n[Error] Cartesian reflex triggered: {error_msg}")
                    print("[Error] Data will NOT be saved due to reflex error")
                    reflex_error_occurred = True
                raise
            except KeyboardInterrupt:
                print("\n[Recording] 检测到 Ctrl+C，停止记录...")
                manual_stop[0] = True
                running[0] = False
            finally:
                self.panda.stop_controller()
                if camera_stop is not None:
                    camera_stop.set()
                if camera_thread is not None:
                    camera_thread.join(timeout=2.0)

                # 如果发生 reflex 错误，删除文件
                if reflex_error_occurred:
                    if h5_file is not None:
                        try:
                            h5_file.close()
                        except Exception:
                            pass
                    if log_file is not None:
                        try:
                            os.remove(log_file)
                            print(f"[Cleanup] Deleted corrupted data file: {log_file}")
                        except Exception as e:
                            print(f"[Cleanup] Failed to delete file: {e}")
                # 没有 reflex 错误时正常保存数据
                elif h5_file is not None and image_ds is not None and action_ds is not None and instr_ds is not None:
                    if len(img_buf) > 0:
                        n0 = int(image_ds.shape[0])
                        batch_n = len(img_buf)
                        image_ds.resize((n0 + batch_n, 224, 224, 3))
                        action_ds.resize((n0 + batch_n, 7))
                        instr_ds.resize((n0 + batch_n,))
                        image_ds[n0 : n0 + batch_n] = np.stack(img_buf, axis=0)
                        action_ds[n0 : n0 + batch_n] = np.stack(act_buf, axis=0)
                        instr_ds[n0 : n0 + batch_n] = np.asarray(instr_buf, dtype=object)
                    total_frames = int(image_ds.shape[0])
                    try:
                        h5_file.close()
                    except Exception:
                        pass
                    if log_file is not None:
                        print(f"\n[Recording] 记录完成！总计 {total_frames} 帧")
                        print(f"数据已保存到: {log_file}")
        else:
            # 离散控制模式（不推荐用于 teleop）
            raise ValueError("Teleop mode requires use_impedance_control=True")

        print("\n远程遥操作控制已退出")

    def replay_trajectory(
        self,
        trajectory_data: dict = None,
        file_path: str = None,
        speed: float = 1.0,
        loop: int = 1,
        start_index: int = 0,
        end_index: int = None,
        *,
        camera: Optional["RealSenseConnector"] = None,
        show_images: bool = True,
        show_live_camera: bool = False,
    ) -> None:
        """
        重放之前记录的轨迹

        Args:
            trajectory_data: 轨迹数据字典（包含 'q' 和 'dq'）
            file_path: HDF5 文件路径（如果不提供 trajectory_data）
            speed: 播放速度倍率 (0.1-2.0)
            loop: 循环次数
            start_index: 起始索引
            end_index: 结束索引
            camera: RealSenseConnector 实例（可选，用于显示实时对比）
            show_images: 是否显示采集时保存的图像（默认True）
            show_live_camera: 是否同时显示实时相机（用于对比，需要提供 camera）

        Examples:
            # 使用最近一次控制的数据
            arm.manual_control(...)
            arm.replay_trajectory()

            # 从文件加载并重放，显示采集的图像
            arm.replay_trajectory(file_path="data/hdf5/manual_control_20260110_120000.h5")

            # 回放 + 显示采集的图像 + 实时相机对比
            camera = RealSenseConnector()
            arm.replay_trajectory(file_path="...", camera=camera, show_live_camera=True)

            # 慢速回放以便观察
            arm.replay_trajectory(speed=0.5, show_images=True)
        """
        from panda_py import controllers

        # 获取轨迹数据
        if trajectory_data is None:
            if file_path is None and self._last_log_filename is not None:
                file_path = self._last_log_filename
            if file_path is not None:
                # 从文件加载
                print(f"正在从文件加载轨迹: {file_path}")
                with h5py.File(file_path, "r") as f:
                    if "action" in f:
                        actions = np.array(f["action"], dtype=np.float32)
                        hz = float(f.attrs.get("control_frequency_hz", 100.0))
                        trajectory_data = {"action": actions, "control_frequency_hz": hz}
                        # 读取初始位姿（如果有）
                        if "initial_position" in f.attrs:
                            trajectory_data["initial_position"] = list(f.attrs["initial_position"])
                        if "initial_orientation" in f.attrs:
                            trajectory_data["initial_orientation"] = list(f.attrs["initial_orientation"])
                        # 读取图像数据（如果有）
                        if "observation" in f and "image_primary" in f["observation"]:
                            images = np.array(f["observation/image_primary"], dtype=np.uint8)
                            trajectory_data["images"] = images
                            print(f"  ✓ 加载了 {len(images)} 帧图像数据 (shape: {images.shape[1:]})")
                        else:
                            print(f"  ⚠️ 该文件没有图像数据")
                    else:
                        q = np.array(f["q"])
                        dq = np.array(f["dq"]) if "dq" in f else np.zeros_like(q)
                        trajectory_data = {"q": q, "dq": dq}
            else:
                raise ValueError("需要提供 trajectory_data 或 file_path，" "或先运行 manual_control() 并启用 logging")

        if "action" in trajectory_data:
            actions = np.asarray(trajectory_data["action"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(f"Expected action shape (T, 7), got {actions.shape}")

            # 获取图像数据（如果有）
            images = trajectory_data.get("images", None)
            has_images = images is not None and len(images) > 0

            hz = float(trajectory_data.get("control_frequency_hz", 100.0))
            if end_index is None:
                end_index = len(actions)

            actions = actions[start_index:end_index]
            if has_images:
                images = images[start_index:end_index]
                if len(images) != len(actions):
                    print(f"  ⚠️ 警告: 图像数量 ({len(images)}) 与动作数量 ({len(actions)}) 不匹配")
                    has_images = False

            # ✅ 验证初始位姿（如果记录中有的话）
            if "initial_position" in trajectory_data and "initial_orientation" in trajectory_data:
                recorded_pos = np.array(trajectory_data["initial_position"], dtype=np.float64)
                recorded_ori = np.array(trajectory_data["initial_orientation"], dtype=np.float64)
                current_pos = self.panda.get_position().astype(np.float64)
                current_ori = self.panda.get_orientation().astype(np.float64)

                pos_diff = np.linalg.norm(recorded_pos - current_pos)
                ori_diff = np.linalg.norm(recorded_ori - current_ori)

                print(f"\n初始位姿验证:")
                print(f"  位置差异: {pos_diff:.4f} m")
                print(f"  姿态差异: {ori_diff:.4f} (四元数)")

                if pos_diff > 0.05:  # 5cm
                    print(f"  ⚠️ 警告: 初始位置差异较大 ({pos_diff*1000:.1f} mm)")
                    print(f"  记录位置: {recorded_pos}")
                    print(f"  当前位置: {current_pos}")
                    response = input("  继续回放可能导致轨迹偏差，是否继续? (y/n): ").strip().lower()
                    if response != "y":
                        print("  取消回放")
                        return
                elif pos_diff > 0.01:  # 1cm
                    print(f"  ⚠️ 提示: 初始位置有微小差异 ({pos_diff*1000:.1f} mm)，回放可能略有偏差")

            print(f"\n{'='*60}")
            print(f"动作序列重放 (EEF_POS delta + gripper)")
            print(f"{'='*60}")
            print(f"轨迹长度: {len(actions)} 个样本")
            print(f"播放速度: {speed}x")
            print(f"循环次数: {loop}")
            print(f"采样频率: {hz:.1f} Hz")
            print(f"{'='*60}\n")

            # input("按 Enter 键开始重放动作序列...")

            # 初始化图像显示（如果需要）
            cv2_available = False
            if show_images or (show_live_camera and camera is not None):
                try:
                    import cv2

                    cv2_available = True
                    print(f"[Display] OpenCV version: {cv2.__version__}")

                    # 如果需要实时相机，先预热
                    if show_live_camera and camera is not None:
                        print(f"[Display] 预热相机...")
                        for _ in range(5):
                            camera.update(timeout=200)
                            time.sleep(0.1)
                        print(f"[Display] 相机预热完成")

                    if show_images and has_images:
                        print(f"[Display] 将显示采集的图像: {len(images)} 帧")
                        if show_live_camera and camera is not None:
                            # 并排显示：左边是采集的图像，右边是实时相机
                            cv2.namedWindow("Replay - Recorded vs Live", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("Replay - Recorded vs Live", 1280, 480)
                            print("[Display] 图像对比模式：左=采集图像，右=实时相机")
                        else:
                            # 只显示采集的图像
                            cv2.namedWindow("Replay - Recorded Images", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("Replay - Recorded Images", 640, 480)
                            print("[Display] 显示采集时保存的图像")
                    elif show_live_camera and camera is not None:
                        # 只显示实时相机
                        cv2.namedWindow("Replay - Live Camera", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Replay - Live Camera", 640, 480)
                        print("[Display] 显示实时相机画面")
                    else:
                        print(
                            f"[Display] 警告: show_images={show_images}, has_images={has_images}, show_live_camera={show_live_camera}, camera={camera is not None}"
                        )

                    print("提示：按 'q' 键关闭窗口（不会停止回放）")
                except ImportError as e:
                    print(f"\n[Error] OpenCV 导入失败: {e}")
                    print("安装: pip install opencv-python")
                    show_images = False
                    show_live_camera = False
                except Exception as e:
                    print(f"\n[Error] 初始化显示失败: {e}")
                    import traceback

                    traceback.print_exc()
                    show_images = False
                    show_live_camera = False
            else:
                print(f"[Display] 不显示图像: show_images={show_images}, show_live_camera={show_live_camera}")

            from transforms3d.euler import euler2mat
            from transforms3d.quaternions import mat2quat, qmult

            for loop_idx in range(loop):
                if loop > 1:
                    print(f"\n--- 第 {loop_idx + 1}/{loop} 次播放 ---")

                ctrl = controllers.CartesianImpedance(filter_coeff=1.0)
                target_position = self.panda.get_position().astype(np.float64)
                target_orientation = self.panda.get_orientation().astype(np.float64)
                last_gripper_open: Optional[bool] = None

                # 跟踪实时相机失败次数
                live_camera_fail_count = 0
                live_camera_disabled = False

                self.panda.start_controller(ctrl)
                try:
                    freq = max(1.0, hz * float(speed))
                    max_runtime = len(actions) / hz / float(speed) if hz > 0 else None
                    ctx_kwargs = {"frequency": freq}
                    if max_runtime is not None:
                        ctx_kwargs["max_runtime"] = max_runtime

                    print(f"[Replay] 控制参数: 频率={freq:.1f}Hz, 最大时长={max_runtime:.1f}秒")

                    i = 0
                    last_error_check = 0
                    with self.panda.create_context(**ctx_kwargs) as ctx:
                        while ctx.ok() and i < len(actions):
                            # 定期检查机器人状态
                            if i - last_error_check >= 100:
                                try:
                                    robot_state = self.panda.get_robot().read_once()
                                    # 检查是否有错误
                                    if hasattr(robot_state, "robot_mode"):
                                        pass  # 可以检查机器人模式
                                    last_error_check = i
                                except Exception as e:
                                    print(f"\n[Warning] 无法读取机器人状态: {e}")

                            a = actions[i].astype(np.float64, copy=False)
                            delta = a[:6]
                            g = float(a[6])

                            g_open = g >= 0.5
                            if last_gripper_open is None or g_open != last_gripper_open:
                                # Mirror teleop behavior: stop controller -> gripper -> resync pose -> restart controller
                                self.panda.stop_controller()
                                if g_open:
                                    self.gripper_open()
                                else:
                                    self.gripper_close()
                                target_position = self.panda.get_position().astype(np.float64)
                                target_orientation = self.panda.get_orientation().astype(np.float64)
                                self.panda.start_controller(ctrl)
                                last_gripper_open = g_open

                            if np.any(delta != 0):
                                target_position = target_position + delta[:3]
                                if np.any(delta[3:] != 0):
                                    delta_quat = mat2quat(euler2mat(delta[3], delta[4], delta[5]))
                                    target_orientation = qmult(target_orientation, delta_quat)
                                ctrl.set_control(target_position, target_orientation)

                            # 显示图像
                            if cv2_available:
                                try:
                                    display_img = None

                                    # 准备采集的图像
                                    if show_images and has_images and i < len(images):
                                        recorded_img = images[i].copy()
                                        # RGB -> BGR
                                        recorded_img_bgr = cv2.cvtColor(recorded_img, cv2.COLOR_RGB2BGR)
                                        if i == 0:
                                            print(
                                                f"[Display] 第一帧采集图像: shape={recorded_img_bgr.shape}, dtype={recorded_img_bgr.dtype}"
                                            )
                                    else:
                                        recorded_img_bgr = None
                                        if i == 0:
                                            print(
                                                f"[Display] 无法获取采集图像: show_images={show_images}, has_images={has_images}, i={i}, len(images)={len(images) if has_images else 0}"
                                            )

                                    # 准备实时相机图像
                                    if show_live_camera and camera is not None and not live_camera_disabled:
                                        ok = camera.update(timeout=50)
                                        if ok and camera.img is not None:
                                            live_img = camera.img.copy()
                                            live_img_bgr = cv2.cvtColor(live_img, cv2.COLOR_RGB2BGR)
                                            if i == 0:
                                                print(
                                                    f"[Display] 第一帧实时图像: shape={live_img_bgr.shape}, dtype={live_img_bgr.dtype}"
                                                )
                                            live_camera_fail_count = 0  # 重置失败计数
                                        else:
                                            live_img_bgr = None
                                            live_camera_fail_count += 1
                                            if i == 0:
                                                print(f"[Display] 无法获取实时图像: ok={ok}")

                                            # 如果连续失败10次，自动禁用实时相机
                                            if live_camera_fail_count >= 10 and not live_camera_disabled:
                                                print(
                                                    f"\n[Display] 实时相机连续失败 {live_camera_fail_count} 次，自动切换为只显示采集图像"
                                                )
                                                live_camera_disabled = True
                                                # 关闭旧窗口，创建新窗口
                                                cv2.destroyAllWindows()
                                                cv2.namedWindow("Replay - Recorded Images", cv2.WINDOW_NORMAL)
                                                cv2.resizeWindow("Replay - Recorded Images", 640, 480)
                                    else:
                                        live_img_bgr = None

                                    # 构建显示图像
                                    if recorded_img_bgr is not None and live_img_bgr is not None:
                                        # 对比模式：并排显示
                                        # 调整大小到相同高度
                                        h = min(recorded_img_bgr.shape[0], live_img_bgr.shape[0])
                                        recorded_resized = cv2.resize(
                                            recorded_img_bgr,
                                            (int(recorded_img_bgr.shape[1] * h / recorded_img_bgr.shape[0]), h),
                                        )
                                        live_resized = cv2.resize(
                                            live_img_bgr, (int(live_img_bgr.shape[1] * h / live_img_bgr.shape[0]), h)
                                        )

                                        # 添加标签
                                        cv2.putText(
                                            recorded_resized,
                                            "RECORDED",
                                            (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 255, 0),
                                            2,
                                        )
                                        cv2.putText(
                                            live_resized, "LIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
                                        )

                                        display_img = np.hstack([recorded_resized, live_resized])
                                        window_name = "Replay - Recorded vs Live"
                                        if i == 0:
                                            print(f"[Display] 对比模式: 合并图像 shape={display_img.shape}")
                                    elif recorded_img_bgr is not None:
                                        # 只显示采集的图像
                                        display_img = recorded_img_bgr
                                        window_name = "Replay - Recorded Images"
                                        if i == 0:
                                            print(f"[Display] 单图模式: 采集图像 shape={display_img.shape}")
                                    elif live_img_bgr is not None:
                                        # 只显示实时图像
                                        display_img = live_img_bgr
                                        window_name = "Replay - Live Camera"
                                        if i == 0:
                                            print(f"[Display] 单图模式: 实时图像 shape={display_img.shape}")

                                    # 在图像上叠加信息
                                    if display_img is not None:
                                        progress = (i / len(actions)) * 100
                                        info_text = [
                                            f"Frame: {i+1}/{len(actions)}",
                                            f"Progress: {progress:.1f}%",
                                            f"Loop: {loop_idx+1}/{loop}",
                                            f"Gripper: {'Open' if g_open else 'Closed'}",
                                        ]

                                        y_offset = display_img.shape[0] - 140
                                        for text in info_text:
                                            cv2.putText(
                                                display_img,
                                                text,
                                                (10, y_offset),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7,
                                                (255, 255, 0),
                                                2,
                                            )
                                            y_offset += 35

                                        # 显示图像
                                        cv2.imshow(window_name, display_img)
                                        if i == 0:
                                            print(f"[Display] 第一帧已显示在窗口: {window_name}")

                                        # 检查按键（非阻塞）
                                        key = cv2.waitKey(1) & 0xFF
                                        if key == ord("q"):
                                            print("\n[Display] 用户按 'q' 键，关闭图像显示（回放继续）")
                                            cv2.destroyAllWindows()
                                            cv2_available = False  # 标记为不可用，后续不再显示
                                    else:
                                        if i == 0:
                                            print(f"[Display] 错误: 无法生成显示图像")
                                except Exception as e:
                                    if i == 0 or i % 100 == 0:
                                        print(f"[Warning] Image display failed at frame {i}: {e}")
                                        import traceback

                                        traceback.print_exc()

                            i += 1
                            if i % 100 == 0:
                                progress = (i / len(actions)) * 100
                                print(f"进度: {progress:.1f}% ({i}/{len(actions)})", end="\r")

                    # 检查是否完整播放
                    if i < len(actions):
                        print(f"\n⚠️ 警告: 播放未完成! 播放了 {i}/{len(actions)} 帧 ({i/len(actions)*100:.1f}%)")
                        print(f"  ctx.ok() = {ctx.ok()}")

                        # 尝试读取机器人错误状态
                        try:
                            robot_state = self.panda.get_robot().read_once()
                            print(f"  机器人状态检查:")

                            # 检查是否有碰撞
                            if hasattr(robot_state, "cartesian_collision"):
                                collision = any(robot_state.cartesian_collision)
                                print(f"    碰撞检测: {collision}")

                            # 检查关节位置
                            if hasattr(robot_state, "q"):
                                q = robot_state.q
                                print(f"    当前关节位置: [{', '.join(f'{x:.3f}' for x in q[:3])}...]")

                            # 检查末端位置
                            pose = self.panda.get_pose()
                            if pose is not None:
                                T = SE3(pose, check=False)
                                pos = T.t
                                print(f"    当前末端位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

                        except Exception as e:
                            print(f"  无法读取详细状态: {e}")

                        print(f"\n可能原因:")
                        print(f"  1. 碰撞检测触发（机器人检测到碰撞）")
                        print(f"  2. 工作空间边界（超出机器人可达范围）")
                        print(f"  3. 关节限位（某个关节超出限制）")
                        print(f"  4. 控制器超时（动作执行时间过长）")
                        print(f"  5. 初始位置不匹配（回放起点与采集起点不同）")
                    else:
                        print(f"\n✓ 播放完成 ({i}/{len(actions)} 样本)")
                except KeyboardInterrupt:
                    print("\n中断播放")
                    break
                finally:
                    self.panda.stop_controller()

                if loop_idx < loop - 1:
                    try:
                        response = input("\n继续下一次播放？(Y/n): ").strip().lower()
                        if response == "n":
                            print("取消剩余播放")
                            break
                    except KeyboardInterrupt:
                        print("\n取消剩余播放")
                        break

            # 清理图像显示窗口
            if cv2_available:
                try:
                    import cv2

                    cv2.destroyAllWindows()
                except Exception:
                    pass

            print("\n✓ 重放完成")
            return

        q = trajectory_data["q"]
        dq = trajectory_data["dq"]

        if end_index is None:
            end_index = len(q)

        # 裁剪轨迹
        q = q[start_index:end_index]
        dq = dq[start_index:end_index]

        print(f"\n{'='*60}")
        print(f"轨迹重放")
        print(f"{'='*60}")
        print(f"轨迹长度: {len(q)} 个样本")
        print(f"播放速度: {speed}x")
        print(f"循环次数: {loop}")
        print(f"{'='*60}\n")

        # 移动到起始位置
        print("正在移动到轨迹起始位置...")
        self.panda.move_to_joint_position(q[0])
        print("已到达起始位置\n")

        input("按 Enter 键开始重放轨迹...")

        for loop_idx in range(loop):
            if loop > 1:
                print(f"\n--- 第 {loop_idx + 1}/{loop} 次播放 ---")

            # 计算控制频率
            base_frequency = 1000.0
            control_frequency = base_frequency * speed
            max_runtime = len(q) / base_frequency / speed

            # 启动控制器
            ctrl = controllers.JointPosition()
            self.panda.start_controller(ctrl)

            try:
                i = 0
                with self.panda.create_context(frequency=control_frequency, max_runtime=max_runtime) as ctx:
                    while ctx.ok() and i < len(q):
                        ctrl.set_control(q[i], dq[i])
                        i += 1

                        if i % 100 == 0:
                            progress = (i / len(q)) * 100
                            print(f"进度: {progress:.1f}% ({i}/{len(q)})", end="\r")

                print(f"\n✓ 播放完成 ({i}/{len(q)} 样本)")

            except KeyboardInterrupt:
                print("\n中断播放")
                break
            except Exception as e:
                print(f"\n播放出错: {e}")
                raise
            finally:
                self.panda.stop_controller()

            # 询问是否继续下一轮
            if loop_idx < loop - 1:
                try:
                    response = input("\n继续下一次播放？(Y/n): ").strip().lower()
                    if response == "n":
                        print("取消剩余播放")
                        break
                except KeyboardInterrupt:
                    print("\n取消剩余播放")
                    break

        print("\n✓ 重放完成")


if __name__ == "__main__":
    robotic_arm_instance = RoboticArmControler()
    try:
        print("")
        robotic_arm_instance.move_to_start()
        robotic_arm_instance.gripper_grasp(width=0.5)
    finally:
        robotic_arm_instance.cleanup()

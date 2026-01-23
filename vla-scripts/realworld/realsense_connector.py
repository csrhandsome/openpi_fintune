from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RealSenseSensorData:
    timestamp: float = 0.0
    color_img: Optional[np.ndarray] = None  # RGB uint8 (H,W,3)
    depth_m: Optional[np.ndarray] = None  # float32 (H,W) in meters
    intrinsics: Optional[np.ndarray] = None  # 3x3 float32
    ee_pose: Optional[np.ndarray] = None  # 4x4 float32 (optional, for stale observation)


class RealSenseConnector:
    """
    RealSense RGBD connector with YOLO hand detection + skin segmentation.

    按照训练数据格式：
    - mask = 0: 物体（target）
    - mask = 1: 手（other）
    - mask = 255: 背景

    流程：
    1. YOLO 手部检测 → Bounding Box (手+物体区域)
    2. 深度阈值过滤 → 切掉远处背景
    3. 肤色分割 → 区分手和物体
    4. 非肤色区域 = 物体 (mask=0)

    The main API surface used by `handover_sim2real.realworld_runner.RealWorldRunner`:
    - `update()`
    - `img` / `depth` / `intrinsics` / `mask`
    """

    def __init__(
        self,
        *,
        serial: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_depth_to_color: bool = True,
        # YOLO 手部检测参数
        yolo_model_name: str = "hand_yolov8n.pt",
        yolo_conf: float = 0.5,
        # 深度过滤参数
        min_depth_m: float = 0.1,
        max_depth_m: float = 1.5,
        # 动态深度裁剪：基于手部深度的容差范围
        depth_tolerance_m: float = 0.15,  # 手部深度 ± 15cm
        # 肤色分割参数 (HSV 范围)
        skin_h_min: int = 0,
        skin_h_max: int = 25,
        skin_s_min: int = 30,
        skin_s_max: int = 170,
        skin_v_min: int = 50,
        skin_v_max: int = 255,
        # Bounding box 扩展比例
        bbox_expand_ratio: float = 0.3,
    ) -> None:
        self._serial = serial
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._align_depth_to_color = align_depth_to_color

        self._pipeline = None
        self._profile = None
        self._align = None
        self._depth_scale = None

        # YOLO 参数
        self._yolo_model_name = yolo_model_name
        self._yolo_conf = yolo_conf
        self._yolo_model = None

        # 深度过滤参数
        self._min_depth_m = min_depth_m
        self._max_depth_m = max_depth_m
        self._depth_tolerance_m = depth_tolerance_m

        # 肤色分割参数 (HSV)
        self._skin_h_min = skin_h_min
        self._skin_h_max = skin_h_max
        self._skin_s_min = skin_s_min
        self._skin_s_max = skin_s_max
        self._skin_v_min = skin_v_min
        self._skin_v_max = skin_v_max

        # Bounding box 扩展
        self._bbox_expand_ratio = bbox_expand_ratio

        self._sensor_data = RealSenseSensorData()
        self._cached_mask: Optional[np.ndarray] = None
        self._mask_dirty = True

        # 缓存检测结果用于调试
        self._last_hand_bbox = None  # (x1, y1, x2, y2)
        # 上一帧有效的完整观测包（用于检测失败时的回退）
        # 包含: rgb, depth_m, mask, intrinsics, timestamp
        self._last_valid_observation: Optional[dict] = None
        # 标记当前帧是否使用了缓存的观测（stale）
        self._is_stale_observation: bool = False
        # 连续使用stale observation的次数
        self._stale_count: int = 0
        self._max_stale_count: int = 3  # 最多连续使用3次陈旧观测

    def connect(self) -> "RealSenseConnector":
        if self._pipeline is not None:
            return self

        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise RuntimeError(
                "RealSense connector requires `pyrealsense2`. "
                "Install Intel RealSense SDK Python bindings."
            ) from e

        pipeline = rs.pipeline()
        cfg = rs.config()
        if self._serial:
            cfg.enable_device(self._serial)

        cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        cfg.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)

        profile = pipeline.start(cfg)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())

        if self._align_depth_to_color:
            align = rs.align(rs.stream.color)
        else:
            align = None

        self._pipeline = pipeline
        self._profile = profile
        self._align = align
        self._depth_scale = depth_scale

        self._refresh_intrinsics()
        return self

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._profile = None
                self._align = None
                self._depth_scale = None
        self._yolo_model = None

    def __enter__(self) -> "RealSenseConnector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def _refresh_intrinsics(self) -> None:
        if self._profile is None:
            return
        try:
            import pyrealsense2 as rs
        except Exception:
            return

        # If we align depth to color, downstream depth pixels are in the color image coordinates,
        # so we should use color intrinsics. Otherwise, use depth intrinsics.
        preferred_stream = rs.stream.color if self._align_depth_to_color else rs.stream.depth
        fallback_stream = rs.stream.depth if preferred_stream == rs.stream.color else rs.stream.color

        try:
            vs_profile = self._profile.get_stream(preferred_stream).as_video_stream_profile()
            print("Using preferred stream intrinsics")
        except Exception:
            try:
                vs_profile = self._profile.get_stream(fallback_stream).as_video_stream_profile()
                print("Using fallback stream intrinsics")
            except Exception:
                return

        intr = vs_profile.get_intrinsics()
        K = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._sensor_data.intrinsics = K
        self._mask_dirty = True

    def set_intrinsics(self, K: np.ndarray) -> None:
        self._sensor_data.intrinsics = np.array(K, dtype=np.float32).reshape(3, 3)
        self._mask_dirty = True

    def update(self, timeout: int = 1000) -> bool:
        if self._pipeline is None:
            self.connect()

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=int(timeout))
        except Exception:
            return False

        if self._align is not None:
            try:
                frames = self._align.process(frames)
            except Exception:
                pass

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False

        color_img = np.asanyarray(color_frame.get_data())
        if color_img.ndim != 3 or color_img.shape[-1] != 3:
            return False
        color_img = color_img.astype(np.uint8, copy=False)  # RGB8

        depth_raw = np.asanyarray(depth_frame.get_data())
        if depth_raw.ndim != 2:
            return False
        depth_scale = 0.001 if self._depth_scale is None else float(self._depth_scale)
        depth_m = depth_raw.astype(np.float32, copy=False) * depth_scale

        ts = float(getattr(depth_frame, "get_timestamp", lambda: 0.0)())
        intrinsics = self._sensor_data.intrinsics
        self._sensor_data = RealSenseSensorData(
            timestamp=ts,
            color_img=color_img,
            depth_m=depth_m,
            intrinsics=intrinsics,
        )
        self._mask_dirty = True
        return True

    @property
    def sensor_data(self) -> RealSenseSensorData:
        return self._sensor_data

    @property
    def img(self) -> Optional[np.ndarray]:
        return self._sensor_data.color_img

    @property
    def depth(self) -> Optional[np.ndarray]:
        return self._sensor_data.depth_m

    @property
    def intrinsics(self) -> Optional[np.ndarray]:
        return self._sensor_data.intrinsics

    def _init_yolo_model(self):
        """初始化 YOLO 手部检测模型"""
        if self._yolo_model is None:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download

            # 从 HuggingFace 下载模型（会自动缓存）
            # model_path = hf_hub_download("Bingsu/adetailer", self._yolo_model_name)
            # print("[RealSenseConnector] Loading YOLO model from:", model_path)
            self._yolo_model = YOLO("/home/three/.cache/huggingface/hub/models--Bingsu--adetailer/snapshots/53cc19de382014514d9d4038601d261a7faa9b7b/hand_yolov8n.pt")
        return self._yolo_model

    def _compute_mask(self) -> Optional[np.ndarray]:
        """
        计算物体 mask：
        1. YOLO 手部检测 → Bounding Box
        2. 深度阈值过滤 → 切掉远处背景
        3. 肤色分割 → 区分手和物体
        4. 非肤色区域 = 物体 (mask=0)

        Returns:
            mask: (H, W) uint8
                - 0 = 物体（target）
                - 1 = 手（other）
                - 255 = 背景
        """
        import cv2

        color_img = self._sensor_data.color_img
        depth_m = self._sensor_data.depth_m

        if color_img is None or depth_m is None:
            return None

        h, w = color_img.shape[:2]
        # 默认全部是背景
        mask = np.full((h, w), 255, dtype=np.uint8)

        # Step 1: YOLO 手部检测
        model = self._init_yolo_model()
        # YOLO 需要 BGR 输入
        img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        results = model.predict(img_bgr, conf=self._yolo_conf, verbose=False)

        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            # 没有检测到手，回退到上一帧完整观测包
            self._last_hand_bbox = None
            if self._last_valid_observation is not None and self._stale_count < self._max_stale_count:
                print(f"[RealSenseConnector] No hand detected, using last valid FULL observation (rgb+depth+mask) [{self._stale_count + 1}/{self._max_stale_count}]")
                # 恢复完整观测包（rgb, depth, mask 都回退到同一帧）
                self._sensor_data = RealSenseSensorData(
                    timestamp=self._last_valid_observation["timestamp"],
                    color_img=self._last_valid_observation["rgb"].copy(),
                    depth_m=self._last_valid_observation["depth_m"].copy(),
                    intrinsics=self._last_valid_observation["intrinsics"],
                )
                self._is_stale_observation = True
                self._stale_count += 1
                return self._last_valid_observation["mask"].copy()
            # 没有缓存的有效观测，或者已超过最大回退次数
            if self._stale_count >= self._max_stale_count:
                print(f"[RealSenseConnector] Exceeded max stale count ({self._max_stale_count}), returning background mask")
            self._is_stale_observation = False
            self._stale_count = 0
            return mask

        # 取置信度最高的检测框
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

        # 扩展 bounding box（手可能握着的物体在框外一点）
        box_w = x2 - x1
        box_h = y2 - y1
        expand_w = int(box_w * self._bbox_expand_ratio)
        expand_h = int(box_h * self._bbox_expand_ratio)

        x1_exp = max(0, x1 - expand_w)
        y1_exp = max(0, y1 - expand_h)
        x2_exp = min(w, x2 + expand_w)
        y2_exp = min(h, y2 + expand_h)

        self._last_hand_bbox = (x1_exp, y1_exp, x2_exp, y2_exp)

        # Step 2: 肤色分割（在 ROI 内）- 先做肤色分割来获取手部区域
        roi_rgb = color_img[y1_exp:y2_exp, x1_exp:x2_exp]
        roi_depth = depth_m[y1_exp:y2_exp, x1_exp:x2_exp]
        roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

        # HSV 肤色范围
        lower_skin = np.array([self._skin_h_min, self._skin_s_min, self._skin_v_min], dtype=np.uint8)
        upper_skin = np.array([self._skin_h_max, self._skin_s_max, self._skin_v_max], dtype=np.uint8)
        skin_mask_roi = cv2.inRange(roi_hsv, lower_skin, upper_skin)

        # 形态学处理：去噪 + 填充
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask_roi = cv2.morphologyEx(skin_mask_roi, cv2.MORPH_OPEN, kernel)
        skin_mask_roi = cv2.morphologyEx(skin_mask_roi, cv2.MORPH_CLOSE, kernel)
        # 膨胀一点，确保手部边缘被覆盖
        skin_mask_roi = cv2.dilate(skin_mask_roi, kernel, iterations=2)

        # Step 3: 动态深度裁剪 - 基于手部深度计算有效范围
        # 提取手部区域的深度值
        hand_depth_values = roi_depth[(skin_mask_roi > 0) & (roi_depth > 0)]

        if len(hand_depth_values) > 0:
            # 使用中位数作为参考深度（抗噪性强）
            z_ref = np.median(hand_depth_values)
            # 动态深度范围：手的深度 ± tolerance
            min_depth_dynamic = max(self._min_depth_m, z_ref - self._depth_tolerance_m)
            max_depth_dynamic = min(self._max_depth_m, z_ref + self._depth_tolerance_m)
        else:
            # 没检测到手的深度，回退到固定阈值
            min_depth_dynamic = self._min_depth_m
            max_depth_dynamic = self._max_depth_m

        # 生成深度有效区域掩码
        valid_depth_mask = (roi_depth > min_depth_dynamic) & (roi_depth < max_depth_dynamic) & (roi_depth > 0)

        # Step 4: 生成最终 mask
        # 在 ROI 内：
        #   - 有效深度 & 非肤色 = 物体 (mask=0)
        #   - 有效深度 & 肤色 = 手 (mask=1)
        #   - 无效深度 = 背景 (mask=255)

        roi_mask = np.full((y2_exp - y1_exp, x2_exp - x1_exp), 255, dtype=np.uint8)

        # 物体：有效深度 & 非肤色
        object_region = valid_depth_mask & (skin_mask_roi == 0)
        roi_mask[object_region] = 0

        # 手：有效深度 & 肤色
        hand_region = valid_depth_mask & (skin_mask_roi > 0)
        roi_mask[hand_region] = 1

        # Step 5: 形态学去噪 - 去除深度边缘的孤立噪点
        # 对物体区域做开运算（先腐蚀后膨胀），去除小的噪点
        object_binary = (roi_mask == 0).astype(np.uint8)
        denoise_kernel = np.ones((3, 3), np.uint8)
        object_binary = cv2.morphologyEx(object_binary, cv2.MORPH_OPEN, denoise_kernel)
        # 恢复去噪后的物体区域，被去掉的噪点变成背景
        roi_mask[(roi_mask == 0) & (object_binary == 0)] = 255

        # 写回到完整 mask
        mask[y1_exp:y2_exp, x1_exp:x2_exp] = roi_mask

        # 保存完整观测包，供下一帧回退使用
        self._last_valid_observation = {
            "rgb": color_img.copy(),
            "depth_m": depth_m.copy(),
            "mask": mask.copy(),
            "intrinsics": self._sensor_data.intrinsics,
            "timestamp": self._sensor_data.timestamp,
        }
        self._is_stale_observation = False
        self._stale_count = 0  # 重置连续stale计数

        return mask

    @property
    def hand_bbox(self):
        """返回最后一帧检测到的手部 bounding box (x1, y1, x2, y2)，用于调试"""
        return self._last_hand_bbox

    @property
    def is_stale_observation(self) -> bool:
        """返回当前观测是否是回退的陈旧观测（YOLO检测失败时使用上一帧）"""
        return self._is_stale_observation

    @property
    def mask(self) -> Optional[np.ndarray]:
        """
        Returns a label mask (H, W) uint8: background=0, instances=1..N.
        """
        if self._mask_dirty or self._cached_mask is None:
            self._cached_mask = self._compute_mask()
            self._mask_dirty = False
        return self._cached_mask


RealsenseConnector = RealSenseConnector


def _main() -> None:
    import argparse
    import time

    p = argparse.ArgumentParser(description="Minimal RealSenseConnector smoke test")
    p.add_argument("--serial", default=None, help="Optional RealSense device serial")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-align", action="store_true", help="Disable depth->color alignment")
    p.add_argument("--frames", type=int, default=30, help="Number of frames to capture")
    p.add_argument("--timeout-ms", type=int, default=1000)
    p.add_argument("--print-mask", action="store_true", help="Compute FastSAM mask and print ids")
    args = p.parse_args()

    connector = RealSenseConnector(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        align_depth_to_color=not args.no_align,
    )

    with connector:
        print("Connected. Intrinsics K:\n", connector.intrinsics)
        for i in range(int(args.frames)):
            ok = connector.update(timeout=int(args.timeout_ms))
            if not ok:
                print(f"[{i}] update() failed")
                continue

            rgb = connector.img
            depth_m = connector.depth
            if rgb is None or depth_m is None:
                print(f"[{i}] missing rgb/depth")
                continue

            msg = f"[{i}] rgb={rgb.shape} {rgb.dtype}, depth={depth_m.shape} {depth_m.dtype}"
            if args.print_mask:
                mask = connector.mask
                if mask is None:
                    msg += ", mask=None"
                else:
                    msg += f", mask_ids={np.unique(mask)[:10]}"
            print(msg)
            time.sleep(0.001)


if __name__ == "__main__":
    _main()

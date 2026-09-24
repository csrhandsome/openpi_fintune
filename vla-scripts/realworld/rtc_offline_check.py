#!/usr/bin/env python
"""离线验证 RTC 训练是否健康 —— 不碰机械臂、无 1kHz 控制环、无线程并发抖动。

动机（见 debug_note.md §3.6）：实机 crash 永远发生在第一次 RTC 切换【之前】，
所以崩溃的命令来自 delay=0 的开环 chunk，与 RTC 的 prefix-conditioning 无因果关系。
要把「RTC 训练到底有没有问题」从所有机器人/并发噪声里隔离出来，唯一干净的办法是
直接对 server 上同一个 checkpoint 发两次请求，在 action 空间里检查：

  1) prefix 保真 (pin fidelity)：
     先 delay=0 生成 chunk A，取 A[skip : skip+delay] 作为 action_prefix 回灌，
     再 delay=D 生成 chunk B。模型应当把 B[0:D] 硬钉成等于该 prefix。
     若 B[0:D] != prefix → 服务端 prefix 注入 / action 空间往返 (delta+normalize) 有 bug。

  2) 缝接平滑 (seam smoothness)：
     缝接点跳变 |B[D] - B[D-1]| 应当与 chunk 内部相邻步 |B[i+1]-B[i]| 同量级，
     没有突跳。若缝接处显著更大 → RTC 采样在 prefix→postfix 边界处不连续。

判定：
  - 两条都通过 → RTC 训练健康。实机 crash 的根因在执行/并发侧（去跑
    debug_note §3.6 的 --position-filter-coeff 0.3 实验），不要再怀疑训练。
  - prefix 不保真 / 缝接突跳 → 训练或采样侧确有 bug，值得继续深挖。

注意：本脚本只合成机械臂低维状态（关节角 + gripper）。图像可以用灰图，也可以
直接从真实 RealSense 双相机抓一帧。它验证的是 RTC 机制本身（pin + seam），不验证
策略在真实场景下的动作质量——后者本来就该在实机上看。

用法（灰图）：
  uv run vla-scripts/realworld/rtc_offline_check.py \
      --remote-host 192.168.5.54 --remote-port 8001 \
      --action-horizon 32 --open-loop-horizon 24 --delay 8 \
      --instruction "Place the object into the bassket"

用法（真实相机图像 + 合成机械臂状态）：
  uv run vla-scripts/realworld/rtc_offline_check.py \
      --remote-host 192.168.5.54 --remote-port 8001 \
      --image-source cameras --color-only \
      --external-camera-serial 825412070292 \
      --wrist-camera-serial 825412070487 \
      --action-horizon 32 --open-loop-horizon 24 --delay 8 \
      --instruction "Place the object into the bassket"
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Literal

import numpy as np
from openpi_client import websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    remote_host: str = "192.168.5.54"
    remote_port: int = 8001

    # 必须与训练 config 的 model.action_horizon 一致（franka joint RTC 工作区为 32）。
    action_horizon: int = 32
    # 实机每次重新请求策略前已经执行掉的旧 chunk 步数；应与推理脚本 --open-loop-horizon 一致。
    open_loop_horizon: int = 24
    # 兼容旧参数名；如果显式传 --skip，则覆盖 --open-loop-horizon。
    skip: int | None = None
    # RTC delay = prefix 长度。必须 <= rtc_train_max_delay(=8)，否则是分布外，结果无意义。
    delay: int = 8
    action_dim: int = 8  # 客户端归一化后实际使用的前 8 维（7 关节 + 1 夹爪）。
    # 总推理 chunk 数。第 1 个开环，后续 chunk 都从前一个 chunk 回灌 RTC prefix。
    num_chunks: int = 10

    instruction: str = "Place the object into the bassket"

    # 图像来源：synthetic=灰图；cameras=真实 RealSense 外部相机 + 腕部相机。
    image_source: Literal["synthetic", "cameras"] = "synthetic"
    external_camera_serial: str | None = None
    wrist_camera_serial: str | None = None
    # 默认要求显式 serial，避免 RealSense auto 选择到错误/已占用设备。
    require_camera_serials: bool = True
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_timeout_ms: int = 500
    camera_warmup_frames: int = 5
    camera_retry_frames: int = 10
    color_only: bool = False
    camera_snapshot_path: str | None = "rtc_offline_camera_views.png"

    # 合成机械臂低维状态（与训练预处理对齐：7 关节 + 1 gripper）。
    joint_position: tuple[float, ...] = (0.00388, -0.45916, -0.15237, -2.47266, -0.06778, 1.96442, 0.70214)
    gripper_position: float = 1.0

    # 缝接突跳判定：seam_jump / median(内部相邻步) 超过该倍数即判 FAIL。
    seam_ratio_threshold: float = 3.0
    # prefix 保真判定：逐位绝对误差超过该值即判 FAIL（归一化后 action 空间，留浮点裕度）。
    pin_atol: float = 1e-3
    # 作为测试脚本使用时，检查失败应返回非零状态码；只想看打印可传 --no-exit-nonzero-on-fail。
    exit_nonzero_on_fail: bool = True


@dataclasses.dataclass(frozen=True)
class ObservationImages:
    exterior: np.ndarray
    wrist: np.ndarray
    source: str


def _center_crop_and_resize(rgb: np.ndarray, out_hw: int = 224, crop_scale: float = 0.9) -> np.ndarray:
    """Center crop + resize to match franka_joint_main_rtc.py preprocessing."""
    from PIL import Image

    rgb = _ensure_rgb_uint8("camera image", rgb)
    h, w = rgb.shape[:2]
    frac = float(np.sqrt(crop_scale))
    crop_h = max(1, min(h, round(h * frac)))
    crop_w = max(1, min(w, round(w * frac)))
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    cropped = rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
    img = Image.fromarray(cropped, mode="RGB").resize((out_hw, out_hw), resample=Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _ensure_rgb_uint8(name: str, image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {arr.shape}.")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 RGB, got dtype {arr.dtype}.")
    return arr


def _synthetic_images() -> ObservationImages:
    gray = np.full((224, 224, 3), 127, dtype=np.uint8)
    return ObservationImages(exterior=gray.copy(), wrist=gray.copy(), source="synthetic-gray")


def _capture_camera_images(args: Args) -> ObservationImages:
    from dual_camera_manager import DualCameraManager
    from PIL import Image

    cameras = DualCameraManager(
        external_serial=args.external_camera_serial,
        wrist_serial=args.wrist_camera_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        color_only=args.color_only,
    )
    try:
        try:
            cameras.connect()
        except RuntimeError as exc:
            raise RuntimeError(
                "failed to connect RealSense cameras. "
                "Pass the same --external-camera-serial and --wrist-camera-serial used by "
                "franka_joint_main_rtc.py; use --no-require-camera-serials only if auto selection is intended."
            ) from exc
        for _ in range(max(0, int(args.camera_warmup_frames))):
            cameras.update(timeout_ms=args.camera_timeout_ms)

        external_img = wrist_img = None
        external_ok = wrist_ok = False
        for attempt in range(1, max(1, int(args.camera_retry_frames)) + 1):
            external_ok, wrist_ok = cameras.update(timeout_ms=args.camera_timeout_ms)
            if external_ok and wrist_ok:
                external_img, wrist_img = cameras.get_images()
            if external_img is not None and wrist_img is not None:
                print(f"[rtc-check] captured RealSense frames on attempt {attempt}")
                break

        if external_img is None or wrist_img is None:
            raise RuntimeError(
                "failed to capture both camera frames: "
                f"external_ok={external_ok}, wrist_ok={wrist_ok}, "
                f"external_img_is_none={external_img is None}, wrist_img_is_none={wrist_img is None}"
            )

        external_raw = _ensure_rgb_uint8("external camera image", external_img).copy()
        wrist_raw = _ensure_rgb_uint8("wrist camera image", wrist_img).copy()
        if args.camera_snapshot_path:
            snapshot = np.concatenate([external_raw, wrist_raw], axis=1)
            Image.fromarray(snapshot).save(Path(args.camera_snapshot_path))
            print(f"[rtc-check] saved camera snapshot: {args.camera_snapshot_path}")

        return ObservationImages(
            exterior=_center_crop_and_resize(external_raw),
            wrist=_center_crop_and_resize(wrist_raw),
            source="realsense-cameras",
        )
    finally:
        cameras.close()


def _load_observation_images(args: Args) -> ObservationImages:
    if args.image_source == "synthetic":
        return _synthetic_images()
    if args.image_source == "cameras":
        return _capture_camera_images(args)
    raise ValueError(f"Unsupported --image-source: {args.image_source!r}")


def _prefix_start(args: Args) -> int:
    return int(args.open_loop_horizon if args.skip is None else args.skip)


def _validate_args(args: Args) -> None:
    if len(args.joint_position) != 7:
        raise ValueError(f"--joint-position must contain 7 joint values, got {len(args.joint_position)}.")
    if args.action_horizon <= 1:
        raise ValueError("--action-horizon must be > 1.")
    if args.action_dim <= 0:
        raise ValueError("--action-dim must be > 0.")
    if args.num_chunks < 2:
        raise ValueError("--num-chunks must be >= 2 to exercise RTC transitions.")
    if args.open_loop_horizon < 0:
        raise ValueError("--open-loop-horizon must be >= 0.")
    if args.skip is not None and args.skip < 0:
        raise ValueError("--skip must be >= 0 when provided.")
    if args.delay <= 0:
        raise ValueError("--delay must be > 0 to exercise RTC.")
    if args.delay >= args.action_horizon:
        raise ValueError("--delay must be < --action-horizon so the seam step B[delay] exists.")
    prefix_start = _prefix_start(args)
    if prefix_start + args.delay > args.action_horizon:
        raise ValueError(
            f"open_loop_horizon/skip + delay ({prefix_start}+{args.delay}) must be <= "
            f"action_horizon ({args.action_horizon})."
        )
    if args.image_source == "cameras" and args.require_camera_serials:
        if args.external_camera_serial is None or args.wrist_camera_serial is None:
            raise ValueError(
                "--image-source cameras requires explicit --external-camera-serial and --wrist-camera-serial. "
                "Use the same serials that work in franka_joint_main_rtc.py, or pass "
                "--no-require-camera-serials to allow RealSense auto selection."
            )
    if args.camera_width <= 0 or args.camera_height <= 0 or args.camera_fps <= 0:
        raise ValueError("camera width/height/fps must be > 0.")
    if args.camera_timeout_ms <= 0:
        raise ValueError("--camera-timeout-ms must be > 0.")
    if args.camera_retry_frames <= 0:
        raise ValueError("--camera-retry-frames must be > 0.")
    if args.seam_ratio_threshold <= 0:
        raise ValueError("--seam-ratio-threshold must be > 0.")
    if args.pin_atol < 0:
        raise ValueError("--pin-atol must be >= 0.")


def _build_request(args: Args, images: ObservationImages) -> dict:
    return {
        "observation/exterior_image_1_left": images.exterior.copy(),
        "observation/wrist_image_left": images.wrist.copy(),
        "observation/joint_position": np.asarray(args.joint_position, dtype=np.float32),
        "observation/gripper_position": np.asarray([args.gripper_position], dtype=np.float32),
        "prompt": args.instruction,
    }


def _print_request_summary(req: dict, images: ObservationImages) -> None:
    print(
        "[rtc-check] observation:"
        f" image_source={images.source},"
        f" exterior={req['observation/exterior_image_1_left'].shape},"
        f" wrist={req['observation/wrist_image_left'].shape},"
        f" joint={np.asarray(req['observation/joint_position']).shape},"
        f" gripper={np.asarray(req['observation/gripper_position']).shape}"
    )


def _normalize_chunk(actions: np.ndarray, action_dim: int) -> np.ndarray:
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2:
        raise ValueError(f"server returned actions with rank {chunk.ndim}, expected rank 2; shape={chunk.shape}")
    if chunk.shape[1] < action_dim:
        raise ValueError(f"server returned action dim {chunk.shape[1]} < requested --action-dim {action_dim}.")
    return chunk[:, :action_dim]


def _infer_chunk(
    client: websocket_client_policy.WebsocketClientPolicy,
    req: dict,
    action_dim: int,
) -> tuple[np.ndarray, float]:
    infer_start = time.monotonic()
    response = client.infer(req)
    infer_time_s = time.monotonic() - infer_start
    return _normalize_chunk(response["actions"], action_dim), infer_time_s


def main(args: Args) -> None:
    _validate_args(args)
    images = _load_observation_images(args)

    client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"[rtc-check] connected to {args.remote_host}:{args.remote_port}")
    print(f"[rtc-check] server metadata: {client.get_server_metadata()}")

    prefix_start = _prefix_start(args)
    print(
        f"[rtc-check] RTC schedule: chunks={args.num_chunks}, action_horizon={args.action_horizon}, "
        f"open_loop_horizon={prefix_start}, delay={args.delay}"
    )

    # --- 第 1 次：delay=0 开环 chunk ---
    req = _build_request(args, images)
    _print_request_summary(req, images)
    previous_chunk, infer_time_s = _infer_chunk(client, req, args.action_dim)
    infer_times_s = [infer_time_s]
    print(
        f"[rtc-check] chunk 00: shape={previous_chunk.shape} "
        f"infer={infer_time_s * 1e3:.0f}ms (delay=0 open-loop)"
    )

    transition_rows: list[dict[str, float | bool | int]] = []
    for chunk_index in range(1, args.num_chunks):
        if previous_chunk.shape[0] < prefix_start + args.delay:
            raise RuntimeError(
                f"previous chunk horizon {previous_chunk.shape[0]} < "
                f"open_loop_horizon+delay {prefix_start + args.delay}; "
                f"check --action-horizon matches the served checkpoint."
            )

        prefix = previous_chunk[prefix_start : prefix_start + args.delay]
        padded_prefix = np.zeros((args.action_horizon, args.action_dim), dtype=np.float32)
        padded_prefix[: prefix.shape[0]] = prefix

        req = _build_request(args, images)
        req["action_prefix"] = padded_prefix
        req["rtc_delay"] = np.asarray(prefix.shape[0], dtype=np.int32)
        chunk, infer_time_s = _infer_chunk(client, req, args.action_dim)
        infer_times_s.append(infer_time_s)
        if chunk.shape[0] <= args.delay:
            raise RuntimeError(
                f"server returned horizon {chunk.shape[0]} <= delay {args.delay}; "
                "cannot evaluate seam step B[delay]."
            )

        pinned = chunk[: args.delay]
        pin_err = np.abs(pinned - prefix)
        pin_max = float(pin_err.max())
        pin_ok = pin_max <= args.pin_atol

        diffs = np.linalg.norm(np.diff(chunk, axis=0), axis=1)
        seam_idx = args.delay - 1
        seam_jump = float(diffs[seam_idx])
        interior = np.delete(diffs, seam_idx)
        if interior.size == 0:
            raise RuntimeError("not enough post-prefix/interior steps to evaluate seam smoothness.")
        interior_median = float(np.median(interior))
        interior_p90 = float(np.percentile(interior, 90))
        seam_ratio = seam_jump / (interior_median + 1e-9)
        seam_ok = seam_ratio <= args.seam_ratio_threshold

        transition_rows.append(
            {
                "index": chunk_index,
                "pin_max": pin_max,
                "pin_ok": pin_ok,
                "seam_jump": seam_jump,
                "interior_median": interior_median,
                "interior_p90": interior_p90,
                "seam_ratio": seam_ratio,
                "seam_ok": seam_ok,
            }
        )
        print(
            f"[rtc-check] chunk {chunk_index:02d}: shape={chunk.shape} "
            f"infer={infer_time_s * 1e3:.0f}ms "
            f"transition {chunk_index - 1:02d}->{chunk_index:02d} "
            f"pin_max={pin_max:.6f} ({'PASS' if pin_ok else 'FAIL'}), "
            f"seam_ratio={seam_ratio:.2f}x ({'PASS' if seam_ok else 'FAIL'})"
        )
        previous_chunk = chunk

    pin_all = all(bool(row["pin_ok"]) for row in transition_rows)
    seam_all = all(bool(row["seam_ok"]) for row in transition_rows)
    pin_max_values = np.asarray([float(row["pin_max"]) for row in transition_rows], dtype=np.float32)
    seam_ratios = np.asarray([float(row["seam_ratio"]) for row in transition_rows], dtype=np.float32)
    seam_jumps = np.asarray([float(row["seam_jump"]) for row in transition_rows], dtype=np.float32)
    interior_medians = np.asarray([float(row["interior_median"]) for row in transition_rows], dtype=np.float32)
    infer_times = np.asarray(infer_times_s, dtype=np.float32)
    worst_pin = max(transition_rows, key=lambda row: float(row["pin_max"]))
    worst_seam = max(transition_rows, key=lambda row: float(row["seam_ratio"]))

    print("\n=== 推理时长 (policy infer RPC) ===")
    print(f"  infer time per chunk (ms) = {np.array2string(infer_times * 1e3, precision=0, floatmode='fixed')}")
    print(
        f"  avg = {float(np.mean(infer_times)) * 1e3:.0f}ms, "
        f"max = {float(np.max(infer_times)) * 1e3:.0f}ms, "
        f"total = {float(np.sum(infer_times)):.2f}s"
    )

    print("\n=== 检查 1: prefix 保真 (pin fidelity) ===")
    print(f"  transitions = {len(transition_rows)}  chunks = {args.num_chunks}")
    print(f"  max |pinned-prefix| per transition = {np.array2string(pin_max_values, precision=6, floatmode='fixed')}")
    print(
        f"  worst transition = {int(worst_pin['index']) - 1:02d}->{int(worst_pin['index']):02d}, "
        f"pin_max={float(worst_pin['pin_max']):.6f}  (阈值 {args.pin_atol})"
    )
    print(f"  => {'PASS' if pin_all else 'FAIL'}")
    if not pin_all:
        print("  [!] 至少一个 transition 的 prefix 没被钉住 → server 端 action_prefix 注入或 delta/normalize 往返有 bug。")

    print("\n=== 检查 2: 缝接平滑 (seam smoothness) ===")
    print(f"  seam ratio per transition = {np.array2string(seam_ratios, precision=2, floatmode='fixed')}")
    print(f"  seam jump per transition = {np.array2string(seam_jumps, precision=5, floatmode='fixed')}")
    print(f"  interior median per transition = {np.array2string(interior_medians, precision=5, floatmode='fixed')}")
    print(
        f"  worst transition = {int(worst_seam['index']) - 1:02d}->{int(worst_seam['index']):02d}, "
        f"ratio={float(worst_seam['seam_ratio']):.2f}x  (阈值 {args.seam_ratio_threshold}x)"
    )
    print(f"  => {'PASS' if seam_all else 'FAIL'}")
    if not seam_all:
        print("  [!] 至少一个 transition 的缝接处跳变显著大于内部 → RTC 采样在 prefix→postfix 边界不连续。")

    print("\n=== 总判定 ===")
    if pin_all and seam_all:
        print("  RTC 训练健康 ✅。实机 crash 根因在执行/并发侧，不要再怀疑训练。")
        print("  下一步：跑 debug_note §3.6 的 --position-filter-coeff 0.3 实验。")
    else:
        print("  RTC 侧确有问题 ❌（见上面 FAIL 项），训练/采样需要继续深挖。")
        if args.exit_nonzero_on_fail:
            raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))

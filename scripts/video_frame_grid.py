#!/usr/bin/env python3
"""Uniformly sample frames from a video and arrange them into a col x row grid.

Usage examples:
  # Default: 1 row x 5 cols, full frame, no timestamp
  uv run scripts/video_frame_grid.py video.mp4

  # Custom layout
  uv run scripts/video_frame_grid.py video.mp4 --col 4 --row 3

  # Dual-view video: keep only left / right half of each frame
  uv run scripts/video_frame_grid.py video.mp4 --side left
  uv run scripts/video_frame_grid.py video.mp4 --side right -o right_preview.png

  # Overlay timestamps
  uv run scripts/video_frame_grid.py video.mp4 --show-time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniformly sample video frames and compose a col x row preview grid."
    )
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument(
        "--col",
        type=int,
        default=5,
        help="Number of columns in the grid. Default: 5",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=1,
        help="Number of rows in the grid. Default: 1",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <video_stem>_grid.png)",
    )
    parser.add_argument(
        "--side",
        choices=("full", "left", "right"),
        default="full",
        help="Crop each frame to left/right half, or keep full frame. Default: full",
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=320,
        help="Width of each cell in pixels (height keeps aspect ratio). Default: 320",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=4,
        help="Gap between cells in pixels. Default: 4",
    )
    parser.add_argument(
        "--show-time",
        action="store_true",
        help="Overlay timestamp (seconds) on each frame (off by default)",
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default="255,255,255",
        help="Background RGB color, e.g. 255,255,255. Default: white",
    )
    return parser.parse_args()


def parse_rgb(text: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3 or any(c < 0 or c > 255 for c in parts):
        raise ValueError(f"Invalid RGB color: {text!r}, expected R,G,B in 0-255")
    return parts[0], parts[1], parts[2]


def crop_side(frame: np.ndarray, side: str) -> np.ndarray:
    """Keep full / left-half / right-half of a frame (width split in half)."""
    if side == "full":
        return frame
    h, w = frame.shape[:2]
    mid = w // 2
    if side == "left":
        return frame[:, :mid]
    if side == "right":
        return frame[:, mid:]
    raise ValueError(f"Unknown side: {side!r}")


def sample_frame_indices(num_frames: int, num_samples: int) -> list[int]:
    """Pick indices evenly spanning [0, num_frames - 1]."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_frames <= 0:
        raise ValueError("video has no frames")
    if num_samples == 1:
        return [0]
    if num_samples >= num_frames:
        # Prefer unique frames; pad last frame if we need more cells than frames.
        indices = list(range(num_frames))
        while len(indices) < num_samples:
            indices.append(num_frames - 1)
        return indices
    # Inclusive endpoints so first/last frames are included.
    positions = np.linspace(0, num_frames - 1, num_samples)
    return [int(round(p)) for p in positions]


def read_frames(video_path: Path, indices: list[int]) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Failed to read frame index {idx} from {video_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()
    return frames, fps


def resize_keep_aspect(frame: np.ndarray, cell_width: int) -> Image.Image:
    h, w = frame.shape[:2]
    cell_height = max(1, int(round(h * cell_width / w)))
    img = Image.fromarray(frame)
    return img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)


def overlay_time(img: Image.Image, seconds: float) -> Image.Image:
    draw = ImageDraw.Draw(img)
    label = f"{seconds:.2f}s"
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    pad = 4
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x0, y0 = pad, pad
    x1, y1 = x0 + tw + 2 * pad, y0 + th + 2 * pad
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    draw.text((x0 + pad, y0 + pad), label, fill=(255, 255, 255), font=font)
    return img


def compose_grid(
    frames: list[Image.Image],
    cols: int,
    rows: int,
    gap: int,
    bg_color: tuple[int, int, int],
) -> Image.Image:
    cell_w = max(im.width for im in frames)
    cell_h = max(im.height for im in frames)
    canvas_w = cols * cell_w + (cols + 1) * gap
    canvas_h = rows * cell_h + (rows + 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    for i, im in enumerate(frames):
        r, c = divmod(i, cols)
        if r >= rows:
            break
        x = gap + c * (cell_w + gap)
        y = gap + r * (cell_h + gap)
        # Center smaller cells (should be rare; all resized to same width).
        ox = x + (cell_w - im.width) // 2
        oy = y + (cell_h - im.height) // 2
        canvas.paste(im, (ox, oy))
    return canvas


def main() -> None:
    args = parse_args()
    if args.col <= 0 or args.row <= 0:
        raise SystemExit("--col and --row must be positive integers")
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")

    num_samples = args.col * args.row
    output = args.output
    if output is None:
        suffix = "" if args.side == "full" else f"_{args.side}"
        output = args.video.with_name(f"{args.video.stem}_grid{suffix}.png")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if num_frames <= 0:
        raise SystemExit(f"Could not determine frame count for: {args.video}")

    indices = sample_frame_indices(num_frames, num_samples)
    raw_frames, _ = read_frames(args.video, indices)

    images: list[Image.Image] = []
    for idx, frame in zip(indices, raw_frames):
        frame = crop_side(frame, args.side)
        img = resize_keep_aspect(frame, args.cell_width)
        if args.show_time:
            seconds = idx / fps if fps > 0 else float(idx)
            img = overlay_time(img, seconds)
        images.append(img)

    grid = compose_grid(images, args.col, args.row, args.gap, parse_rgb(args.bg_color))
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)

    duration = (num_frames / fps) if fps > 0 else float("nan")
    print(
        f"Saved {args.col}x{args.row} grid ({num_samples} frames, side={args.side}) -> {output}\n"
        f"video frames={num_frames}, fps={fps:.3f}, duration≈{duration:.2f}s\n"
        f"sampled indices={indices}"
    )


if __name__ == "__main__":
    main()

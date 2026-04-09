"""Plot top-k EE xyz candidates on a 2D XY (bird's-eye) view."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_topk_xy(
    topk_xyz: np.ndarray,
    probs: np.ndarray,
    *,
    current_xyz: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save_dir: str = "topk_plots",
    filename: str | None = None,
) -> str:
    """Save a bird's-eye (XY) scatter of top-k EE candidates.

    Args:
        topk_xyz: (k, 3) end-effector positions in metres.
        probs: (k,) probability for each candidate.
        current_xyz: (3,) current EE position (optional, drawn as a star).
        xlim: fixed x-axis limits, e.g. (xmin, xmax). If None, matplotlib auto-scales.
        ylim: fixed y-axis limits, e.g. (ymin, ymax). If None, matplotlib auto-scales.
        save_dir: directory to save the plot.
        filename: output filename (auto-generated if None).

    Returns:
        The path of the saved image.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))

    # candidates — size proportional to probability
    sizes = probs / probs.max() * 300 + 50
    scatter = ax.scatter(
        topk_xyz[:, 0],
        topk_xyz[:, 1],
        s=sizes,
        c=np.arange(len(probs)),
        cmap="tab10",
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    # annotate each point
    for i, (xy, p) in enumerate(zip(topk_xyz, probs)):
        ax.annotate(
            f"#{i + 1} {p:.0%}",
            (xy[0], xy[1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    # current position
    if current_xyz is not None:
        ax.scatter(
            current_xyz[0],
            current_xyz[1],
            marker="*",
            s=200,
            c="red",
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
            label="current",
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top-K EE Candidates (XY)")
    ax.set_aspect("equal")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if filename is None:
        idx = len(list(Path(save_dir).glob("topk_*.png")))
        filename = f"topk_{idx:04d}.png"
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

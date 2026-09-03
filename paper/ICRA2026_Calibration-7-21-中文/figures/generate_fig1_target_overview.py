#!/usr/bin/env python3
"""Build the single-column target overview figure from a recorded calibration frame."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORANGE = "#D55E00"
BLUE = "#0072B2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("fig1_target_overview.json"),
    )
    return parser.parse_args()


def resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def extract_crop(
    video_path: Path, time_seconds: float, box: list[int], output_width: int | None = None
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_path = Path(tmp_dir) / "frame.png"
        x, y, width, height = box
        video_filter = f"crop={width}:{height}:{x}:{y}"
        if output_width is not None:
            video_filter += f",scale={output_width}:-2"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{time_seconds:.3f}",
                "-i",
                str(video_path),
                "-vf",
                video_filter,
                "-frames:v",
                "1",
                str(frame_path),
            ],
            check=True,
        )
        return np.asarray(Image.open(frame_path).convert("RGB"))


def crop(image: np.ndarray, box: list[int]) -> np.ndarray:
    x, y, width, height = box
    return image[y : y + height, x : x + width]


def load_crop(path: Path, box: list[int]) -> np.ndarray:
    x, y, width, height = box
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").crop((x, y, x + width, y + height)))


def connected_green_centers(overlay: np.ndarray, box: list[int]) -> np.ndarray:
    """Extract the detector's green internal-point markers in the selected tag."""
    # Use fixed channel thresholds to avoid full-resolution integer copies.
    # The selected crop contains only the rendered marker overlay.
    mask = (
        (overlay[:, :, 1] > 150)
        & (overlay[:, :, 0] < 110)
        & (overlay[:, :, 2] < 150)
    )

    active = {tuple(pixel) for pixel in np.argwhere(mask)}
    centers: list[tuple[float, float]] = []
    while active:
        seed = active.pop()
        queue: deque[tuple[int, int]] = deque([seed])
        pixels = [seed]
        while queue:
            cy, cx = queue.popleft()
            for neighbor in (
                (cy - 1, cx - 1), (cy - 1, cx), (cy - 1, cx + 1),
                (cy, cx - 1), (cy, cx + 1),
                (cy + 1, cx - 1), (cy + 1, cx), (cy + 1, cx + 1),
            ):
                if neighbor in active:
                    active.remove(neighbor)
                    queue.append(neighbor)
                    pixels.append(neighbor)
        if 12 <= len(pixels) <= 500:
            yy, xx = np.asarray(pixels).mean(axis=0)
            centers.append((xx, yy))
    return np.asarray(centers)


def draw_outer_corners(ax: plt.Axes, corners: np.ndarray, offset: np.ndarray) -> None:
    local = corners - offset
    closed = np.vstack([local, local[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=ORANGE, linewidth=1.2, alpha=0.9)
    ax.scatter(
        local[:, 0],
        local[:, 1],
        s=18,
        facecolors="white",
        edgecolors=ORANGE,
        linewidths=1.2,
        zorder=3,
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="white",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.6, "pad": 1.5},
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())

    scene_box = config["scene_crop"]
    focus_box = config["focus_crop"]
    video_path = resolve(config_path, config["video"])
    scene = extract_crop(video_path, config["time_seconds"], scene_box, output_width=1600)
    focus = extract_crop(video_path, config["time_seconds"], focus_box)
    corners = np.asarray(config["focus_outer_corners"], dtype=float)
    overlay_focus = load_crop(resolve(config_path, config["overlay"]), focus_box)
    centers = connected_green_centers(overlay_focus, focus_box)
    if len(centers) < 8:
        raise RuntimeError("Could not recover enough internal-point markers from the detector overlay.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(3.5, 3.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.8, 1.0], hspace=0.02, wspace=0.025)
    ax_scene = fig.add_subplot(grid[0, :])
    ax_outer = fig.add_subplot(grid[1, 0])
    ax_internal = fig.add_subplot(grid[1, 1])

    for ax in (ax_scene, ax_outer, ax_internal):
        ax.set_axis_off()

    ax_scene.imshow(scene)
    add_panel_label(ax_scene, "(a)")

    ax_outer.imshow(focus)
    draw_outer_corners(ax_outer, corners, np.asarray(focus_box[:2], dtype=float))
    add_panel_label(ax_outer, "(b)")

    ax_internal.imshow(focus)
    draw_outer_corners(ax_internal, corners, np.asarray(focus_box[:2], dtype=float))
    ax_internal.scatter(
        centers[:, 0],
        centers[:, 1],
        s=9,
        facecolors=BLUE,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    add_panel_label(ax_internal, "(c)")

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=ORANGE,
               markeredgewidth=1.2, markersize=5, label="Outer corners"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white",
               markeredgewidth=0.35, markersize=5, label="Internal points"),
    ]
    ax_internal.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=1,
        frameon=True,
        framealpha=0.85,
        edgecolor="none",
        fontsize=6.4,
        borderpad=0.3,
        handletextpad=0.35,
    )

    for key in ("output_pdf", "output_png"):
        output = resolve(config_path, config[key])
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.015)
        if not output.is_file():
            raise RuntimeError(f"Figure export failed: {output}")
        print(f"Wrote {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a compact single-column overview of the calibration pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "pic"

CHARCOAL = "#2C3338"
MUTED = "#66727B"
LIGHT = "#F3F5F6"
LINE = "#CBD2D6"
ORANGE = "#D55E00"
BLUE = "#0072B2"
TEAL = "#009E73"
PALE_ORANGE = "#FCF0E6"
PALE_BLUE = "#EAF3F8"
PALE_TEAL = "#E9F4EE"


def arrow(ax, start, end, color=CHARCOAL, width=1.1, mutation=11, **kwargs):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=width,
            color=color,
            shrinkA=0,
            shrinkB=0,
            **kwargs,
        )
    )


def rounded_panel(ax, xy, width, height, color):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.045",
            facecolor=color,
            edgecolor="none",
            zorder=0,
        )
    )


def tag_polygon(center, size, angle=0.0):
    half = size / 2
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return local @ rotation.T + np.asarray(center)


def draw_tag(
    ax,
    center,
    size,
    angle=0.0,
    outer=False,
    internal=False,
    label=None,
    zorder=2,
):
    polygon = tag_polygon(center, size, angle)
    ax.add_patch(
        Polygon(polygon, closed=True, facecolor="white", edgecolor=CHARCOAL, linewidth=0.75, zorder=zorder)
    )
    inner = tag_polygon(center, size * 0.55, angle)
    ax.add_patch(Polygon(inner, closed=True, facecolor=CHARCOAL, edgecolor="none", zorder=zorder + 0.1))

    # A compact, deterministic AprilTag-like code keeps the schematic readable.
    for ix, iy in ((-0.15, 0.16), (0.12, 0.08), (-0.03, -0.12), (0.18, -0.18)):
        point = np.array([ix * size, iy * size])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        point = point @ rotation.T + np.asarray(center)
        ax.add_patch(Circle(point, size * 0.048, facecolor="white", edgecolor="none", zorder=zorder + 0.2))

    if outer:
        corners = polygon
        closed = np.vstack((corners, corners[0]))
        ax.plot(closed[:, 0], closed[:, 1], color=ORANGE, linewidth=0.9, zorder=zorder + 0.8)
        ax.scatter(corners[:, 0], corners[:, 1], s=17, facecolors="white", edgecolors=ORANGE,
                   linewidths=1.1, zorder=zorder + 1)

    if internal:
        grid = np.linspace(-0.23, 0.23, 5)
        gx, gy = np.meshgrid(grid, grid)
        points = np.stack((gx.ravel(), gy.ravel()), axis=1) * size
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = points @ rotation.T + np.asarray(center)
        ax.scatter(points[:, 0], points[:, 1], s=9, facecolors=BLUE, edgecolors="white",
                   linewidths=0.25, zorder=zorder + 1.1)

    if label:
        ax.text(center[0], center[1] - size * 0.72, label, ha="center", va="top", fontsize=6.2, color=MUTED)


def draw_fisheye_frame(ax, center, radius, offset=(0.0, 0.0), alpha=1.0, outer=False):
    ax.add_patch(Circle(center, radius, facecolor="#EDF0F1", edgecolor=CHARCOAL, linewidth=0.7, alpha=alpha))
    tag_specs = [
        ((-0.40, 0.30), 0.28, 0.17),
        ((0.38, 0.25), 0.27, -0.12),
        ((-0.45, -0.28), 0.25, -0.18),
        ((0.37, -0.30), 0.25, 0.12),
        ((0.00, -0.02), 0.22, 0.0),
    ]
    for (dx, dy), scale, angle in tag_specs:
        tag_center = np.asarray(center) + radius * np.array([dx, dy]) + np.asarray(offset)
        draw_tag(ax, tag_center, radius * scale, angle, outer=outer, zorder=3)
    ax.add_patch(Circle(center, radius * 0.035, facecolor=CHARCOAL, edgecolor="white", linewidth=0.35, zorder=4))


def title(ax, x, y, number, text, color):
    ax.add_patch(Circle((x, y), 0.105, facecolor=color, edgecolor="none", zorder=3))
    ax.text(x, y, str(number), ha="center", va="center", fontsize=7.2, color="white", fontweight="bold", zorder=4)
    ax.text(x + 0.16, y, text, ha="left", va="center", fontsize=8.15, color=CHARCOAL, fontweight="bold")


def draw_camera(ax, center, scale=1.0, color=CHARCOAL):
    cx, cy = center
    ax.add_patch(FancyBboxPatch((cx - 0.16 * scale, cy - 0.10 * scale), 0.32 * scale, 0.20 * scale,
                                boxstyle="round,pad=0.01,rounding_size=0.025", facecolor="white",
                                edgecolor=color, linewidth=0.9, zorder=3))
    ax.add_patch(Circle((cx, cy), 0.06 * scale, facecolor=LIGHT, edgecolor=color, linewidth=0.8, zorder=4))
    ax.add_patch(Polygon([(cx - 0.10 * scale, cy + 0.10 * scale), (cx - 0.02 * scale, cy + 0.17 * scale),
                          (cx + 0.08 * scale, cy + 0.10 * scale)], closed=True, facecolor="white",
                         edgecolor=color, linewidth=0.8, zorder=3))


def draw_sphere(ax, center, radius):
    cx, cy = center
    ax.add_patch(Circle(center, radius, facecolor="white", edgecolor=BLUE, linewidth=1.0, zorder=3))
    theta = np.linspace(-0.95, 0.95, 80)
    ax.plot(cx + radius * np.cos(theta), cy + radius * 0.32 * np.sin(theta), color="#86B8D5", linewidth=0.65, zorder=4)
    ax.plot(cx + radius * 0.46 * np.cos(theta + np.pi / 2), cy + radius * np.sin(theta + np.pi / 2),
            color="#86B8D5", linewidth=0.65, zorder=4)
    for angle in (-2.4, -1.0, 0.2, 1.5):
        point = (cx + radius * 0.74 * np.cos(angle), cy + radius * 0.74 * np.sin(angle))
        ax.plot([cx, point[0]], [cy, point[1]], color=ORANGE, linewidth=0.65, alpha=0.85, zorder=5)
    ax.add_patch(Circle(center, radius * 0.08, facecolor=BLUE, edgecolor="white", linewidth=0.35, zorder=6))


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.5, 4.55))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.985)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 4.55)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input sequence.
    ax.text(1.75, 4.40, "Distributed multi-board observations", ha="center", va="center",
            fontsize=8.65, fontweight="bold", color=CHARCOAL)
    for center, radius, alpha in [((1.19, 3.88), 0.40, 0.38), ((1.75, 3.88), 0.40, 0.65), ((2.31, 3.88), 0.40, 1.0)]:
        draw_fisheye_frame(ax, center, radius, alpha=alpha)
    ax.text(1.75, 3.34, "Image sequence with independently identifiable boards", ha="center", va="center",
            fontsize=6.5, color=MUTED)
    arrow(ax, (1.75, 3.18), (0.90, 2.85), color=LINE, width=1.2)

    # Middle-left: Outer4 initialization.
    rounded_panel(ax, (0.06, 1.53), 1.55, 1.19, PALE_ORANGE)
    title(ax, 0.25, 2.57, 1, "Outer-corner bootstrap", ORANGE)
    draw_fisheye_frame(ax, (0.54, 2.07), 0.34, outer=True)
    draw_camera(ax, (1.28, 2.07), 0.82, ORANGE)
    arrow(ax, (0.91, 2.07), (1.05, 2.07), color=ORANGE, width=1.0, mutation=9)
    ax.text(1.28, 1.77, "Intermediate\ncamera model", ha="center", va="top", fontsize=6.5, color=CHARCOAL)

    # Middle-right: internal recovery.
    rounded_panel(ax, (1.89, 1.53), 1.55, 1.19, PALE_BLUE)
    title(ax, 2.08, 2.57, 2, "Viewing-ray recovery", BLUE)
    draw_tag(ax, (2.18, 2.03), 0.47, angle=-0.08, outer=True)
    draw_sphere(ax, (2.70, 2.07), 0.20)
    draw_tag(ax, (3.15, 2.03), 0.47, angle=0.08, outer=True, internal=True)
    arrow(ax, (2.43, 2.07), (2.49, 2.07), color=BLUE, width=0.85, mutation=8)
    arrow(ax, (2.91, 2.07), (2.92, 2.07), color=BLUE, width=0.85, mutation=8)
    ax.text(2.70, 1.65, "Spherical seed", ha="center", va="center", fontsize=6.1, color=MUTED)
    arrow(ax, (1.66, 2.07), (1.84, 2.07), color=CHARCOAL, width=1.25)

    # Final calibration.
    arrow(ax, (2.66, 1.45), (1.75, 1.19), color=LINE, width=1.2)
    rounded_panel(ax, (0.06, 0.10), 3.38, 0.91, PALE_TEAL)
    title(ax, 0.25, 0.86, 3, "Final multi-board calibration", TEAL)
    draw_camera(ax, (0.74, 0.50), 0.88)
    target_centers = [(1.33, 0.72), (1.60, 0.52), (1.34, 0.30)]
    for idx, point in enumerate(target_centers):
        draw_tag(ax, point, 0.18, angle=(-0.14, 0.06, 0.15)[idx], outer=True, internal=True)
        ax.plot([0.88, point[0] - 0.10], [0.50, point[1]], color="#8AB89D", linewidth=0.65, zorder=1)
    arrow(ax, (1.79, 0.50), (2.05, 0.50), color=TEAL, width=1.0, mutation=9)
    ax.add_patch(FancyBboxPatch((2.12, 0.28), 1.04, 0.46, boxstyle="round,pad=0.025,rounding_size=0.025",
                                facecolor="white", edgecolor="#9ECBB1", linewidth=0.75))
    ax.text(2.64, 0.61, "Calibrated camera", ha="center", va="center", fontsize=7.0, color=CHARCOAL, fontweight="bold")
    ax.text(2.64, 0.42, r"$\boldsymbol{\theta},\ \mathbf{T}_{B_0B_b},\ \mathbf{T}_{C_tB_0}$",
            ha="center", va="center", fontsize=6.6, color=TEAL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = OUTPUT_DIR / f"pipeline_overview.{extension}"
        fig.savefig(output, dpi=360, bbox_inches="tight", pad_inches=0.01)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot all frontend outer-corner observations in one image-plane figure.

The Stage5 ``auto_camera_initialization_bootstrap_views.csv`` artifact stores
the four physical outer corners for every frame-board observation that reached
the frontend bootstrap stage.  This tool aggregates those rows; it does not
select frames for BA and it does not rerun camera calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


PALETTE = [
    "#0F4D92", "#3775BA", "#B64342", "#42949E", "#9A4D8E",
    "#D97706", "#2F855A", "#6B46C1", "#C05621", "#4A5568",
]
MARKERS = ("o", "s", "^", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True,
                        help="auto_camera_initialization_bootstrap_views.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=float, default=4512.0)
    parser.add_argument("--height", type=float, default=4512.0)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--include-unsuccessful", action="store_true",
                        help="Include rows marked pose_init_success=0 (default keeps successful rows).")
    return parser.parse_args()


def load_rows(path: Path, include_unsuccessful: bool) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"frame_index", "board_id", "corner_index", "x", "y", "pose_init_success"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
    if not include_unsuccessful:
        rows = [row for row in rows if row.get("pose_init_success", "1") == "1"]
    if not rows:
        raise RuntimeError(f"No rows remain after filtering {path}")
    return rows


def convex_hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return abs(sum(
        hull[i][0] * hull[(i + 1) % len(hull)][1]
        - hull[(i + 1) % len(hull)][0] * hull[i][1]
        for i in range(len(hull))
    )) * 0.5


def summarize(rows: list[dict[str, str]], width: float, height: float) -> dict[str, object]:
    x = np.asarray([float(row["x"]) for row in rows])
    y = np.asarray([float(row["y"]) for row in rows])
    radius = min(width / 2.0, height / 2.0)
    rho = np.hypot((x - width / 2) / radius, (y - height / 2) / radius)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(rho)
    valid = finite & (rho <= 1.0)
    grid = 20
    normalized_x = (x[valid] - width / 2.0) / radius
    normalized_y = (y[valid] - height / 2.0) / radius
    gx = np.clip(((normalized_x + 1.0) * 0.5 * grid).astype(int), 0, grid - 1)
    gy = np.clip(((normalized_y + 1.0) * 0.5 * grid).astype(int), 0, grid - 1)
    valid_cells = {
        (cell_x, cell_y)
        for cell_y in range(grid)
        for cell_x in range(grid)
        if np.hypot(
            -1.0 + (cell_x + 0.5) * 2.0 / grid,
            -1.0 + (cell_y + 0.5) * 2.0 / grid,
        ) <= 1.0
    }
    occupied_cells = set(zip(gx.tolist(), gy.tolist())) & valid_cells
    return {
        "source_rows": len(rows),
        "frame_count": len({row["frame_index"] for row in rows}),
        "frame_board_count": len({(row["frame_index"], row["board_id"]) for row in rows}),
        "board_ids": sorted({int(row["board_id"]) for row in rows}),
        "outer_corner_count_by_corner_index": {
            str(index): sum(row["corner_index"] == str(index) for row in rows)
            for index in range(4)
        },
        "valid_unit_disc_fraction": float(np.mean(valid)),
        "radial_p05": float(np.percentile(rho[finite], 5)),
        "radial_p50": float(np.percentile(rho[finite], 50)),
        "radial_p95": float(np.percentile(rho[finite], 95)),
        "peripheral_fraction_rho_ge_0_7": float(np.mean(rho[valid] >= 0.7)),
        "grid_occupied_cells": len(occupied_cells),
        "grid_total_cells": len(valid_cells),
        "grid_occupancy_fraction": float(len(occupied_cells) / len(valid_cells)),
        "convex_hull_area_fraction": float(convex_hull_area(np.c_[normalized_x, normalized_y]) / np.pi),
    }


def plot(rows: list[dict[str, str]], summary: dict[str, object], args: argparse.Namespace) -> plt.Figure:
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(13.5, 6.3), constrained_layout=True)
    board_ids = sorted({int(row["board_id"]) for row in rows})
    colors = {board_id: PALETTE[index % len(PALETTE)] for index, board_id in enumerate(board_ids)}

    for board_id in board_ids:
        board_rows = [row for row in rows if int(row["board_id"]) == board_id]
        ax.scatter(
            [float(row["x"]) for row in board_rows],
            [float(row["y"]) for row in board_rows],
            s=13, alpha=0.30, linewidths=0, color=colors[board_id],
        )
        ax_zoom.scatter(
            [float(row["x"]) for row in board_rows],
            [float(row["y"]) for row in board_rows],
            s=17, alpha=0.35, linewidths=0, color=colors[board_id],
        )

    # Show one representative marker for each physical corner index.  The
    # actual points remain colour-coded by board, while shape explains corner.
    for corner_index, marker in enumerate(MARKERS):
        corner_rows = [row for row in rows if int(row["corner_index"]) == corner_index]
        if corner_rows:
            ax.scatter([], [], marker=marker, color="#272727", s=45,
                       label=f"corner {corner_index}")
            ax_zoom.scatter([], [], marker=marker, color="#272727", s=45)

    for axis in (ax, ax_zoom):
        radius = min(args.width / 2.0, args.height / 2.0)
        axis.add_patch(Circle((args.width / 2.0, args.height / 2.0), radius,
                              fill=False, edgecolor="#7C828A", linewidth=0.8,
                              linestyle=(0, (3.0, 2.0))))
        axis.add_patch(Rectangle((0, 0), args.width, args.height, fill=False,
                                 edgecolor="#272727", linewidth=1.2))
        axis.axvline(args.width / 2, color="#999999", linewidth=0.7, alpha=0.45)
        axis.axhline(args.height / 2, color="#999999", linewidth=0.7, alpha=0.45)
        axis.set_xlim(0, args.width)
        axis.set_ylim(args.height, 0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("image x [px]")
        axis.set_ylabel("image y [px]")

    # The right panel zooms the useful support region while retaining the
    # complete image-space panel on the left.
    margin = 250.0
    ax_zoom.set_xlim(-margin, args.width + margin)
    ax_zoom.set_ylim(args.height + margin, -margin)
    ax.set_title("All frontend outer-corner observations")
    ax_zoom.set_title("Image-plane support (zoomed)")
    board_handles = [Line2D([], [], linestyle="", marker="o", color=colors[board_id],
                            markersize=6, label=f"board {board_id}") for board_id in board_ids]
    corner_handles = [Line2D([], [], linestyle="", marker=marker, color="#272727",
                             markersize=6, label=f"corner {index}")
                      for index, marker in enumerate(MARKERS)]
    ax.legend(handles=board_handles, loc="upper right", title="board colour",
              ncol=2, fontsize=9)
    ax_zoom.legend(handles=board_handles + corner_handles, loc="upper right",
                   title="board / physical corner", ncol=2, fontsize=8)
    fig.suptitle(
        f"Frontend outer-corner distribution · {summary['source_rows']} points · "
        f"{summary['frame_count']} frames · {summary['frame_board_count']} frame-board observations",
        fontsize=14,
    )
    fig.text(
        0.5, 0.005,
        f"rho P95={summary['radial_p95']:.3f} · rho≥0.7={summary['peripheral_fraction_rho_ge_0_7']:.1%} · "
        f"20×20 occupancy={summary['grid_occupancy_fraction']:.1%} · "
        f"convex-hull area={summary['convex_hull_area_fraction']:.1%}",
        ha="center", va="bottom", fontsize=10, color="#4A5568",
    )
    return fig


def main() -> None:
    args = parse_args()
    rows = load_rows(args.views, args.include_unsuccessful)
    summary = summarize(rows, args.width, args.height)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frontend_outer_corner_distribution_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "frontend_outer_corner_distribution_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())
    figure = plot(rows, summary, args)
    stem = args.output_dir / "frontend_outer_corner_distribution"
    figure.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"png: {stem}.png")
    print(f"pdf: {stem}.pdf")


if __name__ == "__main__":
    main()

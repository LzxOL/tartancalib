#!/usr/bin/env python3
"""Create publication-style visual evidence for corner-distribution bias."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.io import loadmat


COLORS = {
    "checker": "#2563eb",
    "multi": "#dc2626",
    "center": "#111827",
    "board1": "#0072B2",
    "board2": "#E69F00",
    "board3": "#009E73",
    "board4": "#D55E00",
    "board5": "#CC79A7",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkerboard-right", type=Path, required=True)
    parser.add_argument("--multi-right", type=Path, required=True)
    parser.add_argument("--checkerboard-left", type=Path, required=True)
    parser.add_argument("--multi-left", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def field(obj: Any, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    raise KeyError(name)


def structs(value: Any) -> list[Any]:
    return list(np.asarray(value).reshape(-1))


def read_mat(path: Path) -> dict[str, Any]:
    data = loadmat(path, squeeze_me=False, struct_as_record=False)
    image_size = np.asarray(data["imgsize"], dtype=float).reshape(-1)
    width, height = image_size
    board_count = len(structs(data["boards"]))
    outer_ids = {board_id: {1, 2, 3, 4} for board_id in range(1, board_count + 1)}
    points: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    frame_board_points: list[dict[int, np.ndarray]] = []
    outer_points: list[np.ndarray] = []
    outer_frames: list[np.ndarray] = []
    outer_frame_board_points: list[dict[int, np.ndarray]] = []
    for corner in structs(data["corners"]):
        xy = np.asarray(field(corner, "x"), dtype=float).reshape(2, -1).T
        corr = np.asarray(field(corner, "cspond"), dtype=float).reshape(2, -1).T.astype(int)
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(corr).all(axis=1)
        xy, corr = xy[valid], corr[valid]
        points.append(xy)
        frames.append(xy.mean(axis=0) if len(xy) else np.array([np.nan, np.nan]))
        by_board = {}
        for board_id in np.unique(corr[:, 1]) if len(corr) else []:
            by_board[int(board_id)] = xy[corr[:, 1] == board_id]
        frame_board_points.append(by_board)
        outer_by_board = {}
        outer_mask = np.zeros(len(corr), dtype=bool)
        for board_id in np.unique(corr[:, 1]) if len(corr) else []:
            board_mask = (corr[:, 1] == board_id) & np.isin(corr[:, 0], list(outer_ids.get(int(board_id), set())))
            outer_mask |= board_mask
            outer_by_board[int(board_id)] = xy[board_mask]
        outer_xy = xy[outer_mask]
        outer_points.append(outer_xy)
        outer_frames.append(outer_xy.mean(axis=0) if len(outer_xy) else np.array([np.nan, np.nan]))
        outer_frame_board_points.append(outer_by_board)
    return {
        "path": path,
        "width": width,
        "height": height,
        "points": np.vstack(points) if points else np.empty((0, 2)),
        "frames": np.asarray(frames),
        "frame_board_points": frame_board_points,
        "outer_points": np.vstack(outer_points) if outer_points and any(len(x) for x in outer_points) else np.empty((0, 2)),
        "outer_frames": np.asarray(outer_frames),
        "outer_frame_board_points": outer_frame_board_points,
        "outer_point_count": int(sum(len(x) for x in outer_points)),
    }


def setup_axis(ax: plt.Axes, width: float, height: float) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")
    ax.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.55)
    ax.axvline(width / 2, color=COLORS["center"], linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axhline(height / 2, color=COLORS["center"], linestyle="--", linewidth=0.8, alpha=0.8)


def add_mean_marker(ax: plt.Axes, points: np.ndarray, label: str, color: str) -> np.ndarray:
    mean = np.nanmean(points, axis=0)
    ax.scatter(*mean, marker="*", s=90, color=color, edgecolor="white", linewidth=0.7, zorder=5, label=label)
    return mean


def covariance_ellipse(ax: plt.Axes, points: np.ndarray, color: str, label: str) -> None:
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return
    mean = points.mean(axis=0)
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = Ellipse(
        xy=mean,
        width=2.0 * np.sqrt(eigenvalues[0]),
        height=2.0 * np.sqrt(eigenvalues[1]),
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=1.8,
        linestyle="-",
        label=label,
        zorder=4,
    )
    ax.add_patch(ellipse)
    ax.plot(
        [mean[0], mean[0] + np.sqrt(eigenvalues[0]) * eigenvectors[0, 0]],
        [mean[1], mean[1] + np.sqrt(eigenvalues[0]) * eigenvectors[1, 0]],
        color=color,
        linewidth=2.0,
        zorder=5,
    )


def plot_coverage(datasets: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.45), constrained_layout=True)
    for ax, title, checker, multi in [
        (axes[0], "Right camera: 1444190", datasets["checker_right"], datasets["multi_right"]),
        (axes[1], "Left camera: 144928", datasets["checker_left"], datasets["multi_left"]),
    ]:
        setup_axis(ax, checker["width"], checker["height"])
        ax.scatter(checker["points"][:, 0], checker["points"][:, 1], s=1.2, alpha=0.07, color=COLORS["checker"], rasterized=True)
        for board_id in range(1, 6):
            board_points = np.vstack([
                frame[board_id] for frame in multi["outer_frame_board_points"] if board_id in frame
            ]) if any(board_id in frame for frame in multi["outer_frame_board_points"]) else np.empty((0, 2))
            if len(board_points):
                ax.scatter(board_points[:, 0], board_points[:, 1], s=2.0, alpha=0.18,
                           color=COLORS[f"board{board_id}"], rasterized=True)
        checker_mean = add_mean_marker(ax, checker["points"], "checkerboard mean", COLORS["checker"])
        multi_mean = add_mean_marker(ax, multi["outer_points"], "multi-board outer mean", COLORS["multi"])
        ax.set_title(title)
        ax.text(0.02, 0.03, f"checker mean=({checker_mean[0]:.0f}, {checker_mean[1]:.0f})\n"
                f"multi mean=({multi_mean[0]:.0f}, {multi_mean[1]:.0f})",
                transform=ax.transAxes, fontsize=8.2, va="bottom",
                bbox=dict(facecolor="white", edgecolor="#d1d5db", alpha=0.9, pad=3.0))
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["checker"], markersize=5, alpha=0.7, label="checkerboard points"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["board1"], markersize=5, alpha=0.7, label="multi-board points"),
        Line2D([0], [0], marker="*", color=COLORS["checker"], markerfacecolor=COLORS["checker"], markersize=8, label="checkerboard mean"),
        Line2D([0], [0], marker="*", color=COLORS["multi"], markerfacecolor=COLORS["multi"], markersize=8, label="multi-board mean"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_motion(datasets: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.45), constrained_layout=True)
    for ax, title, checker, multi in [
        (axes[0], "Right camera: frame-centroid motion", datasets["checker_right"], datasets["multi_right"]),
        (axes[1], "Left camera: frame-centroid motion", datasets["checker_left"], datasets["multi_left"]),
    ]:
        setup_axis(ax, checker["width"], checker["height"])
        checker_frames = checker["frames"]
        multi_frames = multi["outer_frames"]
        ax.plot(checker_frames[:, 0], checker_frames[:, 1], color=COLORS["checker"], linewidth=1.0, alpha=0.6)
        ax.scatter(checker_frames[:, 0], checker_frames[:, 1], color=COLORS["checker"], s=10, alpha=0.65, label="checkerboard frames")
        ax.plot(multi_frames[:, 0], multi_frames[:, 1], color=COLORS["multi"], linewidth=1.1, alpha=0.75)
        ax.scatter(multi_frames[:, 0], multi_frames[:, 1], color=COLORS["multi"], s=14, alpha=0.85, label="multi-board frames")
        covariance_ellipse(ax, checker_frames, COLORS["checker"], "checkerboard 1-sigma")
        covariance_ellipse(ax, multi_frames, COLORS["multi"], "multi-board 1-sigma")
        ax.scatter(checker_frames[:, 0].mean(), checker_frames[:, 1].mean(), color=COLORS["checker"], marker="*", s=90, edgecolor="white", linewidth=0.7, zorder=6)
        ax.scatter(multi_frames[:, 0].mean(), multi_frames[:, 1].mean(), color=COLORS["multi"], marker="*", s=90, edgecolor="white", linewidth=0.7, zorder=6)
        ax.set_title(title)
    legend = [
        Line2D([0], [0], marker="o", color=COLORS["checker"], markersize=5, label="checkerboard frame centroid"),
        Line2D([0], [0], marker="o", color=COLORS["multi"], markersize=5, label="multi-board frame centroid"),
        Line2D([0], [0], color=COLORS["checker"], label="checkerboard 1-sigma ellipse"),
        Line2D([0], [0], color=COLORS["multi"], label="multi-board 1-sigma ellipse"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_offsets(datasets: dict[str, dict[str, Any]], output: Path) -> None:
    labels = ["Right\n1444190", "Left\n144928"]
    values = []
    for checker_key, multi_key in [("checker_right", "multi_right"), ("checker_left", "multi_left")]:
        checker, multi = datasets[checker_key], datasets[multi_key]
        image_center = np.array([checker["width"], checker["height"]]) / 2.0
        values.append([
            checker["points"].mean(axis=0) - image_center,
            multi["outer_points"].mean(axis=0) - image_center,
        ])
    values = np.asarray(values)
    x = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), constrained_layout=True)
    for dim, axis_label, axis in [(0, "u offset [px]", axes[0]), (1, "v offset [px]", axes[1])]:
        checker_values = values[:, 0, dim]
        multi_values = values[:, 1, dim]
        axis.bar(x - width / 2, checker_values, width, color=COLORS["checker"], label="checkerboard")
        axis.bar(x + width / 2, multi_values, width, color=COLORS["multi"], label="multi-board")
        axis.axhline(0, color=COLORS["center"], linewidth=0.8)
        axis.set_xticks(x, labels)
        axis.set_ylabel(axis_label)
        axis.set_title("Mean corner centroid relative to image center")
        axis.grid(axis="y", color="#d1d5db", linewidth=0.45, alpha=0.55)
        for positions, series in [(x - width / 2, checker_values), (x + width / 2, multi_values)]:
            for position, value in zip(positions, series):
                axis.text(position, value + (8 if value >= 0 else -8), f"{value:+.0f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_board_motion(multi: dict[str, Any], output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.45), constrained_layout=True)
    setup_axis(ax, multi["width"], multi["height"])
    for board_id in range(1, 6):
        frames = []
        for frame in multi["outer_frame_board_points"]:
            if board_id in frame:
                frames.append(frame[board_id].mean(axis=0))
        if not frames:
            continue
        frames = np.asarray(frames)
        color = COLORS[f"board{board_id}"]
        ax.plot(frames[:, 0], frames[:, 1], color=color, linewidth=1.2, alpha=0.75)
        ax.scatter(frames[:, 0], frames[:, 1], color=color, s=14, alpha=0.78)
        mean = frames.mean(axis=0)
        ax.scatter(*mean, marker="X", s=75, color=color, edgecolor="white", linewidth=0.7, zorder=5)
        ax.annotate(f"B{board_id}", mean, xytext=(5, -5), textcoords="offset points", fontsize=8.5, color=color, weight="bold")
    ax.set_title(title)
    ax.text(0.02, 0.03, "Each colored trajectory is one board across frames", transform=ax.transAxes, fontsize=8.3,
            bbox=dict(facecolor="white", edgecolor="#d1d5db", alpha=0.9, pad=3.0))
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_checkerboard_only(datasets: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
    cases = [
        (axes[0, 0], "Right camera: all checkerboard corners", datasets["checker_right"], "coverage"),
        (axes[0, 1], "Left camera: all checkerboard corners", datasets["checker_left"], "coverage"),
        (axes[1, 0], "Right camera: checkerboard frame centroids", datasets["checker_right"], "motion"),
        (axes[1, 1], "Left camera: checkerboard frame centroids", datasets["checker_left"], "motion"),
    ]
    for ax, title, checker, mode in cases:
        setup_axis(ax, checker["width"], checker["height"])
        if mode == "coverage":
            ax.scatter(checker["points"][:, 0], checker["points"][:, 1], s=1.5, alpha=0.11,
                       color=COLORS["checker"], rasterized=True)
            mean = add_mean_marker(ax, checker["points"], "corner mean", COLORS["checker"])
            ax.text(0.02, 0.03, f"mean=({mean[0]:.0f}, {mean[1]:.0f})",
                    transform=ax.transAxes, fontsize=8.4, va="bottom",
                    bbox=dict(facecolor="white", edgecolor="#d1d5db", alpha=0.9, pad=3.0))
        else:
            frames = checker["frames"]
            ax.plot(frames[:, 0], frames[:, 1], color=COLORS["checker"], linewidth=1.0, alpha=0.65)
            ax.scatter(frames[:, 0], frames[:, 1], color=COLORS["checker"], s=14, alpha=0.8)
            covariance_ellipse(ax, frames, COLORS["checker"], "1-sigma ellipse")
            ax.scatter(frames[:, 0].mean(), frames[:, 1].mean(), color=COLORS["checker"], marker="*", s=100,
                       edgecolor="white", linewidth=0.7, zorder=6)
        ax.set_title(title)
    fig.suptitle("Checkerboard-only corner distribution", y=1.01, fontsize=13)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_multiboard_only(datasets: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
    cases = [
        (axes[0, 0], "Right camera: initialization outer corners", datasets["multi_right"], "coverage"),
        (axes[0, 1], "Left camera: initialization outer corners", datasets["multi_left"], "coverage"),
        (axes[1, 0], "Right camera: per-board centroid trajectories", datasets["multi_right"], "motion"),
        (axes[1, 1], "Left camera: per-board centroid trajectories", datasets["multi_left"], "motion"),
    ]
    for ax, title, multi, mode in cases:
        setup_axis(ax, multi["width"], multi["height"])
        if mode == "coverage":
            for board_id in range(1, 6):
                board_points = np.vstack([
                    frame[board_id] for frame in multi["outer_frame_board_points"] if board_id in frame
                ]) if any(board_id in frame for frame in multi["outer_frame_board_points"]) else np.empty((0, 2))
                if len(board_points):
                    ax.scatter(board_points[:, 0], board_points[:, 1], s=2.1, alpha=0.23,
                               color=COLORS[f"board{board_id}"], rasterized=True,
                               label=f"board {board_id}")
            mean = add_mean_marker(ax, multi["outer_points"], "outer-corner mean", COLORS["multi"])
            ax.text(0.02, 0.03, f"mean=({mean[0]:.0f}, {mean[1]:.0f})",
                    transform=ax.transAxes, fontsize=8.4, va="bottom",
                    bbox=dict(facecolor="white", edgecolor="#d1d5db", alpha=0.9, pad=3.0))
        else:
            for board_id in range(1, 6):
                frames = []
                for frame in multi["outer_frame_board_points"]:
                    if board_id in frame:
                        frames.append(frame[board_id].mean(axis=0))
                if not frames:
                    continue
                frames = np.asarray(frames)
                color = COLORS[f"board{board_id}"]
                ax.plot(frames[:, 0], frames[:, 1], color=color, linewidth=1.1, alpha=0.78)
                ax.scatter(frames[:, 0], frames[:, 1], color=color, s=14, alpha=0.8, label=f"board {board_id}")
                mean = frames.mean(axis=0)
                ax.scatter(*mean, marker="X", s=65, color=color, edgecolor="white", linewidth=0.7, zorder=5)
                ax.annotate(f"B{board_id}", mean, xytext=(4, -4), textcoords="offset points", fontsize=8, color=color, weight="bold")
        ax.set_title(title)
    handles = [Line2D([0], [0], marker="o", color=COLORS[f"board{i}"], markersize=5, label=f"board {i}") for i in range(1, 6)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Multi-board-only outer-corner distribution", y=1.01, fontsize=13)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli = args()
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "checker_right": read_mat(cli.checkerboard_right.resolve()),
        "multi_right": read_mat(cli.multi_right.resolve()),
        "checker_left": read_mat(cli.checkerboard_left.resolve()),
        "multi_left": read_mat(cli.multi_left.resolve()),
    }
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    })
    plot_coverage(datasets, cli.out_dir / "corner_coverage_overlay")
    plot_motion(datasets, cli.out_dir / "centroid_motion_and_covariance")
    plot_offsets(datasets, cli.out_dir / "centroid_offset_summary")
    plot_board_motion(datasets["multi_right"], cli.out_dir / "right_multiboard_board_motion", "Right camera: independent board trajectories")
    plot_board_motion(datasets["multi_left"], cli.out_dir / "left_multiboard_board_motion", "Left camera: independent board trajectories")
    plot_checkerboard_only(datasets, cli.out_dir / "checkerboard_only_distribution")
    plot_multiboard_only(datasets, cli.out_dir / "multiboard_only_distribution")
    print(f"Generated figures in {cli.out_dir.resolve()}")


if __name__ == "__main__":
    main()

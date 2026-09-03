#!/usr/bin/env python3
"""Plot board-pose excitation from Stage5 pose-excitation CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


BOARD_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-csv", type=Path, required=True)
    parser.add_argument("--checker-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = []
        for row in csv.DictReader(stream):
            rows.append({key: float(value) if key not in ("frame_label",) else value for key, value in row.items()})
        return rows


def axis_balance(rows: list[dict[str, float]]) -> float:
    xy = np.array([[row["normal_x"], row["normal_y"]] for row in rows], dtype=float)
    covariance = np.cov(xy.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else 0.0


def add_normal_ellipse(ax: plt.Axes, rows: list[dict[str, float]], color: str) -> None:
    xy = np.array([[row["normal_x"], row["normal_y"]] for row in rows], dtype=float)
    if len(xy) < 3:
        return
    mean = xy.mean(axis=0)
    covariance = np.cov(xy.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ax.add_patch(Ellipse(
        mean,
        width=2 * np.sqrt(eigenvalues[0]),
        height=2 * np.sqrt(eigenvalues[1]),
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=1.6,
    ))
    ax.plot(
        [mean[0], mean[0] + np.sqrt(eigenvalues[0]) * eigenvectors[0, 0]],
        [mean[1], mean[1] + np.sqrt(eigenvalues[0]) * eigenvectors[1, 0]],
        color=color,
        linewidth=1.8,
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_dataset(rows: list[dict[str, float]], name: str, output: Path, multi: bool) -> None:
    board_ids = sorted({int(row["board_id"]) for row in rows})
    grouped = {board_id: [row for row in rows if int(row["board_id"]) == board_id] for board_id in board_ids}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), constrained_layout=True)

    ax = axes[0]
    for index, board_id in enumerate(board_ids):
        group = grouped[board_id]
        color = BOARD_COLORS[index % len(BOARD_COLORS)] if multi else "#2563eb"
        x = np.array([row["normal_x"] for row in group])
        y = np.array([row["normal_y"] for row in group])
        ax.plot(x, y, color=color, linewidth=1.0, alpha=0.72)
        ax.scatter(x, y, color=color, s=13, alpha=0.78, label=f"board {board_id}")
        add_normal_ellipse(ax, group, color)
    ax.axhline(0, color="#111827", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="#111827", linewidth=0.7, linestyle="--")
    ax.set_xlabel("normal x")
    ax.set_ylabel("normal y")
    ax.set_title("Board normal trajectory")
    style_axes(ax)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1]
    for index, board_id in enumerate(board_ids):
        group = grouped[board_id]
        color = BOARD_COLORS[index % len(BOARD_COLORS)] if multi else "#2563eb"
        tilt = np.array([row["tilt_deg"] for row in group])
        ax.plot(np.arange(len(tilt)), tilt, color=color, linewidth=1.1, alpha=0.85, label=f"board {board_id}")
    ax.set_xlabel("observation index")
    ax.set_ylabel("tilt [deg]")
    ax.set_title("Per-board tilt variation")
    style_axes(ax)
    if multi:
        ax.legend(frameon=False, ncol=2, loc="best")
    else:
        ax.legend(frameon=False, loc="best")
    fig.suptitle(name, fontsize=13)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metrics(multi_rows: list[dict[str, float]], checker_rows: list[dict[str, float]], output: Path) -> None:
    grouped = {board_id: [row for row in multi_rows if int(row["board_id"]) == board_id] for board_id in range(1, 6)}
    labels = [f"B{i}" for i in range(1, 6)] + ["checkerboard"]
    balances = [axis_balance(grouped[i]) for i in range(1, 6)] + [axis_balance(checker_rows)]
    tilt_ranges = [
        max(row["tilt_deg"] for row in grouped[i]) - min(row["tilt_deg"] for row in grouped[i])
        for i in range(1, 6)
    ] + [max(row["tilt_deg"] for row in checker_rows) - min(row["tilt_deg"] for row in checker_rows)]
    colors = BOARD_COLORS + ["#2563eb"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), constrained_layout=True)
    x = np.arange(len(labels))
    for ax, values, ylabel, title in [
        (axes[0], balances, "axis-balance (minor/major variance)", "2-D normal excitation"),
        (axes[1], tilt_ranges, "tilt range [deg]", "Orientation range"),
    ]:
        bars = ax.bar(x, values, color=colors, width=0.7)
        ax.set_xticks(x, labels, rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        style_axes(ax)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                    ha="center", va="bottom", fontsize=7.8)
    fig.suptitle("Quantitative pose-excitation audit", fontsize=13)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    multi_rows = read_csv(cli.multi_csv.resolve())
    checker_rows = read_csv(cli.checker_csv.resolve())
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "savefig.dpi": 300,
    })
    plot_dataset(multi_rows, "Multi-board 1444190: pose excitation", cli.output_dir / "multiboard_pose_excitation", multi=True)
    plot_dataset(checker_rows, "Checkerboard: pose excitation", cli.output_dir / "checkerboard_pose_excitation", multi=False)
    plot_metrics(multi_rows, checker_rows, cli.output_dir / "pose_excitation_quantitative_comparison")
    print(f"Generated pose-excitation figures in {cli.output_dir.resolve()}")


if __name__ == "__main__":
    main()

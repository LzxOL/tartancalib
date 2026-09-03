#!/usr/bin/env python3
"""Summarize fixed AprilGrid reprojection errors by projected scale.

The input is the long-form per-corner table emitted by
``analyze_aprilgrid_kb_geometry.py``. Frames keep their precomputed geometric
scale labels; no frame or corner is selected from its reprojection error.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCALE_ORDER = ("small", "medium", "large")
METHOD_ORDER = ("ours", "kalibr")
METHOD_LABELS = {"ours": "Ours", "kalibr": "Kalibr"}
COLORS = {"ours": "#0072B2", "kalibr": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-corner", type=Path, required=True)
    parser.add_argument(
        "--per-frame",
        type=Path,
        required=True,
        help="Canonical frame geometry table that assigns one shared scale bin per frame.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(corner_path: Path, frame_path: Path) -> list[dict[str, object]]:
    frame_scale: dict[tuple[str, str], str] = {}
    with frame_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame_scale[(row["dataset"], row["frame_index"])] = row["scale_bin"]

    errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    frames: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    with corner_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame_key = (row["dataset"], row["frame_index"])
            scale = frame_scale.get(frame_key, "unknown")
            method = row["method"]
            if scale not in SCALE_ORDER or method not in METHOD_ORDER:
                continue
            key = (scale, method)
            errors[key].append(float(row["error_px"]))
            frames[key].add(frame_key)

    rows: list[dict[str, object]] = []
    for scale in SCALE_ORDER:
        for method in METHOD_ORDER:
            values = np.asarray(errors[(scale, method)], dtype=np.float64)
            if values.size == 0:
                raise ValueError(f"No values for {scale}/{method}")
            rows.append(
                {
                    "scale": scale,
                    "method": METHOD_LABELS[method],
                    "frame_count": len(frames[(scale, method)]),
                    "corner_count": int(values.size),
                    "rmse_px": math.sqrt(float(np.mean(values * values))),
                    "p95_px": float(np.quantile(values, 0.95)),
                    "inlier_at_1px_percent": float(np.mean(values <= 1.0) * 100.0),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lookup = {(str(row["scale"]), str(row["method"])): row for row in rows}
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Scale & Method & Frames & Corners & RMSE $\downarrow$ & P95 $\downarrow$ & Inlier@1px $\uparrow$ \\",
        r"\midrule",
    ]
    for scale_index, scale in enumerate(SCALE_ORDER):
        for method in ("Ours", "Kalibr"):
            row = lookup[(scale, method)]
            lines.append(
                f"{scale.capitalize()} & {method} & {row['frame_count']} & "
                f"{row['corner_count']} & {row['rmse_px']:.3f} & "
                f"{row['p95_px']:.3f} & {row['inlier_at_1px_percent']:.2f} \\\\"
            )
        if scale_index != len(SCALE_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_figure(output: Path, rows: list[dict[str, object]]) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    lookup = {(str(row["scale"]), str(row["method"])): row for row in rows}
    x = np.arange(len(SCALE_ORDER), dtype=np.float64)
    width = 0.36
    metrics = (
        ("rmse_px", "RMSE [px]", "%.2f"),
        ("p95_px", "P95 [px]", "%.2f"),
        ("inlier_at_1px_percent", "Inlier@1px [%%]", "%.1f"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), constrained_layout=True)
    for ax, (field, ylabel, label_format) in zip(axes, metrics):
        for method_index, method in enumerate(("Ours", "Kalibr")):
            values = [float(lookup[(scale, method)][field]) for scale in SCALE_ORDER]
            offset = (-0.5 if method_index == 0 else 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=method,
                color=COLORS[method.lower()],
                edgecolor="black",
                linewidth=0.4,
            )
            for bar, value in zip(bars, values):
                ax.annotate(
                    label_format % value,
                    (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x, ("Small", "Medium", "Large"))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.18)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = summarize(args.per_corner.resolve(), args.per_frame.resolve())
    write_csv(args.output / "scale_stratified_metrics.csv", rows)
    write_latex(args.output / "scale_stratified_table.tex", rows)
    draw_figure(args.output / "scale_stratified_metrics", rows)


if __name__ == "__main__":
    main()

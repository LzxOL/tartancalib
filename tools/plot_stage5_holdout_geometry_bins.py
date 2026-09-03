#!/usr/bin/env python3
"""Render compact publication-style summaries from a geometry-bin audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("ours", "kalibr", "basalt", "tartancalib")
LABELS = {"ours": "Ours", "kalibr": "Kalibr", "basalt": "Basalt", "tartancalib": "TartanCalib"}
COLORS = {"ours": "#1b9e77", "kalibr": "#d95f02", "basalt": "#7570b3", "tartancalib": "#666666"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matrix(rows: list[dict[str, str]], factor: str, bins: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ours = {(row["bin"]): float(row["rmse_px"]) for row in rows if row["factor"] == factor and row["method"] == "ours"}
    kalibr = {(row["bin"]): float(row["rmse_px"]) for row in rows if row["factor"] == factor and row["method"] == "kalibr"}
    values = np.full((len(bins), 3), np.nan)
    counts = np.zeros((len(bins), 3), dtype=int)
    for row_index, radial in enumerate(bins):
        for col_index, condition in enumerate(("near", "mid_range", "far") if factor == "radius_x_distance" else ("low_tilt", "mid_tilt", "high_tilt")):
            key = f"{radial}__{condition}"
            if key in ours and key in kalibr:
                values[row_index, col_index] = ours[key] - kalibr[key]
                matching = next(row for row in rows if row["factor"] == factor and row["method"] == "ours" and row["bin"] == key)
                counts[row_index, col_index] = int(matching["point_count"])
    return values, counts


def draw_heatmap(ax: plt.Axes, values: np.ndarray, counts: np.ndarray, title: str, columns: list[str]) -> None:
    image = ax.imshow(values, cmap="RdYlGn_r", vmin=-0.20, vmax=0.20, aspect="auto")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(4), ["Center", "Mid", "Edge", "Extreme"])
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if not np.isfinite(values[row, col]):
                ax.text(col, row, "-", ha="center", va="center")
            else:
                text_color = "white" if abs(values[row, col]) > 0.13 else "black"
                ax.text(col, row, f"{values[row, col]:+.2f}\n(n={counts[row, col]})", ha="center", va="center", fontsize=8, color=text_color)
    return image


def main() -> None:
    args = parse_args()
    rows = read_rows(args.audit_dir / "geometry_bin_metrics.csv")
    radius = [row for row in rows if row["factor"] == "point_radius"]
    radial_bins = ["center", "mid", "edge", "extreme"]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 3.45),
        gridspec_kw={"width_ratios": [1.35, 1, 1]},
        layout="constrained",
    )
    x = np.arange(len(radial_bins))
    width = 0.20
    for index, method in enumerate(METHODS):
        values = [float(next(row["rmse_px"] for row in radius if row["method"] == method and row["bin"] == bin_name)) for bin_name in radial_bins]
        axes[0].bar(x + (index - 1.5) * width, values, width, label=LABELS[method], color=COLORS[method])
    axes[0].set_xticks(x, ["Center\n(n=3526)", "Mid\n(n=4050)", "Edge\n(n=2404)", "Extreme\n(n=56)"])
    axes[0].set_ylabel("Reprojection RMSE (px)")
    axes[0].set_title("Error by image radius")
    axes[0].set_ylim(0.0, 1.2)
    axes[0].legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    tilt_values, tilt_counts = matrix(rows, "radius_x_tilt", radial_bins)
    distance_values, distance_counts = matrix(rows, "radius_x_distance", radial_bins)
    heatmap = draw_heatmap(axes[1], tilt_values, tilt_counts, "Ours - Kalibr by tilt", ["Low", "Mid", "High"])
    draw_heatmap(axes[2], distance_values, distance_counts, "Ours - Kalibr by distance", ["Near", "Mid", "Far"])
    for ax in axes[1:]:
        ax.tick_params(length=0)
    colorbar = fig.colorbar(
        heatmap,
        ax=axes[1:],
        orientation="horizontal",
        fraction=0.08,
        pad=0.16,
    )
    colorbar.set_label("RMSE difference (px); negative (green): Ours is better")
    output = args.audit_dir / "geometry_stratification_summary.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    print(output)


if __name__ == "__main__":
    main()

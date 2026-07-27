#!/usr/bin/env python3
"""Plot scale-dependent DS-W1 robustness from the completed paired sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
METHODS = ("Outer-only", "Outer+Internal")
COLORS = {"Outer-only": "#D55E00", "Outer+Internal": "#0072B2"}
BOOTSTRAP_SEED = 20260720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.15g}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def bootstrap_median_ci(
    values: np.ndarray, rng: np.random.Generator, iterations: int = 10_000,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    medians = np.median(values[indices], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def summarize(pixels: pd.DataFrame, pairs: pd.DataFrame) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        pair = pairs[pairs.perturbation_level_deg == level]
        delta = pair.peripheral_pixel_p95_delta_px.to_numpy(float)
        positive = int(np.count_nonzero(delta > 0.0))
        win_ci = binomtest(positive, len(delta)).proportion_ci(
            confidence_level=0.95, method="exact",
        )
        delta_ci = bootstrap_median_ci(delta, rng)
        common = {
            "perturbation_level_deg": level,
            "paired_delta_median_px": float(np.median(delta)),
            "paired_delta_bootstrap_ci_low_px": delta_ci[0],
            "paired_delta_bootstrap_ci_high_px": delta_ci[1],
            "outer_internal_win_count": positive,
            "pair_count": len(delta),
            "outer_internal_win_rate": positive / len(delta),
            "win_rate_ci_low": float(win_ci.low),
            "win_rate_ci_high": float(win_ci.high),
            "bootstrap_iterations": 10_000,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }
        for method in METHODS:
            values = pixels[
                (pixels.perturbation_level_deg == level) & (pixels.method == method)
            ].peripheral_pixel_p95_px.to_numpy(float)
            rows.append({
                **common, "method": method,
                "peripheral_pixel_p95_median_px": float(np.median(values)),
                "peripheral_pixel_p95_q1_px": float(np.percentile(values, 25)),
                "peripheral_pixel_p95_q3_px": float(np.percentile(values, 75)),
                "peripheral_pixel_p95_tail_p95_px": float(np.percentile(values, 95)),
            })
    return rows


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "savefig.dpi": 400,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_axis(axis: plt.Axes, panel: str) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)
    axis.set_xticks(LEVELS)
    axis.text(
        0.012, 0.985, panel, transform=axis.transAxes,
        va="top", ha="left", fontfamily="DejaVu Serif",
        fontweight="bold", fontsize=8,
    )


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.05), constrained_layout=True)
    x_label = "Perturbation level s / target initial Peripheral Ray P95 (deg)"

    for method in METHODS:
        group = summary[summary.method == method].sort_values("perturbation_level_deg")
        x = group.perturbation_level_deg.to_numpy(float)
        axes[0, 0].fill_between(
            x, group.peripheral_pixel_p95_q1_px, group.peripheral_pixel_p95_q3_px,
            color=COLORS[method], alpha=0.15, linewidth=0,
        )
        axes[0, 0].plot(
            x, group.peripheral_pixel_p95_median_px, color=COLORS[method],
            marker="o", markersize=3.2, linewidth=1.5, label=method,
        )
        axes[0, 1].plot(
            x, group.peripheral_pixel_p95_tail_p95_px, color=COLORS[method],
            marker="o", markersize=3.2, linewidth=1.5, label=method,
        )

    paired = summary[summary.method == "Outer-only"].sort_values("perturbation_level_deg")
    x = paired.perturbation_level_deg.to_numpy(float)
    axes[1, 0].fill_between(
        x, paired.paired_delta_bootstrap_ci_low_px,
        paired.paired_delta_bootstrap_ci_high_px,
        color="#009E73", alpha=0.17, linewidth=0,
    )
    axes[1, 0].plot(
        x, paired.paired_delta_median_px, color="#009E73",
        marker="o", markersize=3.2, linewidth=1.5,
    )
    axes[1, 0].axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)

    axes[1, 1].fill_between(
        x, paired.win_rate_ci_low * 100.0, paired.win_rate_ci_high * 100.0,
        color="#CC79A7", alpha=0.17, linewidth=0,
    )
    axes[1, 1].plot(
        x, paired.outer_internal_win_rate * 100.0, color="#CC79A7",
        marker="o", markersize=3.2, linewidth=1.5,
    )
    axes[1, 1].axhline(50.0, color="#666666", linestyle="--", linewidth=0.8)

    labels = (
        "Peripheral Pixel P95 median (px)",
        "Peripheral Pixel P95 across-seed P95 (px)",
        "Paired median improvement (px)",
        "Outer+Internal win rate (%)",
    )
    for axis, ylabel, panel in zip(axes.flat, labels, ("(a)", "(b)", "(c)", "(d)")):
        axis.set_xlabel(x_label)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0.0)
        style_axis(axis, panel)
    axes[1, 1].set_ylim(0.0, 100.0)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=2,
        frameon=False, bbox_to_anchor=(0.5, 1.015),
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            output / f"scale_dependent_robustness_2x2.{suffix}",
            dpi=400, bbox_inches="tight", pad_inches=0.04,
        )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    output = (args.output or sweep_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pixels = pd.read_csv(sweep_dir / "local_recovery_pixel_runs.csv")
    pairs = pd.read_csv(sweep_dir / "paired_pixel_improvement.csv")
    if len(pixels) != 1200 or len(pairs) != 600:
        raise RuntimeError("Scale robustness plot requires 1200 runs and 600 paired rows")
    summary_path = output / "scale_dependent_robustness_summary.csv"
    write_csv(summary_path, summarize(pixels, pairs))
    # The figure reads only the materialized summary CSV.
    make_figure(pd.read_csv(summary_path), output)
    print(summary_path)
    print(output / "scale_dependent_robustness_2x2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

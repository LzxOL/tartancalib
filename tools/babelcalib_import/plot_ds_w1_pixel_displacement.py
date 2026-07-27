#!/usr/bin/env python3
"""Compute and plot exact DS ray-induced pixel displacement by W1 scale."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import run_ds_perturbation_sweep as sweep
import run_ds_weak_mode_perturbation as weak


METHODS = ("Outer-only", "Outer+Internal")
COLORS = {"Outer-only": "#D55E00", "Outer+Internal": "#0072B2"}
METRICS = (
    "full_pixel_p95_px",
    "peripheral_pixel_p95_px",
    "full_pixel_median_px",
    "peripheral_pixel_median_px",
)


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


def camera_from_row(row: pd.Series) -> sweep.Camera:
    return sweep.Camera(
        float(row.final_xi), float(row.final_alpha),
        float(row.final_fu), float(row.final_fv),
        float(row.final_cu), float(row.final_cv), family="ds-none",
    )


def pixel_metrics(
    camera: sweep.Camera, mask: sweep.EvaluationMask,
) -> dict[str, float | int]:
    pixels, candidate_valid = weak.project_ds(camera, mask.reference_rays)
    valid = mask.reference_valid & candidate_valid
    errors = np.linalg.norm(pixels - mask.pixels, axis=1)
    peripheral = valid & (mask.rho >= 0.7) & (mask.rho <= 1.0)
    full_values = errors[valid]
    peripheral_values = errors[peripheral]
    if not len(full_values) or not len(peripheral_values):
        raise RuntimeError("Candidate has no valid fixed-grid pixel displacement samples")
    return {
        "full_pixel_p95_px": float(np.percentile(full_values, 95)),
        "peripheral_pixel_p95_px": float(np.percentile(peripheral_values, 95)),
        "full_pixel_median_px": float(np.median(full_values)),
        "peripheral_pixel_median_px": float(np.median(peripheral_values)),
        "pixel_valid_grid_ratio": float(np.count_nonzero(valid) / np.count_nonzero(mask.reference_valid)),
        "pixel_valid_grid_count": int(np.count_nonzero(valid)),
    }


def compute_run_rows(
    runs: pd.DataFrame, mask: sweep.EvaluationMask,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        output: dict[str, Any] = {
            "perturbation_level_deg": float(run.perturbation_level_deg),
            "seed": int(run.seed), "method": str(run.method),
            "run_key": str(run.run_key), "solver_status": str(run.solver_status),
        }
        if run.solver_status != "converged":
            output.update({metric: math.nan for metric in METRICS})
            output.update({"pixel_valid_grid_ratio": 0.0, "pixel_valid_grid_count": 0})
        else:
            output.update(pixel_metrics(camera_from_row(run), mask))
        rows.append(output)
    return rows


def summarize(run_frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (level, method), group in run_frame.groupby(["perturbation_level_deg", "method"]):
        row: dict[str, Any] = {
            "perturbation_level_deg": float(level), "method": str(method),
            "run_count": len(group),
            "valid_run_count": int(np.count_nonzero(group.solver_status == "converged")),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_p05"] = float(np.percentile(values, 5))
            row[f"{metric}_p95"] = float(np.percentile(values, 95))
            row[f"{metric}_q1"] = float(np.percentile(values, 25))
            row[f"{metric}_q3"] = float(np.percentile(values, 75))
        rows.append(row)
    return sorted(rows, key=lambda row: (row["perturbation_level_deg"], row["method"]))


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "savefig.dpi": 400,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_axis(axis: plt.Axes, panel: str, levels: tuple[float, ...]) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)
    axis.text(
        0.012, 0.985, panel, transform=axis.transAxes,
        va="top", ha="left", fontfamily="DejaVu Serif",
        fontweight="bold", fontsize=8,
    )
    axis.set_xticks(levels)
    axis.set_ylim(bottom=0.0)


def plot_metric(axis: plt.Axes, summary: pd.DataFrame, metric: str) -> None:
    for method in METHODS:
        group = summary[summary.method == method].sort_values("perturbation_level_deg")
        x = group.perturbation_level_deg.to_numpy(float)
        mean = group[f"{metric}_mean"].to_numpy(float)
        median = group[f"{metric}_median"].to_numpy(float)
        p05 = group[f"{metric}_p05"].to_numpy(float)
        p95 = group[f"{metric}_p95"].to_numpy(float)
        axis.fill_between(x, p05, p95, color=COLORS[method], alpha=0.14, linewidth=0)
        axis.plot(
            x, mean, color=COLORS[method], linestyle="-", marker="o",
            markersize=3.0, linewidth=1.5,
        )
        axis.plot(
            x, median, color=COLORS[method], linestyle="--",
            linewidth=1.25,
        )


def make_figure(
    summary: pd.DataFrame, output: Path, levels: tuple[float, ...],
) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.05), constrained_layout=True)
    specs = (
        ("full_pixel_p95_px", "Full Pixel Displacement P95 (px)"),
        ("peripheral_pixel_p95_px", "Peripheral Pixel Displacement P95 (px)"),
        ("full_pixel_median_px", "Full Pixel Displacement Median (px)"),
        ("peripheral_pixel_median_px", "Peripheral Pixel Displacement Median (px)"),
    )
    for axis, (metric, ylabel), panel in zip(
        axes.flat, specs, ("(a)", "(b)", "(c)", "(d)"),
    ):
        plot_metric(axis, summary, metric)
        axis.set_xlabel("Perturbation level / target initial Peripheral Ray P95 (deg)")
        axis.set_ylabel(ylabel)
        style_axis(axis, panel, levels)
    legend = [
        Line2D([0], [0], color=COLORS["Outer-only"], marker="o", linewidth=1.5, label="Outer-only mean"),
        Line2D([0], [0], color=COLORS["Outer+Internal"], marker="o", linewidth=1.5, label="Outer+Internal mean"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.25, label="Median"),
        Patch(facecolor="#888888", alpha=0.14, edgecolor="none", label="P5-P95"),
    ]
    fig.legend(
        handles=legend, loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 1.015), handlelength=2.5,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            output / f"local_recovery_pixel_displacement_2x2.{suffix}",
            dpi=400, bbox_inches="tight", pad_inches=0.04,
        )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    output = (args.output or sweep_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((sweep_dir / "protocol_manifest.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(sweep_dir / "local_recovery_runs.csv")
    levels = tuple(float(value) for value in protocol["levels_deg"])
    seeds = tuple(int(value) for value in protocol["seeds"])
    expected_keys = {
        (level, seed, method)
        for level in levels for seed in seeds for method in METHODS
    }
    actual_keys = set(zip(
        runs.perturbation_level_deg.astype(float), runs.seed.astype(int), runs.method,
        strict=True,
    ))
    duplicate_keys = runs.duplicated(
        ["perturbation_level_deg", "seed", "method"], keep=False,
    )
    if duplicate_keys.any():
        raise RuntimeError("Pixel plot source contains duplicate level/seed/method keys")
    if actual_keys != expected_keys or runs.run_key.nunique() != len(expected_keys):
        raise RuntimeError(
            "Pixel plot requires the complete protocol grid: "
            f"expected={len(expected_keys)}, actual={len(actual_keys)}, "
            f"missing={len(expected_keys - actual_keys)}, extra={len(actual_keys - expected_keys)}"
        )
    scene = weak.parse_scene(Path(protocol["reference_scene"]))
    mask = sweep.build_evaluation_mask(
        scene.camera, int(protocol["image_width"]), int(protocol["image_height"]),
        int(protocol["grid_size"]),
    )
    run_path = output / "local_recovery_pixel_runs.csv"
    summary_path = output / "local_recovery_pixel_level_summary.csv"
    write_csv(run_path, compute_run_rows(runs, mask))
    run_frame = pd.read_csv(run_path)
    summary_rows = summarize(run_frame)
    write_csv(summary_path, summary_rows)
    # The publication figure reads only the materialized summary CSV.
    make_figure(pd.read_csv(summary_path), output, levels)
    caption = (
        f"Mean (solid), median (dashed), and P5-P95 range over {len(seeds)} paired noise "
        "seeds. Pixel displacement is computed by projecting each fixed GT ray "
        "through the optimized DS model; it is not reprojection RMSE. Peripheral "
        "samples use rho>=0.7 on the fixed GT-centered mask.\n"
    )
    (output / "local_recovery_pixel_displacement_caption.txt").write_text(
        caption, encoding="utf-8",
    )
    print(summary_path)
    print(output / "local_recovery_pixel_displacement_2x2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

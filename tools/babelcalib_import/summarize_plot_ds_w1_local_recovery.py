#!/usr/bin/env python3
"""Validate, summarize, and plot the paper DS-W1 recovery experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

import run_ds_perturbation_sweep as sweep
import run_ds_weak_mode_perturbation as weak


LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
METHODS = ("Outer-only", "Outer+Internal")
COLORS = {"Outer-only": "#D55E00", "Outer+Internal": "#0072B2"}
BOOTSTRAP_SEED = 20260720
TIE_TOLERANCE_DEG = 1e-12
REQUIRED_RUN_COLUMNS = {
    "perturbation_level_deg", "seed", "method", "full_ray_p95_deg",
    "peripheral_ray_p95_deg", "xi_abs_error", "alpha_abs_error",
    "fu_relative_error", "fv_relative_error", "mean_focal_relative_error",
    "principal_point_error_px", "training_rmse_px", "valid_grid_ratio",
    "solver_status", "failure_reason", "final_xi", "final_alpha",
    "final_fu", "final_fv", "final_cu", "final_cv", "run_key",
    "noise_fingerprint", "w1_fingerprint", "scene_fingerprint",
    "layout_fingerprint", "initial_camera_fingerprint",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fisher-base", type=Path,
        default=repo / "result_may/ds_weak_mode_recovery_20260720_right_w1",
    )
    parser.add_argument(
        "--fisher-replication-pattern",
        default=str(repo / "result_may/ds_weak_mode_recovery_20260720_right_replication_pair{index:02d}"),
    )
    parser.add_argument("--fisher-replication-count", type=int, default=8)
    parser.add_argument(
        "--large-perturbation-root", type=Path,
        default=repo / "result_may/ds_controlled_gt_recovery_20260720_right_w1_exact_seedfixed",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.15g}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def finite_values(frame: pd.DataFrame, metric: str) -> np.ndarray:
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def quartiles(values: np.ndarray) -> tuple[float, float, float]:
    if not len(values):
        return math.nan, math.nan, math.nan
    return tuple(float(value) for value in np.percentile(values, [50, 25, 75]))


def validate_source_hashes(manifest: dict[str, Any]) -> None:
    for path_key, hash_key in (
        ("reference_scene", "reference_scene_sha256"),
        ("training_points", "training_points_sha256"),
    ):
        path = Path(manifest[path_key])
        if not path.is_file() or sha256_file(path) != manifest[hash_key]:
            raise RuntimeError(f"Experiment source changed or disappeared: {path}")


def validate_sweep(
    sweep_dir: Path, allow_incomplete: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((sweep_dir / "protocol_manifest.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(sweep_dir / "local_recovery_runs.csv")
    poses = pd.read_csv(sweep_dir / "pose_only_run_summary.csv")
    missing = REQUIRED_RUN_COLUMNS - set(runs.columns)
    if missing:
        raise RuntimeError(f"local_recovery_runs.csv missing columns: {sorted(missing)}")
    if runs["run_key"].duplicated().any():
        raise RuntimeError("Duplicate run_key in local_recovery_runs.csv")
    levels = tuple(sorted(float(value) for value in runs["perturbation_level_deg"].unique()))
    expected_levels = tuple(sorted(float(value) for value in manifest["levels_deg"]))
    if levels != expected_levels:
        raise RuntimeError(f"CSV/manifest level mismatch: {levels} != {expected_levels}")
    if not allow_incomplete and levels != LEVELS:
        raise RuntimeError(f"Paper sweep requires levels {LEVELS}, got {levels}")
    seeds = sorted(int(value) for value in runs["seed"].unique())
    expected_count = len(levels) * len(seeds) * len(METHODS)
    if len(runs) != expected_count:
        raise RuntimeError(f"Expected {expected_count} complete run rows, got {len(runs)}")
    if not allow_incomplete and len(seeds) != 100:
        raise RuntimeError(f"Paper sweep requires 100 seeds, got {len(seeds)}")
    for level in levels:
        for seed in seeds:
            pair = runs[(runs.perturbation_level_deg == level) & (runs.seed == seed)]
            if set(pair.method) != set(METHODS) or len(pair) != 2:
                raise RuntimeError(f"Incomplete pair at level={level}, seed={seed}")
    for field in ("w1_fingerprint", "scene_fingerprint", "layout_fingerprint"):
        if runs[field].nunique(dropna=False) != 1:
            raise RuntimeError(f"Non-frozen {field}")
    for seed, group in runs.groupby("seed"):
        if group["noise_fingerprint"].nunique(dropna=False) != 1:
            raise RuntimeError(f"Training noise changed across level/method for seed {seed}")
    for level, group in runs.groupby("perturbation_level_deg"):
        if group["initial_camera_fingerprint"].nunique(dropna=False) != 1:
            raise RuntimeError(f"Initial camera changed within level {level}")
    if len(poses) != len(runs):
        raise RuntimeError("pose_only_run_summary.csv must have one row per intrinsic run")
    pose_keys = poses.assign(
        key=poses.apply(
            lambda row: f"{float(row.perturbation_level_deg):.12g}|{int(row.seed)}|{row.method}",
            axis=1,
        )
    )
    if pose_keys.key.duplicated().any():
        raise RuntimeError("Duplicate pose-only run summary pair")
    for seed, group in poses.groupby("seed"):
        if group["eval_noise_fingerprint"].nunique(dropna=False) != 1:
            raise RuntimeError(f"Evaluation noise changed across level/method for seed {seed}")
    validate_source_hashes(manifest)
    external = json.loads(
        (sweep_dir / "external_pose_template_manifest.json").read_text(encoding="utf-8")
    )
    if not external.get("dataset_is_independent_from_training", False):
        raise RuntimeError("External pose evaluation source is not independent from training")
    return runs, poses, manifest


def level_summaries(runs: pd.DataFrame, poses: pd.DataFrame) -> list[dict[str, Any]]:
    pose_columns = [
        "perturbation_level_deg", "seed", "method", "orientation_error_p95_deg",
        "pose_reprojection_rmse_px", "pose_success_rate",
    ]
    merged = runs.merge(poses[pose_columns], on=["perturbation_level_deg", "seed", "method"])
    metrics = (
        "full_ray_p95_deg", "peripheral_ray_p95_deg", "xi_abs_error",
        "alpha_abs_error", "mean_focal_relative_error",
        "principal_point_error_px", "training_rmse_px", "valid_grid_ratio",
        "orientation_error_p95_deg", "pose_reprojection_rmse_px",
    )
    rows: list[dict[str, Any]] = []
    for (level, method), group in merged.groupby(["perturbation_level_deg", "method"]):
        row: dict[str, Any] = {
            "perturbation_level_deg": float(level), "method": method,
            "total_run_count": len(group),
            "valid_run_count": int(np.count_nonzero(
                (group.solver_status == "converged") & np.isfinite(group.peripheral_ray_p95_deg)
            )),
            "solver_failure_count": int(np.count_nonzero(group.solver_status != "converged")),
            "invalid_grid_run_count": int(np.count_nonzero(group.valid_grid_ratio < 1.0 - 1e-15)),
            "pose_failure_run_count": int(np.count_nonzero(group.pose_success_rate < 1.0)),
        }
        for metric in metrics:
            median, q1, q3 = quartiles(finite_values(group, metric))
            row[f"{metric}_median"] = median
            row[f"{metric}_q1"] = q1
            row[f"{metric}_q3"] = q3
        rows.append(row)
    return sorted(rows, key=lambda row: (row["perturbation_level_deg"], row["method"]))


def hodges_lehmann(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return math.nan
    i, j = np.triu_indices(len(values))
    return float(np.median(0.5 * (values[i] + values[j])))


def bootstrap_ci(
    values: np.ndarray, statistic: Callable[[np.ndarray], float], rng: np.random.Generator,
    iterations: int = 10_000,
) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        estimates[index] = statistic(values[rng.integers(0, len(values), len(values))])
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def paired_statistics(runs: pd.DataFrame) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for level in sorted(runs.perturbation_level_deg.unique()):
        group = runs[runs.perturbation_level_deg == level]
        pivot = group.pivot(index="seed", columns="method", values="peripheral_ray_p95_deg")
        status = group.pivot(index="seed", columns="method", values="solver_status")
        valid = (
            np.isfinite(pivot[METHODS[0]]) & np.isfinite(pivot[METHODS[1]]) &
            (status[METHODS[0]] == "converged") & (status[METHODS[1]] == "converged")
        )
        delta = (
            pivot.loc[valid, METHODS[0]] - pivot.loc[valid, METHODS[1]]
        ).to_numpy(dtype=float)
        median = float(np.median(delta)) if len(delta) else math.nan
        hl = hodges_lehmann(delta)
        median_ci = bootstrap_ci(delta, lambda values: float(np.median(values)), rng)
        hl_ci = bootstrap_ci(delta, hodges_lehmann, rng)
        if len(delta) and np.any(np.abs(delta) > TIE_TOLERANCE_DEG):
            wilcoxon_p = float(wilcoxon(
                delta, alternative="greater", zero_method="wilcox",
            ).pvalue)
        else:
            wilcoxon_p = 1.0
        rows.append({
            "perturbation_level_deg": float(level),
            "complete_pair_count": len(delta),
            "incomplete_pair_count": len(pivot) - len(delta),
            "paired_median_difference_deg": median,
            "paired_median_bootstrap_ci_low_deg": median_ci[0],
            "paired_median_bootstrap_ci_high_deg": median_ci[1],
            "hodges_lehmann_estimate_deg": hl,
            "hodges_lehmann_bootstrap_ci_low_deg": hl_ci[0],
            "hodges_lehmann_bootstrap_ci_high_deg": hl_ci[1],
            "wilcoxon_one_sided_p": wilcoxon_p,
            "outer_internal_better_count": int(np.count_nonzero(delta > TIE_TOLERANCE_DEG)),
            "outer_only_better_count": int(np.count_nonzero(delta < -TIE_TOLERANCE_DEG)),
            "tie_count": int(np.count_nonzero(np.abs(delta) <= TIE_TOLERANCE_DEG)),
            "tie_tolerance_deg": TIE_TOLERANCE_DEG,
            "bootstrap_iterations": 10_000,
            "bootstrap_seed": BOOTSTRAP_SEED,
        })
    return rows


def directional_fisher_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roots = [("base", args.fisher_base.resolve())]
    roots.extend((
        f"replication_{index:02d}",
        Path(args.fisher_replication_pattern.format(index=index)).resolve(),
    ) for index in range(1, args.fisher_replication_count + 1))
    rows: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    for label, root in roots:
        outer_path = root / "outer_only_schur_fisher.csv"
        internal_path = root / "outer_internal_schur_fisher.csv"
        mode_rows = pd.read_csv(root / "outer_only_weak_modes.csv")
        direction = mode_rows.loc[mode_rows.mode_index == 0, [
            "xi", "alpha", "log_fu", "log_fv", "cu_over_width", "cv_over_height",
        ]].to_numpy(dtype=float)
        if direction.shape != (1, 6):
            raise RuntimeError(f"Malformed W1 direction in {root}")
        direction = direction[0] / np.linalg.norm(direction[0])
        outer = np.loadtxt(outer_path, delimiter=",")
        internal = np.loadtxt(internal_path, delimiter=",")
        audit = json.loads((root / "outer_only_weak_mode_audit.json").read_text())
        outer_info = directional_information(outer, direction)
        internal_info = directional_information(internal, direction)
        outer_count = float(audit["outer_point_count"])
        internal_count = float(audit["outer_internal_point_count"])
        source_hash = sha256_file(outer_path) + "+" + sha256_file(internal_path)
        if source_hash in source_hashes:
            raise RuntimeError(f"Repeated Fisher matrix pair: {root}")
        source_hashes.add(source_hash)
        rows.append({
            "schedule": label, "source_root": str(root),
            "outer_w1_information_raw": outer_info,
            "outer_internal_w1_information_raw": internal_info,
            "raw_gain": internal_info / outer_info,
            "outer_effective_point_weight_sum": outer_count,
            "outer_internal_effective_point_weight_sum": internal_count,
            "outer_w1_information_normalized": outer_info / outer_count,
            "outer_internal_w1_information_normalized": internal_info / internal_count,
            "normalized_gain": (internal_info / internal_count) / (outer_info / outer_count),
            "w1_direction_fingerprint": hashlib.sha256(direction.tobytes()).hexdigest(),
            "matrix_pair_fingerprint": source_hash,
        })
    raw_gains = np.asarray([row["raw_gain"] for row in rows])
    normalized_gains = np.asarray([row["normalized_gain"] for row in rows])
    summary = [{
        "schedule_count": len(rows),
        "raw_gain_min": float(np.min(raw_gains)),
        "raw_gain_median": float(np.median(raw_gains)),
        "raw_gain_max": float(np.max(raw_gains)),
        "normalized_gain_min": float(np.min(normalized_gains)),
        "normalized_gain_median": float(np.median(normalized_gains)),
        "normalized_gain_max": float(np.max(normalized_gains)),
        "raw_improvement_count": int(np.count_nonzero(raw_gains > 1.0)),
        "normalized_improvement_count": int(np.count_nonzero(normalized_gains > 1.0)),
        "raw_gain_sign_test_p": float(binomtest(
            np.count_nonzero(raw_gains > 1.0), len(raw_gains), 0.5,
            alternative="greater",
        ).pvalue),
        "normalized_gain_sign_test_p": float(binomtest(
            np.count_nonzero(normalized_gains > 1.0), len(normalized_gains), 0.5,
            alternative="greater",
        ).pvalue),
    }]
    return rows, summary


def directional_information(fisher: np.ndarray, direction: np.ndarray) -> float:
    fisher = np.asarray(fisher, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if fisher.shape != (len(direction), len(direction)):
        raise ValueError("Fisher matrix and direction dimensions do not match")
    return float(direction @ fisher @ direction)


def camera_from_row(row: pd.Series, prefix: str) -> sweep.Camera:
    return sweep.Camera(
        float(row[f"{prefix}_xi"]), float(row[f"{prefix}_alpha"]),
        float(row[f"{prefix}_fu"]), float(row[f"{prefix}_fv"]),
        float(row[f"{prefix}_cu"]), float(row[f"{prefix}_cv"]), family="ds-none",
    )


def radial_profile(
    mask: sweep.EvaluationMask, reference: sweep.Camera, candidate: sweep.Camera,
    label: str, bins: int = 20,
) -> list[dict[str, Any]]:
    candidate_rays, candidate_valid = sweep.unproject_ds(candidate, mask.pixels)
    valid = mask.reference_valid & candidate_valid
    dot = np.sum(mask.reference_rays * candidate_rays, axis=1)
    angles = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    rows = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = valid & (mask.rho >= low) & (
            (mask.rho <= high) if index == bins - 1 else (mask.rho < high)
        )
        values = angles[selected]
        rows.append({
            "model": label, "rho_bin_index": index,
            "rho_low": low, "rho_high": high, "rho_center": 0.5 * (low + high),
            "ray_p95_deg": float(np.percentile(values, 95)) if len(values) else math.nan,
            "valid_sample_count": int(len(values)),
        })
    return rows


def fixed_backend_data(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = pd.read_csv(root / "weak_mode_perturbation_results.csv")
    selected = source[
        (source.direction == "W1_plus") & source["scale"].isin([0.5, 2.0, 4.0])
    ].copy()
    if len(selected) != 6:
        raise RuntimeError("Fixed Backend source must contain paired 0.5/2/4 degree rows")
    compact = []
    for _, row in selected.iterrows():
        compact.append({
            "initial_peripheral_ray_p95_deg": float(row["initial_common_peripheral_ray_p95_deg"]),
            "method": row["method"],
            "final_full_ray_p95_deg": float(row["final_common_full_ray_p95_deg"]),
            "final_peripheral_ray_p95_deg": float(row["final_common_peripheral_ray_p95_deg"]),
            "valid_grid_ratio": float(row["final_common_valid_grid_ratio"]),
            "solver_status": int(row["solver_status"]),
            "source_run_dir": row["run_dir"],
        })
    reference_scene = weak.parse_scene(root / "reference/clean_reference_scene.txt")
    mask = sweep.build_evaluation_mask(reference_scene.camera, 4512, 4512, 121)
    row4 = selected[np.isclose(selected["scale"], 4.0)]
    scene_path = root / "W1_plus/rayp95_04000mdeg/perturbed_reference_scene.txt"
    perturbed = weak.parse_scene(scene_path).camera
    profiles = radial_profile(mask, reference_scene.camera, reference_scene.camera, "GT")
    profiles += radial_profile(mask, reference_scene.camera, perturbed, "Perturbed initialization")
    for _, row in row4.iterrows():
        final = camera_from_row(row, "final")
        profiles += radial_profile(mask, reference_scene.camera, final, str(row["method"]))
    return compact, profiles


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "lines.linewidth": 1.5,
        "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_axis(axis: plt.Axes, panel: str) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)
    axis.text(
        0.012, 0.985, panel, transform=axis.transAxes,
        va="top", ha="left", fontfamily="DejaVu Serif",
        fontweight="bold", fontsize=8,
    )


def plot_summary_metric(
    axis: plt.Axes, summary: pd.DataFrame, metric: str, ylabel: str,
    *, percent: bool = False, valid_percent: bool = False,
) -> None:
    factor = 100.0 if percent or valid_percent else 1.0
    for method in METHODS:
        group = summary[summary.method == method].sort_values("perturbation_level_deg")
        x = group.perturbation_level_deg.to_numpy(float)
        median = group[f"{metric}_median"].to_numpy(float) * factor
        q1 = group[f"{metric}_q1"].to_numpy(float) * factor
        q3 = group[f"{metric}_q3"].to_numpy(float) * factor
        axis.fill_between(x, q1, q3, color=COLORS[method], alpha=0.15, linewidth=0)
        axis.plot(x, median, marker="o", markersize=3.2, color=COLORS[method], label=method)
    axis.set_xlabel("Initial Peripheral Ray Error, P95 (deg)")
    axis.set_ylabel(ylabel)
    axis.set_xticks(LEVELS)
    if valid_percent:
        axis.set_ylim(0.0, 100.0)
    else:
        axis.set_ylim(bottom=0.0)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_main(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.15), constrained_layout=True)
    specs = (
        ("full_ray_p95_deg", "Final Ray Error, P95 (deg)", False),
        ("peripheral_ray_p95_deg", "Final Peripheral Ray Error, P95 (deg)", False),
        ("xi_abs_error", r"$|\xi-\xi_{gt}|$", False),
        ("mean_focal_relative_error", "Mean Relative Focal Error (%)", True),
    )
    for axis, (metric, ylabel, percent), panel in zip(axes.flat, specs, ("(a)", "(b)", "(c)", "(d)")):
        plot_summary_metric(axis, summary, metric, ylabel, percent=percent)
        style_axis(axis, panel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, output, "local_recovery_main_2x2")


def plot_geometric(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.15), constrained_layout=True)
    specs = (
        ("principal_point_error_px", "Principal-Point Error (px)", False),
        ("orientation_error_p95_deg", "External Orientation Error, P95 (deg)", False),
        ("training_rmse_px", "Common-Outer Training RMSE (px)", False),
        ("valid_grid_ratio", "Valid Projection Ratio (%)", False),
    )
    for axis, (metric, ylabel, _), panel in zip(axes.flat, specs, ("(a)", "(b)", "(c)", "(d)")):
        plot_summary_metric(
            axis, summary, metric, ylabel, valid_percent=(metric == "valid_grid_ratio"),
        )
        style_axis(axis, panel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, output, "local_recovery_geometric_2x2")


def plot_fisher(rows: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
    x = np.arange(len(rows))
    for index in range(len(rows)):
        axes[0].plot(
            [x[index], x[index]],
            [rows.iloc[index].outer_w1_information_raw, rows.iloc[index].outer_internal_w1_information_raw],
            color="#B5B5B5", linewidth=0.8, zorder=1,
        )
    axes[0].scatter(x, rows.outer_w1_information_raw, color=COLORS[METHODS[0]], s=18, label=METHODS[0], zorder=2)
    axes[0].scatter(x, rows.outer_internal_w1_information_raw, color=COLORS[METHODS[1]], s=18, label=METHODS[1], zorder=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Raw $I_{W1}=d_{W1}^{\mathsf{T}}Fd_{W1}$")
    axes[0].set_xlabel("Independent schedule")
    axes[0].set_xticks(x, [str(index + 1) for index in x])
    style_axis(axes[0], "(a)")
    axes[1].axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    axes[1].plot(x, rows.raw_gain, marker="o", markersize=3, color="#009E73", label="Raw gain")
    axes[1].plot(x, rows.normalized_gain, marker="s", markersize=3, color="#CC79A7", label="Weight-normalized gain")
    axes[1].set_ylabel("Outer+Internal / Outer-only")
    axes[1].set_xlabel("Independent schedule")
    axes[1].set_xticks(x, [str(index + 1) for index in x])
    axes[1].set_ylim(bottom=0.0)
    style_axis(axes[1], "(b)")
    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    fig.legend(handles0 + handles1, labels0 + labels1, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    save_figure(fig, output, "weak_direction_information")


def plot_large(fixed: pd.DataFrame, radial: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
    for method in METHODS:
        group = fixed[fixed.method == method].sort_values("initial_peripheral_ray_p95_deg")
        axes[0].plot(
            group.initial_peripheral_ray_p95_deg, group.final_peripheral_ray_p95_deg,
            marker="o", markersize=3.2, color=COLORS[method], label=method,
        )
    axes[0].set_xlabel("Initial Peripheral Ray Error, P95 (deg)")
    axes[0].set_ylabel("Final Peripheral Ray Error, P95 (deg)")
    axes[0].set_xticks([0.5, 2.0, 4.0])
    axes[0].set_ylim(bottom=0.0)
    style_axis(axes[0], "(a)")
    radial_colors = {
        "GT": "#000000", "Perturbed initialization": "#777777",
        "Outer-only": COLORS["Outer-only"], "Outer+Internal": COLORS["Outer+Internal"],
    }
    radial_styles = {"GT": ":", "Perturbed initialization": "--", "Outer-only": "-", "Outer+Internal": "-"}
    for model, group in radial.groupby("model", sort=False):
        axes[1].plot(
            group.rho_center, group.ray_p95_deg, color=radial_colors[model],
            linestyle=radial_styles[model], label=model,
        )
    axes[1].set_xlabel(r"Normalized radius $\rho$")
    axes[1].set_ylabel("Ray Deviation, P95 (deg)")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(bottom=0.0)
    style_axis(axes[1], "(b)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.015, "Fixed-observation Backend experiment", ha="center", fontsize=7.5, style="italic")
    save_figure(fig, output, "fixed_backend_large_perturbation")


def write_captions(
    output: Path, manifest: dict[str, Any], level_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]], fisher_summary: list[dict[str, Any]],
    external_manifest: dict[str, Any],
) -> None:
    failures = sum(int(row["solver_failure_count"]) for row in level_rows)
    pose_failures = sum(int(row["pose_failure_run_count"]) for row in level_rows)
    seed_count = len(manifest["seeds"])
    fisher = fisher_summary[0]
    paired_medians = [float(row["paired_median_difference_deg"]) for row in paired_rows]
    paired_p = [float(row["wilcoxon_one_sided_p"]) for row in paired_rows]
    better_counts = [int(row["outer_internal_better_count"]) for row in paired_rows]
    complete_counts = [int(row["complete_pair_count"]) for row in paired_rows]
    paired_sentence = (
        f"The paired median Outer-only minus Outer+Internal peripheral error ranges from "
        f"{min(paired_medians):.6f} to {max(paired_medians):.6f} deg; Outer+Internal is "
        f"better in {min(better_counts)}-{max(better_counts)} of "
        f"{min(complete_counts)} complete pairs per level, with one-sided Wilcoxon "
        f"$p$ in [{min(paired_p):.3g}, {max(paired_p):.3g}]."
    )
    text = f"""# Figure Captions

**Local recovery.** Median and interquartile range over {seed_count} fixed noise seeds at each measured perturbation level. Gaussian image noise has $\\sigma={manifest['noise_sigma_px']:.2f}$ px. Peripheral rays use $\\rho\\geq0.7$ on the fixed GT-centered valid disc. {paired_sentence} The sweep retained {failures} intrinsic solver failures and {pose_failures} runs with at least one external pose failure.

**Geometric evaluation.** Intrinsics are fixed after training; each of {external_manifest['successful_frame_count']} independent external evaluation frames optimizes only $T_{{camera,reference}}$ from the frozen GT pose using common outer observations. Curves show medians and IQR over the same {seed_count} seeds.

**Weak-direction information.** Directional information is evaluated as $d_{{W1}}^T F d_{{W1}}$ along each schedule's Outer-only W1 for both residual sets. Raw gains improve in {fisher['raw_improvement_count']}/{fisher['schedule_count']} schedules (one-sided sign test $p={fisher['raw_gain_sign_test_p']:.4g}$); effective-weight-normalized gains improve in {fisher['normalized_improvement_count']}/{fisher['schedule_count']} schedules.

**Large perturbation.** Fixed-observation Backend experiment at initial peripheral Ray P95 levels 0.5, 2.0, and 4.0 deg. The radial panel reports unsmoothed per-bin P95 deviation in 20 fixed $\\rho$ bins at 4 deg; this is not a complete Stage5 selection pipeline.
"""
    (output / "figure_captions.md").write_text(text, encoding="utf-8")
    includes = rf"""\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{local_recovery_main_2x2.pdf}}
  \caption{{Median and IQR over {seed_count} paired noise seeds ($\sigma={manifest['noise_sigma_px']:.2f}$ px). Peripheral rays use $\rho\geq0.7$ on the fixed GT mask. The paired median improvement is {min(paired_medians):.6f}--{max(paired_medians):.6f} deg across measured levels; {failures} solver failures are retained.}}
  \label{{fig:ds_w1_local_recovery}}
\end{{figure*}}

\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{local_recovery_geometric_2x2.pdf}}
  \caption{{Independent pose-only evaluation on {external_manifest['successful_frame_count']} external views. Final intrinsics and GT board layout are fixed; only $T_{{camera,reference}}$ is optimized from the frozen GT pose using common outer observations.}}
  \label{{fig:ds_w1_geometric}}
\end{{figure*}}

\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{weak_direction_information.pdf}}
  \caption{{Directional Fisher information along each schedule's Outer-only W1. Raw information improves in {fisher['raw_improvement_count']}/{fisher['schedule_count']} schedules (one-sided sign test $p={fisher['raw_gain_sign_test_p']:.4g}$); effective-weight-normalized information improves in {fisher['normalized_improvement_count']}/{fisher['schedule_count']}.}}
  \label{{fig:ds_w1_information}}
\end{{figure*}}

\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{fixed_backend_large_perturbation.pdf}}
  \caption{{Fixed-observation Backend experiment at 0.5, 2.0, and 4.0 deg initial peripheral Ray P95. The 4-deg profile contains 20 fixed radial bins without smoothing; this is not a complete Stage5 selection pipeline.}}
  \label{{fig:ds_w1_large_perturbation}}
\end{{figure*}}
"""
    (output / "latex_includes.tex").write_text(includes, encoding="utf-8")


def main() -> int:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    output = (args.output or sweep_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs, poses, manifest = validate_sweep(sweep_dir, args.allow_incomplete)
    external_manifest = json.loads(
        (sweep_dir / "external_pose_template_manifest.json").read_text(encoding="utf-8")
    )
    level_rows = level_summaries(runs, poses)
    paired_rows = paired_statistics(runs)
    fisher_rows, fisher_summary = directional_fisher_rows(args)
    fixed_rows, radial_rows = fixed_backend_data(args.large_perturbation_root.resolve())
    write_csv(output / "local_recovery_level_summary.csv", level_rows)
    write_csv(output / "local_recovery_paired_statistics.csv", paired_rows)
    write_csv(output / "fisher_information_by_schedule.csv", fisher_rows)
    write_csv(output / "fisher_information_summary.csv", fisher_summary)
    write_csv(output / "fixed_backend_large_perturbation.csv", fixed_rows)
    write_csv(output / "fixed_backend_radial_profile.csv", radial_rows)
    configure_plot_style()
    level_frame = pd.DataFrame(level_rows)
    plot_main(level_frame, output)
    plot_geometric(level_frame, output)
    plot_fisher(pd.DataFrame(fisher_rows), output)
    plot_large(pd.DataFrame(fixed_rows), pd.DataFrame(radial_rows), output)
    write_captions(
        output, manifest, level_rows, paired_rows, fisher_summary, external_manifest,
    )
    generated = [
        "local_recovery_main_2x2", "local_recovery_geometric_2x2",
        "weak_direction_information", "fixed_backend_large_perturbation",
    ]
    artifacts = []
    for stem in generated:
        for suffix in ("pdf", "png"):
            path = output / f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing plot artifact: {path}")
            artifacts.append({"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    audit = {
        "schema": "ds_w1_paper_statistics_v1",
        "source_sweep": str(sweep_dir),
        "source_runs_sha256": sha256_file(sweep_dir / "local_recovery_runs.csv"),
        "run_count": len(runs), "seed_count": int(runs.seed.nunique()),
        "level_count": int(runs.perturbation_level_deg.nunique()),
        "artifacts": artifacts,
    }
    (output / "paper_statistics_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

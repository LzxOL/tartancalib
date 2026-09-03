#!/usr/bin/env python3
"""Aggregate and plot the paired Right-camera P1 deep experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_ds_perturbation_sweep import Camera, build_evaluation_mask, ray_metrics


COLORS = {"Outer-only": "#D55E00", "Outer+Internal": "#0072B2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {
                key: (f"{value:.12f}" if isinstance(value, (float, np.floating)) else value)
                for key, value in row.items()
            }
            for row in rows
        ])


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def ds_camera_fingerprint(values: tuple[float, ...]) -> str:
    payload = "|".join(("ds-none", *(f"{value:.15g}" for value in values)))
    return "sha256:" + hashlib.sha256(payload.encode("ascii")).hexdigest()


def bootstrap_median_ci(values: np.ndarray, seed: int = 1337) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, values.size), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_strength(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharex=True)
    metrics = [
        ("final_common_full_ray_p95_deg", "Full-view Ray P95 (deg)"),
        ("final_common_peripheral_ray_p95_deg", "Peripheral Ray P95 (deg)"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics):
        for method in ("Outer-only", "Outer+Internal"):
            selected = sorted((r for r in rows if r["method"] == method), key=lambda r: float(r["scale"]))
            ax.plot([float(r["scale"]) for r in selected], [float(r[key]) for r in selected],
                    marker="o", ms=4, lw=1.8, color=COLORS[method], label=method)
        ax.set_xlabel("Perturbation scale $s$")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", lw=0.6)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(output / "fig1_p1_strength_sweep.png")
    plt.close(fig)


def plot_neighborhood(rows: list[dict[str, str]], output: Path) -> None:
    ordered = sorted(rows, key=lambda r: (r["direction"], float(r["scale"])))
    labels = [f"{r['direction']}-{float(r['scale']):.1f}" for r in ordered]
    values = [float(r["delta_peripheral_ray_p95_deg"]) for r in ordered]
    colors = ["#009E73" if value >= 0 else "#D55E00" for value in values]
    fig, ax = plt.subplots(figsize=(7.1, 2.8))
    ax.bar(np.arange(len(values)), values, width=0.72, color=colors)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    ax.set_ylabel(r"Paired improvement $\Delta E_{peri}$ (deg)")
    ax.grid(axis="y", color="#D9D9D9", lw=0.6)
    fig.tight_layout()
    fig.savefig(output / "fig2_p1_neighborhood_improvement.png")
    plt.close(fig)


def plot_trajectory(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), sharex="col")
    for column, scale in enumerate((0.8, 1.0)):
        for method in ("Outer-only", "Outer+Internal"):
            selected = [r for r in rows if math.isclose(float(r["scale"]), scale) and r["method"] == method]
            selected.sort(key=lambda r: int(r["increment_index"]))
            x = [int(r["selected_frame_count"]) for r in selected]
            axes[0, column].plot(x, [float(r["peripheral_ray_p95_deg"]) for r in selected],
                                 color=COLORS[method], lw=1.5, label=method)
            axes[1, column].plot(x, [float(r["valid_grid_ratio"]) for r in selected],
                                 color=COLORS[method], lw=1.5, label=method)
        axes[0, column].set_title(f"P1, $s={scale:.1f}$")
        axes[1, column].set_xlabel("Selected frame count")
        for row in range(2):
            axes[row, column].grid(axis="y", color="#D9D9D9", lw=0.6)
    axes[0, 0].set_ylabel("Peripheral Ray P95 (deg)")
    axes[1, 0].set_ylabel("Valid grid ratio")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "fig3_p1_incremental_trajectory.png")
    plt.close(fig)


def plot_density(rows: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharex=True)
    for ax, scale in zip(axes, (0.8, 1.0)):
        selected = sorted((r for r in rows if float(r["scale"]) == scale and r["weighting_mode"] == "current_board_type_balanced"), key=lambda r: float(r["internal_ratio"]))
        ax.plot([100 * float(r["internal_ratio"]) for r in selected],
                [float(r["final_peripheral_ray_p95"]) for r in selected],
                marker="o", ms=4, lw=1.8, color=COLORS["Outer+Internal"])
        ax.set_title(f"P1, $s={scale:.1f}$")
        ax.set_xlabel("Retained internal observations (%)")
        ax.set_xticks([0, 25, 50, 100])
        ax.grid(axis="y", color="#D9D9D9", lw=0.6)
    axes[0].set_ylabel("Peripheral Ray P95 (deg)")
    fig.tight_layout()
    fig.savefig(output / "fig4_p1_internal_density.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    strength = read_csv(root / "strength/perturbation_results.csv")
    neighborhood = read_csv(root / "neighborhood/perturbation_results.csv")
    neighborhood_pairs = read_csv(root / "neighborhood/paired_improvements.csv")
    shutil.copyfile(root / "strength/perturbation_results.csv", root / "p1_strength_sweep.csv")
    shutil.copyfile(root / "neighborhood/perturbation_results.csv", root / "p1_neighborhood_results.csv")

    manifest = json.loads((root / "density_mats/internal_subset_manifest.json").read_text())
    common_payload = json.loads((root / "strength/reference/common_reference.json").read_text())
    common_intrinsics = common_payload["camera_intrinsics"]
    public_reference_values = tuple(float(common_intrinsics[key]) for key in (
        "xi", "alpha", "fu", "fv", "cu", "cv"
    ))
    public_reference_camera = Camera(*public_reference_values, family="ds-none")
    eval_spec = common_payload["evaluation_mask"]
    public_mask = build_evaluation_mask(
        public_reference_camera,
        int(eval_spec["width"]),
        int(eval_spec["height"]),
        int(eval_spec["grid_size"]),
    )
    density_rows: list[dict[str, object]] = []
    for suffix, ratio in (("000", 0.0), ("025", 0.25), ("050", 0.5), ("100", 1.0)):
        variant = manifest["variants"][f"internal_{suffix}"]
        density_run_root = (
            root / "density/internal_000_corrected"
            if suffix == "000" and (root / "density/internal_000_corrected/perturbation_results.csv").is_file()
            else root / f"density/internal_{suffix}"
        )
        rows = read_csv(density_run_root / "perturbation_results.csv")
        for row in rows:
            if float(row["scale"]) not in (0.8, 1.0):
                continue
            # With no internal points, use the actual Outer-only branch. With
            # retained internal points, use the paired Outer+Internal branch.
            expected_method = "Outer-only" if ratio == 0.0 else "Outer+Internal"
            if row["method"] != expected_method:
                continue
            base = {
                "experiment_type": "internal_density",
                "direction_id": "P1",
                "scale": row["scale"],
                "method": expected_method,
                "internal_ratio": ratio,
                "internal_subset_fingerprint": (
                    "ablated_after_recovery:" + row["frozen_observation_fingerprint"]
                    if ratio == 0.0
                    else variant["internal_subset_fingerprint"]
                ),
                "internal_source_mode": (
                    "full_observation_set_ablated_after_recovery"
                    if ratio == 0.0
                    else "nested_internal_mat_subset"
                ),
                "weighting_mode": "current_board_type_balanced",
                "board_normalized_control_equivalent": 1,
                "final_full_ray_p95": row["final_common_full_ray_p95_deg"],
                "final_peripheral_ray_p95": row["final_common_peripheral_ray_p95_deg"],
                "valid_grid_ratio": row["final_common_valid_grid_ratio"],
                "holdout_rmse": row["heldout_overall_rmse"],
                "final_xi": row["final_xi"], "final_alpha": row["final_alpha"],
                "final_fu": row["final_fu"], "final_fv": row["final_fv"],
                "final_ray_mean": row.get("final_common_ray_mean_deg", ""),
                "final_ray_median": row.get("final_common_full_ray_median_deg", ""),
                "final_ray_p95": row.get("final_common_full_ray_p95_deg", ""),
                "final_ray_max": row.get("final_common_full_ray_max_deg", ""),
                "final_camera_fingerprint": row.get("final_camera_fingerprint", ""),
                "max_abs_parameter_difference": row.get("max_abs_parameter_difference", ""),
            }
            training_values = read_kv(Path(row["run_dir"]) / "backend_training_summary.txt")
            intrinsic_csv = training_values.get("camera_intrinsics_csv", "")
            if intrinsic_csv:
                final_values = tuple(float(value) for value in intrinsic_csv.split(","))
                if len(final_values) == 6:
                    for key, value in zip(
                        ("final_xi", "final_alpha", "final_fu", "final_fv", "final_cu", "final_cv"),
                        final_values,
                    ):
                        base[key] = f"{value:.12f}"
                    base["final_camera_fingerprint"] = ds_camera_fingerprint(final_values)
                    base["max_abs_parameter_difference"] = f"{max(abs(a - b) for a, b in zip(final_values, public_reference_values)):.12f}"
                    final_camera = Camera(*final_values, family="ds-none")
                    public_metrics = ray_metrics(public_mask, final_camera)
                    base["final_full_ray_p95"] = f"{public_metrics['full_ray_p95_deg']:.12f}"
                    base["final_peripheral_ray_p95"] = f"{public_metrics['peripheral_ray_p95_deg']:.12f}"
                    base["final_ray_mean"] = f"{public_metrics['ray_mean_deg']:.12f}"
                    base["final_ray_median"] = f"{public_metrics['full_ray_median_deg']:.12f}"
                    base["final_ray_max"] = f"{public_metrics['full_ray_max_deg']:.12f}"
            density_rows.append(base)
            equivalent = dict(base)
            equivalent["weighting_mode"] = "board_normalized_equivalent_control"
            density_rows.append(equivalent)
    write_csv(root / "p1_density_ablation.csv", density_rows)

    trajectory = [r for r in read_csv(root / "strength/incremental_trajectory.csv") if float(r["scale"]) in (0.8, 1.0)]
    write_csv(root / "p1_incremental_trajectory.csv", trajectory)
    deltas = np.asarray([float(r["delta_peripheral_ray_p95_deg"]) for r in neighborhood_pairs])
    ci_low, ci_high = bootstrap_median_ci(deltas)
    q1, q3 = np.quantile(deltas, [0.25, 0.75])
    # Strength and neighborhood clean runs are independent deterministic
    # repeats of the same full observation set.
    strength_clean = {(r["method"]): float(r["final_common_peripheral_ray_p95_deg"]) for r in strength if float(r["scale"]) == 0.0}
    neighborhood_clean = {(r["method"]): float(r["final_common_peripheral_ray_p95_deg"]) for r in neighborhood if float(r["scale"]) == 0.0}
    tie_epsilon = max(abs(strength_clean[m] - neighborhood_clean[m]) for m in strength_clean) + 1e-12
    summaries: list[dict[str, object]] = [{
        "scope": "P1_neighborhood",
        "condition_count": len(deltas),
        "median_delta_peripheral_deg": float(np.median(deltas)),
        "q1_delta_peripheral_deg": float(q1),
        "q3_delta_peripheral_deg": float(q3),
        "bootstrap_median_ci95_low_deg": ci_low,
        "bootstrap_median_ci95_high_deg": ci_high,
        "tie_epsilon_deg": tie_epsilon,
        "internal_better_count": int(np.sum(deltas > tie_epsilon)),
        "tie_count": int(np.sum(np.abs(deltas) <= tie_epsilon)),
        "outer_only_better_count": int(np.sum(deltas < -tie_epsilon)),
    }]
    for scale in (0.8, 1.0):
        for method in ("Outer-only", "Outer+Internal"):
            selected = sorted((r for r in trajectory if float(r["scale"]) == scale and r["method"] == method), key=lambda r: int(r["selected_frame_count"]))
            x = np.asarray([float(r["selected_frame_count"]) for r in selected])
            y = np.asarray([float(r["peripheral_ray_p95_deg"]) for r in selected])
            summaries.append({
                "scope": "incremental_trajectory",
                "scale": scale,
                "method": method,
                "peripheral_ray_auc_frame_deg": float(np.trapezoid(y, x)) if len(x) > 1 else math.nan,
                "peripheral_ray_auc_normalized_deg": float(np.trapezoid(y, x) / (x[-1] - x[0])) if len(x) > 1 and x[-1] > x[0] else math.nan,
            })
    write_csv(root / "p1_summary_metrics.csv", summaries)
    if not args.no_figures:
        setup_style()
        plot_strength(strength, root)
        plot_neighborhood(neighborhood_pairs, root)
        plot_trajectory(trajectory, root)
        plot_density(density_rows, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

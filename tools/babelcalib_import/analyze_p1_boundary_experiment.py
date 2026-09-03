#!/usr/bin/env python3
"""Merge formal P1 runs, audit pairing, and report the recovery boundary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_ORDER = {"Outer-only": 0, "Outer+Internal": 1}
COLORS = {"Outer-only": "#D55E00", "Outer+Internal": "#0072B2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty result: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_columns(row: dict[str, str], source: str) -> dict[str, object]:
    return {
        "direction": "P1",
        "scale": f"{float(row['scale']):.12f}",
        "method": row["method"],
        "source": source,
        "reference_scene_fingerprint": row["reference_scene_fingerprint"],
        "perturbed_scene_fingerprint": row["perturbed_scene_fingerprint"],
        "frozen_observation_fingerprint": row["frozen_observation_fingerprint"],
        "initial_camera_fingerprint": row["initial_camera_fingerprint"],
        "final_camera_fingerprint": row["final_camera_fingerprint"],
        "initial_full_ray_p95": row["initial_common_full_ray_p95_deg"],
        "initial_valid_grid_ratio": row["initial_common_valid_grid_ratio"],
        "initial_invalid_grid_count": row["initial_common_invalid_grid_count"],
        "common_full_ray_median": row["final_common_full_ray_median_deg"],
        "common_full_ray_p95": row["final_common_full_ray_p95_deg"],
        "common_peripheral_ray_median": row["final_common_peripheral_ray_median_deg"],
        "common_peripheral_ray_p95": row["final_common_peripheral_ray_p95_deg"],
        "common_valid_grid_ratio": row["final_common_valid_grid_ratio"],
        "branch_full_ray_median": row["final_branch_full_ray_median_deg"],
        "branch_full_ray_p95": row["final_branch_full_ray_p95_deg"],
        "branch_peripheral_ray_median": row["final_branch_peripheral_ray_median_deg"],
        "branch_peripheral_ray_p95": row["final_branch_peripheral_ray_p95_deg"],
        "branch_valid_grid_ratio": row["final_branch_valid_grid_ratio"],
        "holdout_rmse": row["heldout_overall_rmse"],
        "xi": row["final_xi"], "alpha": row["final_alpha"],
        "fu": row["final_fu"], "fv": row["final_fv"],
        "cu": row["final_cu"], "cv": row["final_cv"],
        "solver_status": row["solver_status"],
        "runtime_sec": row.get("runtime_sec", ""),
    }


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_metric(rows: list[dict[str, object]], key: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.15))
    for method in METHOD_ORDER:
        selected = sorted(
            (row for row in rows if row["method"] == method and float(row["scale"]) >= 0.6),
            key=lambda row: float(row["scale"]),
        )
        ax.plot(
            [float(row["scale"]) for row in selected],
            [float(row[key]) for row in selected],
            marker="o", markersize=4, linewidth=1.7,
            color=COLORS[method], label=method,
        )
    ax.set_xlabel("P1 perturbation scale $s$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0.6, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0])
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    strength = read_csv(root / "strength/perturbation_results.csv")
    boundary = read_csv(root / "boundary/perturbation_results.csv")
    boundary_pairs = read_csv(root / "boundary/paired_improvements.csv")
    if not boundary_pairs or any(row["paired_initial_state_valid"] != "1" for row in boundary_pairs):
        raise RuntimeError("Boundary sweep failed strict paired-state validation")

    strength_common = {
        row["common_reference_camera_fingerprint"] for row in strength
        if row["method"] == "Outer+Internal"
    }
    boundary_common = {
        row["common_reference_camera_fingerprint"] for row in boundary
        if row["method"] == "Outer+Internal"
    }
    if len(strength_common) != 1 or strength_common != boundary_common:
        raise RuntimeError(
            "Boundary and strength sweeps do not share the same common-reference camera"
        )

    merged: dict[tuple[float, str], dict[str, object]] = {}
    for source, source_rows in (("strength", strength), ("boundary", boundary)):
        for row in source_rows:
            scale = float(row["scale"])
            if source == "boundary" or (scale not in {0.0}):
                merged[(scale, row["method"])] = select_columns(row, source)
    all_rows = sorted(
        merged.values(),
        key=lambda row: (float(row["scale"]), METHOD_ORDER[str(row["method"])]),
    )
    boundary_scales = {0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0}
    boundary_rows = [row for row in all_rows if float(row["scale"]) in boundary_scales]
    write_csv(root / "p1_boundary_sweep.csv", boundary_rows)
    write_csv(root / "p1_branch_relative_metrics.csv", all_rows)

    configure_plot_style()
    plot_metric(
        all_rows, "branch_peripheral_ray_p95",
        "Branch-relative peripheral Ray P95 (deg)",
        root / "p1_boundary_branch_relative_peripheral.png",
    )
    plot_metric(
        all_rows, "common_peripheral_ray_p95",
        "Common-reference peripheral Ray P95 (deg)",
        root / "p1_boundary_common_reference_peripheral.png",
    )
    audit = {
        "paired_condition_count": len(boundary_pairs),
        "all_paired_initial_states_valid": True,
        "common_reference_camera_fingerprint": next(iter(strength_common)),
        "branch_relative_protocol": "per_pixel_unprojection_on_fixed_common_mask_against_method_specific_clean_camera",
        "p95_values_subtracted_directly": False,
        "final_ba_added": False,
    }
    (root / "p1_boundary_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

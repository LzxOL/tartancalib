#!/usr/bin/env python3
"""Audit P1 s=0.95 repeatability and summarize its dense local boundary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"Outer-only": "#D55E00", "Outer+Internal": "#0072B2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def accepted_frame_count(run_dir: str) -> int:
    path = Path(run_dir) / "trial_backend_frame_board_selection_decisions.csv"
    rows = read_csv(path)
    return len({
        row["frame_index"] for row in rows
        if row.get("persistent_incremental_batch_accepted") == "1"
    })


def compact(row: dict[str, str], source: str) -> dict[str, object]:
    return {
        "scale": f"{float(row['scale']):.12f}",
        "method": row["method"],
        "source": source,
        "common_full_ray_p95": row["final_common_full_ray_p95_deg"],
        "common_peripheral_ray_p95": row["final_common_peripheral_ray_p95_deg"],
        "branch_full_ray_p95": row["final_branch_full_ray_p95_deg"],
        "branch_peripheral_ray_p95": row["final_branch_peripheral_ray_p95_deg"],
        "holdout_rmse": row["heldout_overall_rmse"],
        "accepted_frame_count": accepted_frame_count(row["run_dir"]),
        "final_camera_fingerprint": row["final_camera_fingerprint"],
        "reference_scene_fingerprint": row["reference_scene_fingerprint"],
        "perturbed_scene_fingerprint": row["perturbed_scene_fingerprint"],
        "frozen_observation_fingerprint": row["frozen_observation_fingerprint"],
    }


def plot(rows: list[dict[str, object]], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    specs = [
        ("common_peripheral_ray_p95", "Common-reference peripheral P95 (deg)"),
        ("branch_peripheral_ray_p95", "Branch-relative peripheral P95 (deg)"),
    ]
    for ax, (key, ylabel) in zip(axes, specs):
        ax.axvspan(0.95, 0.955, color="#CC6677", alpha=0.13, linewidth=0)
        for method in COLORS:
            selected = sorted(
                (row for row in rows if row["method"] == method),
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
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    formal = read_csv(root / "boundary/perturbation_results.csv")
    local = read_csv(root / "boundary_local/perturbation_results.csv")
    repeat = read_csv(root / "s095_repeat/perturbation_results.csv")

    selected: dict[tuple[float, str], dict[str, object]] = {}
    for source, rows in (("formal", formal), ("local_dense", local)):
        for row in rows:
            scale = float(row["scale"])
            if scale in {0.85, 0.9, 0.925, 0.94, 0.945, 0.95, 0.955, 0.96, 0.975}:
                selected[(scale, row["method"])] = compact(row, source)
    # The formal s=1.0 endpoint lives in the original boundary merge output.
    strength = read_csv(root / "strength/perturbation_results.csv")
    for row in strength:
        if float(row["scale"]) == 1.0:
            selected[(1.0, row["method"])] = compact(row, "formal_endpoint")
    result = sorted(selected.values(), key=lambda row: (float(row["scale"]), str(row["method"])))
    write_csv(root / "p1_local_boundary_stability.csv", result)

    repeats: list[dict[str, object]] = []
    for repeat_id, rows in (("formal", formal), ("dense_repeat", local), ("independent_repeat", repeat)):
        for row in rows:
            if float(row["scale"]) == 0.95:
                item = compact(row, repeat_id)
                item["repeat_id"] = repeat_id
                repeats.append(item)
    write_csv(root / "p1_s095_repeatability.csv", repeats)

    internal_better_common = 0
    condition_count = 0
    by_key = {(float(row["scale"]), str(row["method"])): row for row in result}
    for scale in sorted({float(row["scale"]) for row in result}):
        outer = by_key[(scale, "Outer-only")]
        internal = by_key[(scale, "Outer+Internal")]
        condition_count += 1
        internal_better_common += int(
            float(internal["common_peripheral_ray_p95"])
            < float(outer["common_peripheral_ray_p95"])
        )
    internal_repeat = [row for row in repeats if row["method"] == "Outer+Internal"]
    repeat_fingerprints = {str(row["final_camera_fingerprint"]) for row in internal_repeat}
    summary = {
        "condition_count": condition_count,
        "internal_better_common_count": internal_better_common,
        "outer_only_better_common_count": condition_count - internal_better_common,
        "failure_band": [0.95, 0.955],
        "s095_repeat_count": len(internal_repeat),
        "s095_unique_internal_final_camera_fingerprint_count": len(repeat_fingerprints),
        "s095_deterministically_reproduced": len(repeat_fingerprints) == 1,
        "interpretation": "deterministic_discrete_selection_acceptance_failure_not_random_cherry_pick",
    }
    (root / "p1_local_boundary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(result, root / "p1_local_boundary_stability.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

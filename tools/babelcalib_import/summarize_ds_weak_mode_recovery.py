#!/usr/bin/env python3
"""Aggregate the preregistered right-camera DS weak-mode experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.12f}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def find_condition(
    rows: list[dict[str, str]], direction: str, target: float, mode: str
) -> dict[str, str]:
    matches = [
        row for row in rows
        if row["direction"] == direction and row["mode"] == mode and
        math.isclose(float(row["scale"]), target, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {direction}/{target}/{mode} row, got {len(matches)}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--replication-root-pattern", required=True)
    parser.add_argument("--replication-count", type=int, default=8)
    parser.add_argument("--subspace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    roots = [("base_00_to_15", args.base_root.resolve())]
    for index in range(1, args.replication_count + 1):
        roots.append((
            f"pose_pair_{index:02d}",
            Path(args.replication_root_pattern.format(index=index)).resolve(),
        ))

    observability_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    gains: list[float] = []
    for label, root in roots:
        audit = json.loads((root / "outer_only_weak_mode_audit.json").read_text())
        outer_eigenvalues = audit["fisher_eigenvalues"]
        internal_eigenvalues = audit["outer_internal_fisher_eigenvalues"]
        gain = float(audit["minimum_eigenvalue_gain_outer_internal_over_outer_only"])
        gains.append(gain)
        observability_rows.append({
            "schedule": label,
            "outer_point_count": audit["outer_point_count"],
            "outer_internal_point_count": audit["outer_internal_point_count"],
            "outer_lambda_min": outer_eigenvalues[0],
            "outer_internal_lambda_min": internal_eigenvalues[0],
            "lambda_min_gain": gain,
            "outer_condition_number": audit["fisher_condition_number"],
            "outer_internal_condition_number": audit["outer_internal_fisher_condition_number"],
        })

        rows = read_csv(root / "weak_mode_perturbation_results.csv")
        outer = find_condition(rows, "W1_plus", 2.5, "outer_only")
        internal = find_condition(rows, "W1_plus", 2.5, "outer_internal")
        outer_branch = float(outer["final_branch_peripheral_ray_p95_deg"])
        internal_branch = float(internal["final_branch_peripheral_ray_p95_deg"])
        delta = outer_branch - internal_branch
        deltas.append(delta)
        recovery_rows.append({
            "schedule": label,
            "initial_target_peripheral_ray_p95_deg": 2.5,
            "outer_branch_relative_full_ray_p95_deg":
                float(outer["final_branch_full_ray_p95_deg"]),
            "internal_branch_relative_full_ray_p95_deg":
                float(internal["final_branch_full_ray_p95_deg"]),
            "outer_branch_relative_peripheral_ray_p95_deg": outer_branch,
            "internal_branch_relative_peripheral_ray_p95_deg": internal_branch,
            "paired_peripheral_improvement_deg": delta,
            "outer_common_peripheral_ray_p95_deg":
                float(outer["final_common_peripheral_ray_p95_deg"]),
            "internal_common_peripheral_ray_p95_deg":
                float(internal["final_common_peripheral_ray_p95_deg"]),
            "outer_holdout_rmse_px": float(outer["heldout_overall_rmse"]),
            "internal_holdout_rmse_px": float(internal["heldout_overall_rmse"]),
            "outer_committed_batch_count": int(outer["backend_committed_batch_count"]),
            "internal_committed_batch_count": int(internal["backend_committed_batch_count"]),
            "paired_initial_scene_valid": int(
                outer["reference_scene_fingerprint"] == internal["reference_scene_fingerprint"] and
                outer["perturbed_scene_fingerprint"] == internal["perturbed_scene_fingerprint"] and
                outer["frozen_observation_fingerprint"] == internal["frozen_observation_fingerprint"] and
                outer["backend_seed_set_fingerprint"] == internal["backend_seed_set_fingerprint"] and
                outer["backend_attempted_schedule_fingerprint"] == internal["backend_attempted_schedule_fingerprint"]
            ),
        })

    subspace_rows: list[dict[str, Any]] = []
    subspace_deltas: list[float] = []
    raw_subspace = read_csv(args.subspace_root / "weak_mode_perturbation_results.csv")
    directions = sorted({row["direction"] for row in raw_subspace})
    for direction in directions:
        outer = find_condition(raw_subspace, direction, 2.5, "outer_only")
        internal = find_condition(raw_subspace, direction, 2.5, "outer_internal")
        delta = (
            float(outer["final_branch_peripheral_ray_p95_deg"]) -
            float(internal["final_branch_peripheral_ray_p95_deg"])
        )
        subspace_deltas.append(delta)
        subspace_rows.append({
            "direction": direction,
            "angle_deg": float(outer["weak_subspace_angle_deg"]),
            "outer_branch_relative_peripheral_ray_p95_deg":
                float(outer["final_branch_peripheral_ray_p95_deg"]),
            "internal_branch_relative_peripheral_ray_p95_deg":
                float(internal["final_branch_peripheral_ray_p95_deg"]),
            "paired_peripheral_improvement_deg": delta,
            "outer_committed_batch_count": int(outer["backend_committed_batch_count"]),
            "internal_committed_batch_count": int(internal["backend_committed_batch_count"]),
        })

    write_csv(output / "observability_by_schedule.csv", observability_rows)
    write_csv(output / "w1_plus_recovery_by_schedule.csv", recovery_rows)
    write_csv(output / "weak_subspace_recovery.csv", subspace_rows)
    summary = {
        "schedule_count": len(roots),
        "all_input_pairs_valid": all(row["paired_initial_scene_valid"] == 1 for row in recovery_rows),
        "lambda_min_gain_min": min(gains),
        "lambda_min_gain_median": float(np.median(gains)),
        "lambda_min_gain_max": max(gains),
        "lambda_min_gain_all_above_one": all(gain > 1.0 for gain in gains),
        "lambda_min_gain_one_sided_sign_test_p": float(
            binomtest(
                sum(gain > 1.0 for gain in gains), len(gains), 0.5,
                alternative="greater",
            ).pvalue
        ),
        "w1_plus_internal_better_count": sum(delta > 1e-9 for delta in deltas),
        "w1_plus_outer_better_count": sum(delta < -1e-9 for delta in deltas),
        "w1_plus_tie_count": sum(abs(delta) <= 1e-9 for delta in deltas),
        "w1_plus_paired_improvement_median_deg": float(np.median(deltas)),
        "w1_plus_paired_improvement_mean_deg": float(np.mean(deltas)),
        "w1_plus_improvement_one_sided_wilcoxon_p": float(
            wilcoxon(deltas, alternative="greater", zero_method="wilcox").pvalue
        ),
        "w1_plus_improvement_one_sided_sign_test_p": float(
            binomtest(
                sum(delta > 0.0 for delta in deltas if delta != 0.0),
                sum(delta != 0.0 for delta in deltas), 0.5,
                alternative="greater",
            ).pvalue
        ),
        "w1_plus_outer_branch_p95_median_deg": float(np.median([
            row["outer_branch_relative_peripheral_ray_p95_deg"] for row in recovery_rows
        ])),
        "w1_plus_internal_branch_p95_median_deg": float(np.median([
            row["internal_branch_relative_peripheral_ray_p95_deg"] for row in recovery_rows
        ])),
        "subspace_direction_count": len(subspace_rows),
        "subspace_internal_better_count": sum(delta > 1e-9 for delta in subspace_deltas),
        "subspace_outer_better_count": sum(delta < -1e-9 for delta in subspace_deltas),
        "subspace_tie_count": sum(abs(delta) <= 1e-9 for delta in subspace_deltas),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

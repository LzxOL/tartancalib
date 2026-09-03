#!/usr/bin/env python3
"""Summarize fixed-Backend-input P1 recovery against end-to-end selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from run_ds_perturbation_sweep import (
    Camera, build_evaluation_mask, camera_fingerprint, ray_metrics,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-root", required=True, type=Path)
    parser.add_argument("--end-to-end-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixed_root = args.fixed_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    formal_root = args.end_to_end_csv.resolve().parent
    formal_common_payload = json.loads(
        (formal_root / "strength/reference/common_reference.json").read_text()
    )
    formal_intrinsics = formal_common_payload["camera_intrinsics"]
    formal_common_camera = Camera(*(
        float(formal_intrinsics[key])
        for key in ("xi", "alpha", "fu", "fv", "cu", "cv")
    ), family="ds-none")
    eval_spec = formal_common_payload["evaluation_mask"]
    formal_mask = build_evaluation_mask(
        formal_common_camera,
        int(eval_spec["width"]), int(eval_spec["height"]),
        int(eval_spec["grid_size"]),
    )
    fixed = [
        row for row in read_csv(fixed_root / "perturbation_results.csv")
        if float(row["scale"]) > 0.0
    ]
    paired = read_csv(fixed_root / "paired_improvements.csv")
    required_pair_fields = (
        "paired_initial_state_valid",
        "backend_seed_set_identical",
        "backend_attempted_schedule_identical",
        "backend_committed_schedule_identical",
        "backend_committed_set_identical",
    )
    if any(any(row[field] != "1" for field in required_pair_fields) for row in paired):
        raise RuntimeError("fixed Backend input audit failed")

    compact: list[dict[str, object]] = []
    for row in sorted(fixed, key=lambda value: (float(value["scale"]), value["method"])):
        final_camera = Camera(*(
            float(row[key]) for key in
            ("final_xi", "final_alpha", "final_fu", "final_fv", "final_cu", "final_cv")
        ), family="ds-none")
        formal_common_metrics = ray_metrics(formal_mask, final_camera)
        compact.append({
            "scale": row["scale"], "method": row["method"],
            "initial_full_ray_p95": row["initial_common_full_ray_p95_deg"],
            "common_full_ray_p95": f"{formal_common_metrics['full_ray_p95_deg']:.12f}",
            "common_peripheral_ray_p95": f"{formal_common_metrics['peripheral_ray_p95_deg']:.12f}",
            "branch_full_ray_p95": row["final_branch_full_ray_p95_deg"],
            "branch_peripheral_ray_p95": row["final_branch_peripheral_ray_p95_deg"],
            "valid_grid_ratio": row["final_common_valid_grid_ratio"],
            "holdout_rmse": row["heldout_overall_rmse"],
            "xi": row["final_xi"], "alpha": row["final_alpha"],
            "fu": row["final_fu"], "fv": row["final_fv"],
            "cu": row["final_cu"], "cv": row["final_cv"],
            "backend_seed_set_fingerprint": row["backend_seed_set_fingerprint"],
            "backend_schedule_fingerprint": row["backend_committed_schedule_fingerprint"],
            "backend_frame_board_set_fingerprint": row["backend_committed_frame_board_set_fingerprint"],
            "backend_batch_count": row["backend_committed_batch_count"],
            "backend_frame_board_count": row["backend_committed_frame_board_count"],
            "final_camera_fingerprint": row["final_camera_fingerprint"],
        })
    write_csv(output / "fixed_backend_input_results.csv", compact)

    end_to_end = read_csv(args.end_to_end_csv.resolve())
    e2e_index = {
        (float(row["scale"]), row["method"]): row for row in end_to_end
    }
    comparison: list[dict[str, object]] = []
    for row in compact:
        key = (float(row["scale"]), str(row["method"]))
        e2e = e2e_index[key]
        comparison.append({
            "scale": row["scale"], "method": row["method"],
            "fixed_common_peripheral_ray_p95": row["common_peripheral_ray_p95"],
            "end_to_end_common_peripheral_ray_p95": e2e["common_peripheral_ray_p95"],
            "fixed_branch_peripheral_ray_p95": row["branch_peripheral_ray_p95"],
            "end_to_end_branch_peripheral_ray_p95": e2e["branch_peripheral_ray_p95"],
            "fixed_holdout_rmse": row["holdout_rmse"],
            "end_to_end_holdout_rmse": e2e["holdout_rmse"],
        })
    write_csv(output / "fixed_vs_end_to_end.csv", comparison)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharex=True)
    for ax, prefix, title in (
        (axes[0], "fixed", "Fixed Backend input"),
        (axes[1], "end_to_end", "End-to-end selection"),
    ):
        for method in COLORS:
            selected = sorted(
                (row for row in comparison if row["method"] == method),
                key=lambda value: float(value["scale"]),
            )
            ax.plot(
                [float(row["scale"]) for row in selected],
                [float(row[f"{prefix}_common_peripheral_ray_p95"]) for row in selected],
                marker="o", linewidth=1.7, markersize=4,
                color=COLORS[method], label=method,
            )
        ax.set_title(title)
        ax.set_xlabel("P1 perturbation scale $s$")
        ax.set_ylabel("Common-reference peripheral Ray P95 (deg)")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(frameon=False)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.86, wspace=0.38)
    fig.savefig(output / "fixed_vs_end_to_end_peripheral_ray_p95.png")
    plt.close(fig)

    schedules = {row["backend_schedule_fingerprint"] for row in compact}
    sets = {row["backend_frame_board_set_fingerprint"] for row in compact}
    summary = {
        "success": True,
        "condition_count": len(paired),
        "all_pair_checks_passed": True,
        "unique_backend_schedule_fingerprint_count": len(schedules),
        "unique_backend_frame_board_set_fingerprint_count": len(sets),
        "backend_batch_count": int(compact[0]["backend_batch_count"]),
        "backend_frame_board_count": int(compact[0]["backend_frame_board_count"]),
        "only_experimental_variable": "internal_residual_enabled",
        "formal_common_reference_camera_fingerprint": camera_fingerprint(formal_common_camera),
        "common_reference_protocol": "original_formal_end_to_end_common_camera_and_fixed_mask",
        "final_ba_added": False,
    }
    (output / "fixed_backend_input_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

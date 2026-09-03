#!/usr/bin/env python3
"""Evaluate semi-synthetic DS runs against the known source camera."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import run_ds_perturbation_sweep as sweep
import run_ds_weak_mode_perturbation as weak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--run-pattern", required=True)
    parser.add_argument("--run-count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-mat", type=Path, required=True)
    args = parser.parse_args()
    source_scene = weak.parse_scene(args.source_scene.resolve())
    width, height = sweep.image_size_from_mat(args.train_mat.resolve())
    mask = sweep.build_evaluation_mask(source_scene.camera, width, height, 121)
    rows: list[dict[str, object]] = []
    for index in range(1, args.run_count + 1):
        root = Path(args.run_pattern.format(index=index)).resolve()
        result_rows = []
        audit = json.loads((root / "weak_mode_experiment_audit.json").read_text())
        with (root / "weak_mode_perturbation_results.csv").open(newline="", encoding="utf-8") as handle:
            result_rows = list(csv.DictReader(handle))
        for row in result_rows:
            final = sweep.camera_from_training_summary(
                Path(row["run_dir"]) / "backend_training_summary.txt",
                "ds-none",
            )
            final_metrics = sweep.ray_metrics(mask, final, source_scene.camera)
            rows.append({
                "pair": index,
                "method": row["method"],
                "initial_peripheral_ray_p95_deg": row["initial_common_peripheral_ray_p95_deg"],
                "source_relative_full_ray_p95_deg": final_metrics["full_ray_p95_deg"],
                "source_relative_peripheral_ray_p95_deg": final_metrics["peripheral_ray_p95_deg"],
                "source_relative_valid_grid_ratio": final_metrics["valid_grid_ratio"],
                "heldout_overall_rmse_px": row["heldout_overall_rmse"],
                "committed_batch_count": row["backend_committed_batch_count"],
                "final_xi": final.xi,
                "final_alpha": final.alpha,
                "final_fu": final.fu,
                "final_fv": final.fv,
                "final_cu": final.cu,
                "final_cv": final.cv,
                "paired_input_valid": int(
                    audit["strictly_valid_input_pair_count"] > 0
                ),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"row_count": len(rows), "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Controlled DS intrinsic-only recovery with a known semi-synthetic truth.

The scene poses and board layout are fixed.  Both branches start from the
same weak-mode camera perturbation and the same noisy observations; the only
difference is whether internal points are included in the camera residual.
This is intentionally separate from the full Stage5 pose/layout BA because it
measures intrinsic information rather than allowing nuisance variables to
hide it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import wilcoxon

import run_ds_perturbation_sweep as sweep
import run_ds_weak_mode_perturbation as weak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scene", type=Path, required=True)
    parser.add_argument("--training-points", type=Path, required=True)
    parser.add_argument("--width", type=int, default=4512)
    parser.add_argument("--height", type=int, default=4512)
    parser.add_argument("--target-peripheral-p95-deg", type=float, default=4.0)
    parser.add_argument("--noise-sigma-px", type=float, default=0.25)
    parser.add_argument("--noise-seeds", default="11,22,33,44,55")
    parser.add_argument("--grid-size", type=int, default=121)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_point_rows(path: Path, scene: weak.Scene) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame_index"])
            board = int(row["board_id"])
            if frame not in scene.frames or board not in scene.boards:
                continue
            local = np.asarray([
                float(row["target_x"]), float(row["target_y"]),
                float(row["target_z"]), 1.0,
            ], dtype=np.float64)
            camera_point = (scene.frames[frame] @ scene.boards[board] @ local)[:3]
            rows.append({
                "frame": frame,
                "board": board,
                "point": local,
                "point_type": row["point_type"],
                "camera_point": camera_point,
            })
    if not rows:
        raise RuntimeError(f"No usable observations in {path}")
    return rows


def coordinates(camera: sweep.Camera, width: int, height: int) -> np.ndarray:
    return weak.camera_to_coordinates(camera, width, height)


def camera(values: np.ndarray, width: int, height: int) -> sweep.Camera:
    return weak.camera_from_coordinates(values, width, height)


def run_branch(
    initial: sweep.Camera,
    truth: sweep.Camera,
    rows: list[dict[str, object]],
    noise: np.ndarray,
    point_types: set[str],
    width: int,
    height: int,
    grid_size: int,
) -> tuple[sweep.Camera, dict[str, float | int]]:
    selected = [
        (index, row) for index, row in enumerate(rows)
        if row["point_type"] in point_types
    ]
    points = np.asarray([row["camera_point"] for _, row in selected], dtype=np.float64)
    clean, valid = weak.project_ds(truth, points)
    if not np.all(valid):
        raise RuntimeError("Ground-truth projection contains invalid observations")
    observed = clean + np.asarray([noise[index] for index, _ in selected])

    def residual(values: np.ndarray) -> np.ndarray:
        candidate = camera(values, width, height)
        projected, valid_candidate = weak.project_ds(candidate, points)
        if not np.all(valid_candidate):
            return np.full(observed.size, 1e6, dtype=np.float64)
        return (projected - observed).reshape(-1)

    initial_values = coordinates(initial, width, height)
    lower = np.asarray([-0.95, 0.01, math.log(400.0), math.log(400.0), 0.2, 0.2])
    upper = np.asarray([4.0, 0.99, math.log(6000.0), math.log(6000.0), 0.8, 0.8])
    result = least_squares(
        residual,
        np.clip(initial_values, lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper),
        method="trf",
        loss="linear",
        x_scale="jac",
        max_nfev=250,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    final = camera(result.x, width, height)
    residual_values = residual(result.x)
    metrics = sweep.ray_metrics(
        sweep.build_evaluation_mask(truth, width, height, grid_size), final,
    )
    metrics.update({
        "point_count": len(selected),
        "image_rmse_px": float(np.sqrt(np.mean(residual_values ** 2))),
        "solver_success": int(result.success),
        "iterations": int(result.nfev),
        "final_cost": float(result.cost),
    })
    return final, metrics


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = weak.parse_scene(args.reference_scene.resolve())
    rows = read_point_rows(args.training_points.resolve(), scene)
    outer_rows = [row for row in rows if row["point_type"] == "outer"]
    width, height = args.width, args.height
    mask = sweep.build_evaluation_mask(scene.camera, width, height, args.grid_size)
    modes, _, fisher_audit = weak.compute_weak_modes(scene, outer_rows, width, height)
    _, _, full_fisher_audit = weak.compute_weak_modes(scene, rows, width, height)
    amplitude, initial, initial_metrics = weak.calibrate_perturbation(
        scene.camera, modes[0].direction, 1, args.target_peripheral_p95_deg,
        width, height, mask, 0.99,
    )
    seeds = [int(value.strip()) for value in args.noise_seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--noise-seeds must not be empty")
    rows_out: list[dict[str, object]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pair_noise = rng.normal(0.0, args.noise_sigma_px, size=(len(rows), 2))
        for label, types in (("Outer-only", {"outer"}), ("Outer+Internal", {"outer", "internal"})):
            final, metrics = run_branch(
                initial, scene.camera, rows, pair_noise, types, width, height,
                args.grid_size,
            )
            rows_out.append({
                "seed": seed,
                "method": label,
                "target_initial_peripheral_ray_p95_deg": args.target_peripheral_p95_deg,
                "noise_sigma_px": args.noise_sigma_px,
                "perturbation_amplitude": amplitude,
                "initial_xi": initial.xi,
                "initial_alpha": initial.alpha,
                "initial_fu": initial.fu,
                "initial_fv": initial.fv,
                "final_xi": final.xi,
                "final_alpha": final.alpha,
                "final_fu": final.fu,
                "final_fv": final.fv,
                "final_cu": final.cu,
                "final_cv": final.cv,
                "truth_xi": scene.camera.xi,
                "truth_alpha": scene.camera.alpha,
                "truth_fu": scene.camera.fu,
                "truth_fv": scene.camera.fv,
                "truth_cu": scene.camera.cu,
                "truth_cv": scene.camera.cv,
                "final_full_ray_p95_deg": metrics["full_ray_p95_deg"],
                "final_peripheral_ray_p95_deg": metrics["peripheral_ray_p95_deg"],
                "final_ray_median_deg": metrics["full_ray_median_deg"],
                "valid_grid_ratio": metrics["valid_grid_ratio"],
                "image_rmse_px": metrics["image_rmse_px"],
                "point_count": metrics["point_count"],
                "solver_success": metrics["solver_success"],
                "iterations": metrics["iterations"],
            })
    fields = list(rows_out[0])
    with (output / "intrinsic_only_recovery.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    summary_rows: list[dict[str, object]] = []
    for label in ("Outer-only", "Outer+Internal"):
        group = [row for row in rows_out if row["method"] == label]
        summary_rows.append({
            "method": label,
            "run_count": len(group),
            "ray_p95_median_deg": float(np.median([row["final_peripheral_ray_p95_deg"] for row in group])),
            "ray_p95_mean_deg": float(np.mean([row["final_peripheral_ray_p95_deg"] for row in group])),
            "full_ray_p95_median_deg": float(np.median([row["final_full_ray_p95_deg"] for row in group])),
            "image_rmse_median_px": float(np.median([row["image_rmse_px"] for row in group])),
            "xi_abs_error_median": float(np.median([abs(row["final_xi"] - row["truth_xi"]) for row in group])),
            "focal_relative_error_median": float(np.median([
                0.5 * (abs(row["final_fu"] / row["truth_fu"] - 1.0) +
                       abs(row["final_fv"] / row["truth_fv"] - 1.0))
                for row in group
            ])),
        })
    with (output / "intrinsic_only_recovery_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    paired_rows: list[dict[str, object]] = []
    for seed in seeds:
        outer = next(row for row in rows_out if row["seed"] == seed and row["method"] == "Outer-only")
        internal = next(row for row in rows_out if row["seed"] == seed and row["method"] == "Outer+Internal")
        paired_rows.append({
            "seed": seed,
            "outer_peripheral_ray_p95_deg": outer["final_peripheral_ray_p95_deg"],
            "internal_peripheral_ray_p95_deg": internal["final_peripheral_ray_p95_deg"],
            "paired_improvement_deg": outer["final_peripheral_ray_p95_deg"] - internal["final_peripheral_ray_p95_deg"],
        })
    with (output / "intrinsic_only_paired_improvements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    paired_deltas = np.asarray([row["paired_improvement_deg"] for row in paired_rows], dtype=np.float64)
    statistics = {
        "pair_count": len(paired_rows),
        "internal_better_count": int(np.count_nonzero(paired_deltas > 0.0)),
        "outer_better_count": int(np.count_nonzero(paired_deltas < 0.0)),
        "tie_count": int(np.count_nonzero(paired_deltas == 0.0)),
        "paired_improvement_median_deg": float(np.median(paired_deltas)),
        "paired_improvement_mean_deg": float(np.mean(paired_deltas)),
        "one_sided_wilcoxon_p": float(wilcoxon(paired_deltas, alternative="greater").pvalue),
    }
    (output / "intrinsic_only_recovery_statistics.json").write_text(
        __import__("json").dumps(statistics, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "protocol.json").write_text(
        __import__("json").dumps({
            "protocol": "fixed_scene_intrinsic_only_ground_truth_recovery",
            "reference_scene": str(args.reference_scene.resolve()),
            "reference_scene_sha256": "sha256:" + hashlib.sha256(
                args.reference_scene.resolve().read_bytes()
            ).hexdigest(),
            "training_points": str(args.training_points.resolve()),
            "training_points_sha256": "sha256:" + hashlib.sha256(
                args.training_points.resolve().read_bytes()
            ).hexdigest(),
            "ground_truth_camera_fingerprint": sweep.camera_fingerprint(scene.camera),
            "outer_point_count": len(outer_rows),
            "all_point_count": len(rows),
            "weak_mode": "W1",
            "outer_fisher_eigenvalues": fisher_audit["fisher_eigenvalues"],
            "outer_internal_fisher_eigenvalues": full_fisher_audit["fisher_eigenvalues"],
            "minimum_fisher_eigenvalue_gain": (
                full_fisher_audit["fisher_eigenvalues"][0] /
                fisher_audit["fisher_eigenvalues"][0]
            ),
            "target_initial_peripheral_ray_p95_deg": args.target_peripheral_p95_deg,
            "initial_full_ray_p95_deg": initial_metrics["full_ray_p95_deg"],
            "initial_peripheral_ray_p95_deg": initial_metrics["peripheral_ray_p95_deg"],
            "noise_sigma_px": args.noise_sigma_px,
            "noise_seeds": seeds,
            "same_noise_within_pair": True,
            "pose_and_layout_optimized": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output / 'intrinsic_only_recovery.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

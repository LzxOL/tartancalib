#!/usr/bin/env python3
"""Summarize the fixed-input 1444190 local-whitening ablation."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from generate_residual_mode_multiset_table import (
    angular_error_rad,
    camera_parameters,
    load_holdout_points,
    point_key,
    polar_angle_deg,
    read_summary,
    unproject_ds,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT / "paper/experiments/local_whitened_spherical_ablation_1444190_seed1337"
)
RESULT_DIRS = {
    "Pixel": ROOT / "result_may/stage5_fixed_backend_current_20260715_1444190clear_pixel",
    "Spherical": ROOT / "result_may/stage5_fixed_backend_current_20260715_1444190clear_spherical",
    "Local-Whitened": ROOT
    / "result_may/stage5_fixed_backend_current_20260715_1444190clear_spherical_local_whitened_v3",
}
POLAR_BINS = ((0.0, 30.0, "0--30"), (30.0, 50.0, "30--50"),
              (50.0, math.inf, "50+"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    initializations = {
        method: read_summary(path / "auto_camera_initialization_summary.txt")
        for method, path in RESULT_DIRS.items()
    }
    initial_intrinsics = {
        method: summary["selected_intrinsics_csv"]
        for method, summary in initializations.items()
    }
    if len(set(initial_intrinsics.values())) != 1:
        raise RuntimeError(f"Initialization mismatch: {initial_intrinsics}")

    backend_inputs = {
        method: (path / "backend_input_used_frame_board_list.csv").read_bytes()
        for method, path in RESULT_DIRS.items()
    }
    if len(set(backend_inputs.values())) != 1:
        raise RuntimeError("Backend inputs differ")

    summaries = {
        method: read_summary(path / "backend_holdout_summary.txt")
        for method, path in RESULT_DIRS.items()
    }
    selection = {
        method: read_summary(path / "trial_backend_frame_board_selection_summary.txt")
        for method, path in RESULT_DIRS.items()
    }
    if len({summary["split_signature"] for summary in summaries.values()}) != 1:
        raise RuntimeError("Split signatures differ")

    reference_camera = tuple(
        float(value) for value in initial_intrinsics["Pixel"].split(",")
    )
    points = {
        method: load_holdout_points(path) for method, path in RESULT_DIRS.items()
    }
    reference_observations = {
        point_key(point): (float(point["observed_x"]), float(point["observed_y"]))
        for point in points["Pixel"]
    }
    reference_keys = set(reference_observations)

    rows: list[dict[str, object]] = []
    polar_rows: list[dict[str, object]] = []
    for method in RESULT_DIRS:
        if {point_key(point) for point in points[method]} != reference_keys:
            raise RuntimeError(f"Holdout correspondence mismatch for {method}")
        camera = camera_parameters(summaries[method])
        pixel_errors: list[float] = []
        angular_errors: list[float] = []
        accumulators = {
            label: {"count": 0, "pixel_sq": 0.0, "angular_sq": 0.0}
            for _, _, label in POLAR_BINS
        }
        for point in points[method]:
            key = point_key(point)
            observed_x = float(point["observed_x"])
            observed_y = float(point["observed_y"])
            if (observed_x, observed_y) != reference_observations[key]:
                raise RuntimeError(f"Observed pixel mismatch for {method}: {key}")
            reference_ray = unproject_ds(observed_x, observed_y, reference_camera)
            observed_ray = unproject_ds(observed_x, observed_y, camera)
            predicted_ray = unproject_ds(
                float(point["predicted_x"]), float(point["predicted_y"]), camera
            )
            if reference_ray is None or observed_ray is None or predicted_ray is None:
                raise RuntimeError(f"Invalid ray for {method}: {key}")
            pixel_error = float(point["residual_norm"])
            angular_error = angular_error_rad(observed_ray, predicted_ray)
            pixel_errors.append(pixel_error)
            angular_errors.append(angular_error)
            polar = polar_angle_deg(reference_ray)
            for lower, upper, label in POLAR_BINS:
                if lower <= polar < upper:
                    accumulator = accumulators[label]
                    accumulator["count"] += 1
                    accumulator["pixel_sq"] += pixel_error * pixel_error
                    accumulator["angular_sq"] += angular_error * angular_error
                    break

        for _, _, label in POLAR_BINS:
            accumulator = accumulators[label]
            count = int(accumulator["count"])
            polar_rows.append(
                {
                    "method": method,
                    "polar_bin_deg": label,
                    "point_count": count,
                    "pixel_rmse": math.sqrt(accumulator["pixel_sq"] / count),
                    "true_angular_rmse_mrad": 1000.0
                    * math.sqrt(accumulator["angular_sq"] / count),
                    "bin_assignment_camera": "shared_stage5_initial_camera",
                }
            )

        summary = summaries[method]
        selected = selection[method]
        rows.append(
            {
                "method": method,
                "holdout_rmse_px": float(summary["overall_rmse"]),
                "holdout_p95_px": float(summary["p95_reprojection_error"]),
                "outer_rmse_px": float(summary["outer_only_rmse"]),
                "internal_rmse_px": float(summary["internal_only_rmse"]),
                "inlier_at_1px_percent": 100.0
                * sum(error <= 1.0 for error in pixel_errors)
                / len(pixel_errors),
                "true_angular_rmse_mrad": 1000.0
                * math.sqrt(sum(error * error for error in angular_errors) / len(angular_errors)),
                "ba_time_seconds": float(
                    selected["persistent_incremental_total_elapsed_time_seconds"]
                ),
                "camera_xi": camera[0],
                "camera_alpha": camera[1],
                "camera_fu": camera[2],
                "camera_fv": camera[3],
                "camera_cu": camera[4],
                "camera_cv": camera[5],
                "whitening_success_count": int(
                    selected["persistent_incremental_angular_local_whitening_success_count"]
                ),
                "whitening_failure_count": int(
                    selected["persistent_incremental_angular_local_whitening_failure_count"]
                ),
                "whitening_clamped_count": int(
                    selected["persistent_incremental_angular_local_whitening_clamped_count"]
                ),
                "whitening_sigma_mean_rad": float(
                    selected["persistent_incremental_angular_local_whitening_sigma_mean_rad"]
                ),
                "whitening_weight_mean": float(
                    selected["persistent_incremental_angular_local_whitening_weight_mean"]
                ),
                "source_result_dir": str(RESULT_DIRS[method].relative_to(ROOT)),
            }
        )

    write_csv(OUTPUT_DIR / "summary.csv", rows)
    write_csv(OUTPUT_DIR / "polar_holdout.csv", polar_rows)

    metric = {str(row["method"]): row for row in rows}
    polar = {
        (str(row["method"]), str(row["polar_bin_deg"])): row
        for row in polar_rows
    }
    table = rf"""% Generated by scripts/generate_local_whitening_ablation.py.
\begin{{table}}[t]
\centering
\caption{{Local covariance whitening for the Spherical BA residual on Sequence A. All variants share the initialization, fixed backend input, and held-out observations.}}
\label{{tab:local_whitened_spherical}}
\footnotesize
\setlength{{\tabcolsep}}{{2.3pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
Residual & RMSE & P95 & Ang. RMSE & Inlier@1 \\
 & [px] $\downarrow$ & [px] $\downarrow$ & [mrad] $\downarrow$ & [\%] $\uparrow$ \\
\midrule
Pixel & {metric['Pixel']['holdout_rmse_px']:.4f} & {metric['Pixel']['holdout_p95_px']:.4f} & {metric['Pixel']['true_angular_rmse_mrad']:.4f} & {metric['Pixel']['inlier_at_1px_percent']:.2f} \\
Spherical & {metric['Spherical']['holdout_rmse_px']:.4f} & {metric['Spherical']['holdout_p95_px']:.4f} & {metric['Spherical']['true_angular_rmse_mrad']:.4f} & {metric['Spherical']['inlier_at_1px_percent']:.2f} \\
Local-whitened & {metric['Local-Whitened']['holdout_rmse_px']:.4f} & {metric['Local-Whitened']['holdout_p95_px']:.4f} & {metric['Local-Whitened']['true_angular_rmse_mrad']:.4f} & {metric['Local-Whitened']['inlier_at_1px_percent']:.2f} \\
\midrule
Residual & $0$--$30^\circ$ & $30$--$50^\circ$ & $\geq50^\circ$ & BA time \\
 & \multicolumn{{3}}{{c}}{{True angular RMSE [mrad] $\downarrow$}} & [s] $\downarrow$ \\
\midrule
Pixel & {polar[('Pixel', '0--30')]['true_angular_rmse_mrad']:.4f} & {polar[('Pixel', '30--50')]['true_angular_rmse_mrad']:.4f} & {polar[('Pixel', '50+')]['true_angular_rmse_mrad']:.4f} & {metric['Pixel']['ba_time_seconds']:.2f} \\
Spherical & {polar[('Spherical', '0--30')]['true_angular_rmse_mrad']:.4f} & {polar[('Spherical', '30--50')]['true_angular_rmse_mrad']:.4f} & {polar[('Spherical', '50+')]['true_angular_rmse_mrad']:.4f} & {metric['Spherical']['ba_time_seconds']:.2f} \\
Local-whitened & {polar[('Local-Whitened', '0--30')]['true_angular_rmse_mrad']:.4f} & {polar[('Local-Whitened', '30--50')]['true_angular_rmse_mrad']:.4f} & {polar[('Local-Whitened', '50+')]['true_angular_rmse_mrad']:.4f} & {metric['Local-Whitened']['ba_time_seconds']:.2f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (OUTPUT_DIR / "table_local_whitening_ablation.tex").write_text(
        table, encoding="utf-8"
    )
    protocol = "\n".join(
        [
            "dataset: stereo_dataset_20260430_1444190-clear/right",
            f"split_signature: {summaries['Pixel']['split_signature']}",
            f"shared_initial_intrinsics_csv: {initial_intrinsics['Pixel']}",
            "fixed_backend_frame_count: 26",
            "fixed_backend_board_observation_count: 130",
            "fixed_backend_point_count: 4330",
            f"holdout_point_count: {len(reference_keys)}",
            "initialization_equal: 1",
            "backend_input_equal: 1",
            "holdout_correspondences_equal: 1",
            "local_whitening_pixel_sigma_px: 1",
            "local_whitening_covariance_damping: 1e-12",
            "local_whitening_min_sigma_rad: 1e-6",
            "local_whitening_max_weight: 1e5",
            "",
        ]
    )
    (OUTPUT_DIR / "protocol_summary.txt").write_text(protocol, encoding="utf-8")


if __name__ == "__main__":
    main()

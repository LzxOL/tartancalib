#!/usr/bin/env python3
"""Generate the fixed-input Pixel vs Spherical close-calibration ablation."""

from __future__ import annotations

import argparse
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
METHODS = ("Pixel", "Spherical")
POLAR_BINS = ((0.0, 30.0, "0--30"), (30.0, 50.0, "30--50"),
              (50.0, math.inf, "50+"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pixel-result",
        default="result_may/stage5_angular_closecalib_seed1337_fixed_pixel",
    )
    parser.add_argument(
        "--spherical-result",
        default="result_may/stage5_angular_closecalib_seed1337_fixed_spherical",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/experiments/close_calibration_residual_ablation_seed1337",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result_dirs = {
        "Pixel": (ROOT / args.pixel_result).resolve(),
        "Spherical": (ROOT / args.spherical_result).resolve(),
    }
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    initialization = {
        method: read_summary(path / "auto_camera_initialization_summary.txt")
        for method, path in result_dirs.items()
    }
    selected_intrinsics = {
        method: initialization[method]["selected_intrinsics_csv"]
        for method in METHODS
    }
    if len(set(selected_intrinsics.values())) != 1:
        raise RuntimeError(f"Initialization mismatch: {selected_intrinsics}")

    backend_inputs = {
        method: (path / "backend_input_used_frame_board_list.csv").read_bytes()
        for method, path in result_dirs.items()
    }
    if backend_inputs["Pixel"] != backend_inputs["Spherical"]:
        raise RuntimeError("Fixed backend frame-board inputs differ")

    holdout_summaries = {
        method: read_summary(path / "backend_holdout_summary.txt")
        for method, path in result_dirs.items()
    }
    split_signatures = {
        holdout_summaries[method]["split_signature"] for method in METHODS
    }
    if len(split_signatures) != 1:
        raise RuntimeError(f"Split mismatch: {split_signatures}")

    selection_summaries = {
        method: read_summary(
            path / "trial_backend_frame_board_selection_summary.txt"
        )
        for method, path in result_dirs.items()
    }
    for method in METHODS:
        if int(selection_summaries[method]["persistent_incremental_candidate_batch_count"]) != 0:
            raise RuntimeError(f"{method} did not use an exact fixed backend input")

    reference_camera = tuple(
        float(value) for value in selected_intrinsics["Pixel"].split(",")
    )
    method_points = {
        method: load_holdout_points(path) for method, path in result_dirs.items()
    }
    reference_observations = {
        point_key(point): (float(point["observed_x"]), float(point["observed_y"]))
        for point in method_points["Pixel"]
    }
    reference_keys = set(reference_observations)

    metric_rows: list[dict[str, object]] = []
    polar_rows: list[dict[str, object]] = []
    for method in METHODS:
        points = method_points[method]
        if {point_key(point) for point in points} != reference_keys:
            raise RuntimeError(f"Holdout correspondence mismatch for {method}")
        camera = camera_parameters(holdout_summaries[method])
        pixel_errors: list[float] = []
        angular_errors: list[float] = []
        polar_accumulators = {
            label: {"count": 0, "pixel_sq": 0.0, "angular_sq": 0.0}
            for _, _, label in POLAR_BINS
        }
        for point in points:
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
                raise RuntimeError(f"Invalid DS ray for {method}: {key}")
            pixel_error = float(point["residual_norm"])
            angular_error = angular_error_rad(observed_ray, predicted_ray)
            pixel_errors.append(pixel_error)
            angular_errors.append(angular_error)
            polar = polar_angle_deg(reference_ray)
            for lower, upper, label in POLAR_BINS:
                if lower <= polar < upper:
                    accumulator = polar_accumulators[label]
                    accumulator["count"] += 1
                    accumulator["pixel_sq"] += pixel_error * pixel_error
                    accumulator["angular_sq"] += angular_error * angular_error
                    break

        for _, _, label in POLAR_BINS:
            accumulator = polar_accumulators[label]
            count = int(accumulator["count"])
            if count == 0:
                raise RuntimeError(f"Empty polar bin {label} for {method}")
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

        summary = holdout_summaries[method]
        selection = selection_summaries[method]
        metric_rows.append(
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
                "true_angular_p95_mrad": 1000.0 * percentile(angular_errors, 0.95),
                "pose_success_percent": 100.0
                * float(summary["pose_only_refit_success_rate"]),
                "holdout_point_count": len(points),
                "incremental_ba_time_seconds": float(
                    selection["persistent_incremental_total_elapsed_time_seconds"]
                ),
                "camera_xi": camera[0],
                "camera_alpha": camera[1],
                "camera_fu": camera[2],
                "camera_fv": camera[3],
                "camera_cu": camera[4],
                "camera_cv": camera[5],
                "source_result_dir": str(result_dirs[method].relative_to(ROOT)),
            }
        )

    write_csv(output_dir / "summary.csv", metric_rows)
    write_csv(output_dir / "polar_holdout.csv", polar_rows)

    metrics = {str(row["method"]): row for row in metric_rows}
    polar = {
        (str(row["method"]), str(row["polar_bin_deg"])): row
        for row in polar_rows
    }
    latex = rf"""% Generated by scripts/generate_close_calibration_residual_ablation.py.
\begin{{table}}[t]
\centering
\caption{{Fixed-input residual ablation when the close-distance sequence is used for calibration. Both variants share the split, initialization, backend frame-board inputs, and held-out observations.}}
\label{{tab:close_calibration_spherical_ablation}}
\footnotesize
\setlength{{\tabcolsep}}{{2.5pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
Residual & RMSE & P95 & Ang. RMSE & Inlier@1 \\
 & [px] $\downarrow$ & [px] $\downarrow$ & [mrad] $\downarrow$ & [\%] $\uparrow$ \\
\midrule
Pixel & {metrics['Pixel']['holdout_rmse_px']:.4f} & {metrics['Pixel']['holdout_p95_px']:.4f} & {metrics['Pixel']['true_angular_rmse_mrad']:.4f} & \textbf{{{metrics['Pixel']['inlier_at_1px_percent']:.2f}}} \\
Spherical & \textbf{{{metrics['Spherical']['holdout_rmse_px']:.4f}}} & \textbf{{{metrics['Spherical']['holdout_p95_px']:.4f}}} & \textbf{{{metrics['Spherical']['true_angular_rmse_mrad']:.4f}}} & {metrics['Spherical']['inlier_at_1px_percent']:.2f} \\
\midrule
\multicolumn{{5}}{{@{{}}l}}{{\textit{{Polar-angle-conditioned holdout error}}}} \\
Residual & $0$--$30^\circ$ & $30$--$50^\circ$ & $\geq50^\circ$ & BA time \\
 & \multicolumn{{3}}{{c}}{{True angular RMSE [mrad] $\downarrow$}} & [s] $\downarrow$ \\
\midrule
Pixel & {polar[('Pixel', '0--30')]['true_angular_rmse_mrad']:.4f} & {polar[('Pixel', '30--50')]['true_angular_rmse_mrad']:.4f} & {polar[('Pixel', '50+')]['true_angular_rmse_mrad']:.4f} & \textbf{{{metrics['Pixel']['incremental_ba_time_seconds']:.2f}}} \\
Spherical & \textbf{{{polar[('Spherical', '0--30')]['true_angular_rmse_mrad']:.4f}}} & \textbf{{{polar[('Spherical', '30--50')]['true_angular_rmse_mrad']:.4f}}} & \textbf{{{polar[('Spherical', '50+')]['true_angular_rmse_mrad']:.4f}}} & {metrics['Spherical']['incremental_ba_time_seconds']:.2f} \\
\bottomrule
\end{{tabular}}
\vspace{{2pt}}
\parbox{{\columnwidth}}{{\scriptsize \textit{{Protocol.}} The 19-frame close-distance sequence is split 70/30 with seed 1337. Polar bins are assigned using the shared initialization camera; angular errors use each method's final DS camera. The Pixel-selected 11-frame/55-board manifest is frozen for both optimizations.}}
\end{{table}}
"""
    (output_dir / "table_close_calibration_residual.tex").write_text(
        latex, encoding="utf-8"
    )

    protocol = "\n".join(
        [
            "protocol_name: close_calibration_fixed_backend_residual_ablation",
            f"split_signature: {next(iter(split_signatures))}",
            f"shared_initial_intrinsics_csv: {selected_intrinsics['Pixel']}",
            "fixed_backend_manifest_source: pixel_selected_backend.csv",
            "fixed_backend_frame_count: 11",
            "fixed_backend_board_observation_count: 55",
            "fixed_backend_point_count: 1830",
            f"holdout_point_count: {len(reference_keys)}",
            "pixel_and_spherical_initialization_equal: 1",
            "pixel_and_spherical_backend_input_equal: 1",
            "pixel_and_spherical_holdout_correspondences_equal: 1",
            "hybrid_included: 0",
            "",
        ]
    )
    (output_dir / "protocol_summary.txt").write_text(protocol, encoding="utf-8")


if __name__ == "__main__":
    main()

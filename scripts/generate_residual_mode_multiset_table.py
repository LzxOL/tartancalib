#!/usr/bin/env python3
"""Generate the single-column residual-mode table from Stage5 summaries."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "tables"

DATASETS = [
    ("1444190", "Sequence A"),
    ("134853", "Sequence B"),
    ("192347", "Sequence C"),
]

RESULT_DIRS = {
    ("1444190", "Pixel"): "result_may/stage5_residual_multiseed_20260712_seed4242_1444190clear_pixel",
    ("1444190", "Spherical"): "result_may/stage5_residual_multiseed_20260712_seed4242_1444190clear_spherical",
    ("1444190", "Hybrid"): "result_may/stage5_residual_multiseed_20260712_seed4242_1444190clear_hybrid",
    ("134853", "Pixel"): "result_may/stage5_residual_multiseed_20260712_seed4242_134853clear_pixel",
    ("134853", "Spherical"): "result_may/stage5_residual_multiseed_20260712_seed4242_134853clear_spherical",
    ("134853", "Hybrid"): "result_may/stage5_residual_multiseed_20260712_seed4242_134853clear_hybrid",
    ("192347", "Pixel"): "result_may/stage5_residual_multiseed_20260712_seed4242_192347clear_pixel",
    ("192347", "Spherical"): "result_may/stage5_residual_multiseed_20260712_seed4242_192347clear_spherical",
    ("192347", "Hybrid"): "result_may/stage5_residual_multiseed_20260712_seed4242_192347clear_hybrid",
}

METHODS = ["Pixel", "Spherical", "Hybrid"]
POLAR_BINS = [(0.0, 30.0), (30.0, 50.0), (50.0, math.inf)]
POLAR_LABELS = ["0--30", "30--50", "50+"]


def read_summary(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_key, dataset_label in DATASETS:
        for method in METHODS:
            result_dir = ROOT / RESULT_DIRS[(dataset_key, method)]
            holdout = read_summary(result_dir / "backend_holdout_summary.txt")
            selection = read_summary(
                result_dir / "trial_backend_frame_board_selection_summary.txt"
            )
            comparison = read_summary(result_dir / "backend_vs_kalibr_summary.txt")
            rows.append(
                {
                    "dataset": dataset_label,
                    "dataset_key": dataset_key,
                    "method": method,
                    "overall_rmse": float(holdout["overall_rmse"]),
                    "outer_rmse": float(holdout["outer_only_rmse"]),
                    "internal_rmse": float(holdout["internal_only_rmse"]),
                    "holdout_points": int(holdout["point_count"]),
                    "attempted_batches": int(
                        selection["persistent_incremental_attempted_batch_count"]
                    ),
                    "accepted_batches": int(
                        selection["persistent_incremental_accepted_batch_count"]
                    ),
                    "objective_decreased_batches": int(
                        selection[
                            "persistent_incremental_solver_objective_decreased_batch_count"
                        ]
                    ),
                    "guard_hits": int(
                        selection[
                            "persistent_incremental_solver_continuation_guard_hit_count"
                        ]
                    ),
                    "geometry_failures": int(
                        selection[
                            "persistent_incremental_angular_geometry_failure_count"
                        ]
                    ),
                    "incremental_ba_time_seconds": float(
                        selection["persistent_incremental_total_elapsed_time_seconds"]
                    ),
                    "kalibr_overall_rmse": float(
                        comparison["kalibr_holdout_overall_rmse"]
                    ),
                    "source_result_dir": str(result_dir.relative_to(ROOT)),
                }
            )
    return rows


def method_rows(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    return [row for row in rows if row["method"] == method]


def camera_parameters(summary: dict[str, str], prefix: str = "camera_") -> tuple[float, ...]:
    return tuple(
        float(summary[f"{prefix}{name}"])
        for name in ("xi", "alpha", "fu", "fv", "cu", "cv")
    )


def unproject_ds(
    image_x: float, image_y: float, parameters: tuple[float, ...]
) -> tuple[float, float, float] | None:
    xi, alpha, fu, fv, cu, cv = parameters
    mx = (image_x - cu) / fu
    my = (image_y - cv) / fv
    radius_squared = mx * mx + my * my
    sqrt_argument = 1.0 - (2.0 * alpha - 1.0) * radius_squared
    if sqrt_argument <= 0.0:
        return None
    mz = (1.0 - alpha * alpha * radius_squared) / (
        alpha * math.sqrt(sqrt_argument) + 1.0 - alpha
    )
    second_sqrt_argument = mz * mz + (1.0 - xi * xi) * radius_squared
    denominator = mz * mz + radius_squared
    if second_sqrt_argument <= 0.0 or denominator <= 0.0:
        return None
    scale = (mz * xi + math.sqrt(second_sqrt_argument)) / denominator
    ray = (scale * mx, scale * my, scale * mz - xi)
    norm = math.sqrt(sum(component * component for component in ray))
    if norm <= 0.0 or not math.isfinite(norm):
        return None
    return tuple(component / norm for component in ray)


def polar_angle_deg(ray: tuple[float, float, float]) -> float:
    return math.degrees(math.atan2(math.hypot(ray[0], ray[1]), ray[2]))


def angular_error_rad(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    cross_norm = math.sqrt(sum(component * component for component in cross))
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(first, second))))
    return math.atan2(cross_norm, dot)


def point_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["frame_index"],
        row["board_id"],
        row["point_type"],
        row["source_kind"],
        row["source_point_index"],
    )


def load_holdout_points(result_dir: Path) -> list[dict[str, str]]:
    with (result_dir / "backend_holdout_points.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        return list(csv.DictReader(stream))


def compute_polar_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result_lookup = {
        (str(row["dataset_key"]), str(row["method"])): ROOT
        / str(row["source_result_dir"])
        for row in rows
    }
    polar_rows: list[dict[str, object]] = []
    for dataset_key, dataset_label in DATASETS:
        initialization_summary = read_summary(
            result_lookup[(dataset_key, "Pixel")]
            / "auto_camera_initialization_summary.txt"
        )
        reference_camera = tuple(
            float(value)
            for value in initialization_summary["selected_intrinsics_csv"].split(",")
        )
        for method in METHODS[1:]:
            method_initialization = read_summary(
                result_lookup[(dataset_key, method)]
                / "auto_camera_initialization_summary.txt"
            )
            method_reference = tuple(
                float(value)
                for value in method_initialization["selected_intrinsics_csv"].split(",")
            )
            if method_reference != reference_camera:
                raise RuntimeError(
                    f"Initialization mismatch for {dataset_label} / {method}"
                )
        reference_points = load_holdout_points(
            result_lookup[(dataset_key, "Pixel")]
        )
        reference_observations = {
            point_key(point): (float(point["observed_x"]), float(point["observed_y"]))
            for point in reference_points
        }

        for method in METHODS:
            result_dir = result_lookup[(dataset_key, method)]
            holdout_summary = read_summary(result_dir / "backend_holdout_summary.txt")
            method_camera = camera_parameters(holdout_summary)
            method_points = load_holdout_points(result_dir)
            if {point_key(point) for point in method_points} != set(reference_observations):
                raise RuntimeError(
                    f"Holdout correspondence mismatch for {dataset_label} / {method}"
                )

            accumulators = [
                {"count": 0, "pixel_squared": 0.0, "angular_squared": 0.0}
                for _ in POLAR_BINS
            ]
            for point in method_points:
                key = point_key(point)
                observed_x = float(point["observed_x"])
                observed_y = float(point["observed_y"])
                reference_x, reference_y = reference_observations[key]
                if abs(observed_x - reference_x) > 1e-6 or abs(observed_y - reference_y) > 1e-6:
                    raise RuntimeError(
                        f"Observed pixel mismatch for {dataset_label} / {method} / {key}"
                    )
                reference_ray = unproject_ds(
                    observed_x, observed_y, reference_camera
                )
                observed_ray = unproject_ds(observed_x, observed_y, method_camera)
                predicted_ray = unproject_ds(
                    float(point["predicted_x"]),
                    float(point["predicted_y"]),
                    method_camera,
                )
                if reference_ray is None or observed_ray is None or predicted_ray is None:
                    raise RuntimeError(
                        f"Invalid DS unprojection for {dataset_label} / {method} / {key}"
                    )
                angle = polar_angle_deg(reference_ray)
                bin_index = next(
                    (
                        index
                        for index, (lower, upper) in enumerate(POLAR_BINS)
                        if lower <= angle < upper
                    ),
                    None,
                )
                if bin_index is None:
                    raise RuntimeError(f"Polar angle outside bins: {angle}")
                accumulator = accumulators[bin_index]
                pixel_error = float(point["residual_norm"])
                angular_error = angular_error_rad(observed_ray, predicted_ray)
                accumulator["count"] += 1
                accumulator["pixel_squared"] += pixel_error * pixel_error
                accumulator["angular_squared"] += angular_error * angular_error

            for label, accumulator in zip(POLAR_LABELS, accumulators):
                count = int(accumulator["count"])
                if count == 0:
                    raise RuntimeError(
                        f"Empty polar bin {label} for {dataset_label} / {method}"
                    )
                polar_rows.append(
                    {
                        "dataset": dataset_label,
                        "dataset_key": dataset_key,
                        "method": method,
                        "polar_bin_deg": label,
                        "point_count": count,
                        "pixel_rmse": math.sqrt(
                            float(accumulator["pixel_squared"]) / count
                        ),
                        "angular_rmse_mrad": 1000.0
                        * math.sqrt(float(accumulator["angular_squared"]) / count),
                        "bin_assignment_camera": "shared_stage5_initial_camera",
                        "source_result_dir": str(result_dir.relative_to(ROOT)),
                    }
                )
    return polar_rows


def fmt(value: float) -> str:
    return f"{value:.4f}"


def best_method_by_dataset(rows: list[dict[str, object]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for dataset_key, _ in DATASETS:
        candidates = [row for row in rows if row["dataset_key"] == dataset_key]
        result[dataset_key] = str(min(candidates, key=lambda row: row["overall_rmse"])["method"])
    return result


def maybe_bold(value: float, is_best: bool) -> str:
    rendered = fmt(value)
    return f"\\textbf{{{rendered}}}" if is_best else rendered


def write_csv(rows: list[dict[str, object]]) -> None:
    output = OUTPUT_DIR / "table_residual_modes_multiset_data.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_polar_csv(rows: list[dict[str, object]]) -> None:
    output = OUTPUT_DIR / "table_residual_modes_polar_data.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_ray_consistency(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result_lookup = {
        (str(row["dataset_key"]), str(row["method"])): ROOT
        / str(row["source_result_dir"])
        for row in rows
    }
    sample_file = (
        result_lookup[(DATASETS[0][0], "Pixel")] / "camera_ray_curve_samples.csv"
    )
    with sample_file.open(newline="", encoding="utf-8") as stream:
        sample_positions = sorted(
            {
                (float(row["image_x"]), float(row["image_y"]))
                for row in csv.DictReader(stream)
            }
        )

    consistency_rows: list[dict[str, object]] = []
    for method in METHODS:
        cameras = []
        for dataset_key, _ in DATASETS:
            summary = read_summary(
                result_lookup[(dataset_key, method)] / "backend_holdout_summary.txt"
            )
            cameras.append(camera_parameters(summary))

        pairwise_errors_deg = []
        valid_sample_count = 0
        for image_x, image_y in sample_positions:
            rays = [unproject_ds(image_x, image_y, camera) for camera in cameras]
            if any(ray is None for ray in rays):
                continue
            valid_sample_count += 1
            for first_index, second_index in ((0, 1), (0, 2), (1, 2)):
                pairwise_errors_deg.append(
                    math.degrees(
                        angular_error_rad(rays[first_index], rays[second_index])
                    )
                )
        if not pairwise_errors_deg:
            raise RuntimeError(f"No valid ray consistency samples for {method}")
        consistency_rows.append(
            {
                "method": method,
                "sequence_count": len(DATASETS),
                "sequence_pair_count": 3,
                "valid_image_sample_count": valid_sample_count,
                "pairwise_ray_count": len(pairwise_errors_deg),
                "pairwise_mean_deg": fmean(pairwise_errors_deg),
                "pairwise_rms_deg": math.sqrt(
                    fmean(error * error for error in pairwise_errors_deg)
                ),
                "sample_grid_source": str(sample_file.relative_to(ROOT)),
            }
        )
    return consistency_rows


def write_ray_consistency_csv(rows: list[dict[str, object]]) -> None:
    output = OUTPUT_DIR / "table_residual_modes_ray_consistency_data.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_table(
    rows: list[dict[str, object]],
    polar_rows: list[dict[str, object]],
    ray_consistency_rows: list[dict[str, object]],
) -> None:
    lookup = {
        (str(row["dataset_key"]), str(row["method"])): row for row in rows
    }
    dataset_best = best_method_by_dataset(rows)
    method_average = {
        method: fmean(float(row["overall_rmse"]) for row in method_rows(rows, method))
        for method in METHODS
    }
    average_best = min(method_average, key=method_average.get)
    kalibr_by_dataset = {
        dataset_key: float(lookup[(dataset_key, "Pixel")]["kalibr_overall_rmse"])
        for dataset_key, _ in DATASETS
    }

    lines = [
        "% Generated by scripts/generate_residual_mode_multiset_table.py.",
        "% Values are read from Stage5 result summaries; do not edit manually.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Residual-objective ablation under the Double Sphere model. All modes share the split, initialization, and frozen holdout observations within each sequence. Aggregates are sequence-level macro-averages over three independent captures; best results are in bold.}",
        "\\label{tab:residual_mode_multiset}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.3pt}",
        "\\renewcommand{\\arraystretch}{1.03}",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "\\multicolumn{5}{@{}l}{\\textit{(a) Holdout image-plane accuracy}} \\\\",
        "Residual & Seq. A & Seq. B & Seq. C & Mean \\\\",
        " & \\multicolumn{4}{c}{Reprojection RMSE [px] $\\downarrow$} \\\\",
        "\\midrule",
    ]

    for method in METHODS:
        values = []
        for dataset_key, _ in DATASETS:
            value = float(lookup[(dataset_key, method)]["overall_rmse"])
            values.append(maybe_bold(value, dataset_best[dataset_key] == method))
        average = maybe_bold(method_average[method], average_best == method)
        lines.append(f"{method} & {' & '.join(values)} & {average} \\\\")

    lines.extend(
        [
            "\\addlinespace[1pt]",
            "Kalibr (DS) & "
            + " & ".join(fmt(kalibr_by_dataset[key]) for key, _ in DATASETS)
            + f" & {fmt(fmean(kalibr_by_dataset.values()))} \\\\",
            "\\midrule",
            "\\multicolumn{5}{@{}l}{\\textit{(b) View-angle-conditioned ray accuracy}} \\\\",
            "Residual & $0$--$30^{\\circ}$ & $30$--$50^{\\circ}$ & $\\geq 50^{\\circ}$ & Overall \\\\",
            " & \\multicolumn{4}{c}{True angular RMSE [mrad] $\\downarrow$} \\\\",
            "\\midrule",
        ]
    )

    polar_lookup = {
        (str(row["method"]), str(row["polar_bin_deg"])): []
        for row in polar_rows
    }
    for row in polar_rows:
        polar_lookup[(str(row["method"]), str(row["polar_bin_deg"]))].append(
            float(row["angular_rmse_mrad"])
        )
    polar_average = {
        key: fmean(values) for key, values in polar_lookup.items()
    }
    polar_best = {
        label: min(METHODS, key=lambda method: polar_average[(method, label)])
        for label in POLAR_LABELS
    }
    angular_overall_by_method: dict[str, float] = {}
    for method in METHODS:
        dataset_angular_overall = []
        for dataset_key, _ in DATASETS:
            dataset_rows = [
                row
                for row in polar_rows
                if row["method"] == method and row["dataset_key"] == dataset_key
            ]
            count = sum(int(row["point_count"]) for row in dataset_rows)
            squared_sum = sum(
                int(row["point_count"])
                * float(row["angular_rmse_mrad"]) ** 2
                for row in dataset_rows
            )
            dataset_angular_overall.append(math.sqrt(squared_sum / count))
        angular_overall_by_method[method] = fmean(dataset_angular_overall)
    angular_overall_best = min(angular_overall_by_method, key=angular_overall_by_method.get)

    for method in METHODS:
        polar_values = [
            maybe_bold(
                polar_average[(method, label)], polar_best[label] == method
            )
            for label in POLAR_LABELS
        ]
        overall = maybe_bold(
            angular_overall_by_method[method], angular_overall_best == method
        )
        lines.append(f"{method} & {' & '.join(polar_values)} & {overall} \\\\")

    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{5}{@{}l}{\\textit{(c) Cross-sequence geometric stability and cost}} \\\\",
            "Residual & Overall & Outer & Cross-seq. & BA time \\\\",
            " & RMSE [px] & RMSE [px] & ray RMS [deg] & [s] \\\\",
            "\\midrule",
        ]
    )

    metric_best = {
        metric: min(
            METHODS,
            key=lambda method: fmean(
                float(row[metric]) for row in method_rows(rows, method)
            ),
        )
        for metric in ("overall_rmse", "outer_rmse")
    }
    ray_consistency = {
        str(row["method"]): float(row["pairwise_rms_deg"])
        for row in ray_consistency_rows
    }
    ray_best = min(ray_consistency, key=ray_consistency.get)
    time_best = min(
        METHODS,
        key=lambda method: fmean(
            float(row["incremental_ba_time_seconds"])
            for row in method_rows(rows, method)
        ),
    )
    for method in METHODS:
        metrics = []
        for metric in ("overall_rmse", "outer_rmse"):
            value = fmean(float(row[metric]) for row in method_rows(rows, method))
            metrics.append(maybe_bold(value, metric_best[metric] == method))
        metrics.append(
            maybe_bold(ray_consistency[method], ray_best == method)
        )
        mean_time = fmean(
            float(row["incremental_ba_time_seconds"])
            for row in method_rows(rows, method)
        )
        rendered_time = f"{mean_time:.1f}"
        if time_best == method:
            rendered_time = f"\\textbf{{{rendered_time}}}"
        lines.append(f"{method} & {' & '.join(metrics)} & {rendered_time} \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{2pt}",
            "\\parbox{\\columnwidth}{\\footnotesize \\textit{Protocol.} Polar bins are assigned once using the shared initialization camera. Cross-seq. ray RMS is the pairwise ray-curve disagreement over 1,422 shared image locations (4,266 ray pairs). BA time is the mean incremental-optimization runtime. All metrics are lower-is-better.}",
            "\\end{table}",
            "",
        ]
    )
    (OUTPUT_DIR / "table_residual_modes_multiset.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    polar_rows = compute_polar_rows(rows)
    ray_consistency_rows = compute_ray_consistency(rows)
    write_csv(rows)
    write_polar_csv(polar_rows)
    write_ray_consistency_csv(ray_consistency_rows)
    write_table(rows, polar_rows, ray_consistency_rows)


if __name__ == "__main__":
    main()

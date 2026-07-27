#!/usr/bin/env python3
"""Generate the fixed-input close-distance residual ablation table."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import fmean

import generate_residual_mode_multiset_table as common


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT
    / "paper"
    / "experiments"
    / "fixed_backend_residual_ablation_seed1337"
    / "close_distance_test"
)
DATASETS = [("1444190", "Train A"), ("134853", "Train B")]
METHODS = ["Pixel", "Spherical", "Hybrid"]
SUFFIXES = {"Pixel": "pixel", "Spherical": "spherical", "Hybrid": "hybrid"}


def result_dir(dataset: str, method: str) -> Path:
    return (
        ROOT
        / "result_may"
        / f"stage5_close_test_fixed_seed1337_{dataset}train_{SUFFIXES[method]}"
    )


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * probability)]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, label in DATASETS:
        for method in METHODS:
            directory = result_dir(dataset, method)
            summary = common.read_summary(directory / "backend_holdout_summary.txt")
            selection = common.read_summary(
                directory / "trial_backend_frame_board_selection_summary.txt"
            )
            points = common.load_holdout_points(directory)
            residuals = [float(point["residual_norm"]) for point in points]
            rows.append(
                {
                    "dataset": label,
                    "dataset_key": dataset,
                    "method": method,
                    "overall_rmse": float(summary["overall_rmse"]),
                    "outer_rmse": float(summary["outer_only_rmse"]),
                    "internal_rmse": float(summary["internal_only_rmse"]),
                    "p95_px": quantile(residuals, 0.95),
                    "p99_px": quantile(residuals, 0.99),
                    "inlier_1px_percent": 100.0
                    * sum(value <= 1.0 for value in residuals)
                    / len(residuals),
                    "point_count": len(points),
                    "ba_time_seconds": float(
                        selection["persistent_incremental_total_elapsed_time_seconds"]
                    ),
                    "camera": common.camera_parameters(summary),
                    "source_result_dir": str(directory.relative_to(ROOT)),
                }
            )
    return rows


def compute_polar_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    old_datasets = common.DATASETS
    old_result_dirs = common.RESULT_DIRS
    try:
        common.DATASETS = DATASETS
        common.RESULT_DIRS = {
            (dataset, method): str(result_dir(dataset, method).relative_to(ROOT))
            for dataset, _ in DATASETS
            for method in METHODS
        }
        compatibility_rows = [
            {
                "dataset_key": row["dataset_key"],
                "method": row["method"],
                "source_result_dir": row["source_result_dir"],
            }
            for row in rows
        ]
        return common.compute_polar_rows(compatibility_rows)
    finally:
        common.DATASETS = old_datasets
        common.RESULT_DIRS = old_result_dirs


def compute_cross_train_ray_rms(rows: list[dict[str, object]]) -> dict[str, float]:
    sample_file = result_dir(DATASETS[0][0], "Pixel") / "camera_ray_curve_samples.csv"
    with sample_file.open(newline="", encoding="utf-8") as stream:
        positions = sorted(
            {
                (float(row["image_x"]), float(row["image_y"]))
                for row in csv.DictReader(stream)
            }
        )
    result: dict[str, float] = {}
    for method in METHODS:
        cameras = [
            next(
                row["camera"]
                for row in rows
                if row["dataset_key"] == dataset and row["method"] == method
            )
            for dataset, _ in DATASETS
        ]
        errors = []
        for x, y in positions:
            first = common.unproject_ds(x, y, cameras[0])
            second = common.unproject_ds(x, y, cameras[1])
            if first is not None and second is not None:
                errors.append(math.degrees(common.angular_error_rad(first, second)))
        result[method] = math.sqrt(fmean(error * error for error in errors))
    return result


def macro(rows: list[dict[str, object]], method: str, key: str) -> float:
    return fmean(float(row[key]) for row in rows if row["method"] == method)


def polar_macro(
    rows: list[dict[str, object]], method: str, polar_bin: str
) -> float:
    return fmean(
        float(row["angular_rmse_mrad"])
        for row in rows
        if row["method"] == method and row["polar_bin_deg"] == polar_bin
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    serializable = []
    for row in rows:
        output = dict(row)
        output.pop("camera", None)
        serializable.append(output)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)


def bold(value: float, is_best: bool, digits: int = 4) -> str:
    rendered = f"{value:.{digits}f}"
    return f"\\textbf{{{rendered}}}" if is_best else rendered


def write_latex(
    rows: list[dict[str, object]],
    polar_rows: list[dict[str, object]],
    ray_rms: dict[str, float],
) -> None:
    row_end = r" \\"
    overall = {method: macro(rows, method, "overall_rmse") for method in METHODS}
    outer = {method: macro(rows, method, "outer_rmse") for method in METHODS}
    p95 = {method: macro(rows, method, "p95_px") for method in METHODS}
    inlier = {method: macro(rows, method, "inlier_1px_percent") for method in METHODS}
    times = {method: macro(rows, method, "ba_time_seconds") for method in METHODS}
    polar = {
        (method, bucket): polar_macro(polar_rows, method, bucket)
        for method in METHODS
        for bucket in common.POLAR_LABELS
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Spherical bundle-adjustment ablation on the close-distance frozen test set. All variants use identical training frames, fixed backend inputs, and 3,150 test points.}",
        "\\label{tab:spherical_ba_ablation}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "\\multicolumn{5}{@{}l}{\\textit{(a) Image-plane accuracy}}" + row_end,
        "Residual & Train A & Train B & Mean & P95" + row_end,
        " & \\multicolumn{4}{c}{Reprojection error [px] $\\downarrow$}" + row_end,
        "\\midrule",
    ]
    for method in METHODS:
        values = [
            next(float(row["overall_rmse"]) for row in rows if row["dataset_key"] == dataset and row["method"] == method)
            for dataset, _ in DATASETS
        ]
        fields = [
            bold(value, value == min(next(float(row["overall_rmse"]) for row in rows if row["dataset_key"] == dataset and row["method"] == candidate) for candidate in METHODS))
            for value, (dataset, _) in zip(values, DATASETS)
        ]
        fields += [
            bold(overall[method], overall[method] == min(overall.values())),
            bold(p95[method], p95[method] == min(p95.values())),
        ]
        lines.append(f"{method} & {' & '.join(fields)}" + row_end)
    lines += [
        "\\midrule",
        "\\multicolumn{5}{@{}l}{\\textit{(b) View-angle accuracy}}" + row_end,
        "Residual & $0$--$30^{\\circ}$ & $30$--$50^{\\circ}$ & $\\geq50^{\\circ}$ & Ray RMS" + row_end,
        " & \\multicolumn{3}{c}{True angular RMSE [mrad] $\\downarrow$} & [deg] $\\downarrow$" + row_end,
        "\\midrule",
    ]
    for method in METHODS:
        fields = []
        for bucket in common.POLAR_LABELS:
            value = polar[(method, bucket)]
            best = min(polar[(candidate, bucket)] for candidate in METHODS)
            fields.append(bold(value, value == best))
        fields.append(bold(ray_rms[method], ray_rms[method] == min(ray_rms.values())))
        lines.append(f"{method} & {' & '.join(fields)}" + row_end)
    lines += [
        "\\midrule",
        "\\multicolumn{5}{@{}l}{\\textit{(c) Robustness and cost}}" + row_end,
        "Residual & Overall & Outer & Inlier@1 & BA time" + row_end,
        " & RMSE [px] & RMSE [px] & [\\%] $\\uparrow$ & [s] $\\downarrow$" + row_end,
        "\\midrule",
    ]
    for method in METHODS:
        fields = [
            bold(overall[method], overall[method] == min(overall.values())),
            bold(outer[method], outer[method] == min(outer.values())),
            bold(
                inlier[method],
                round(inlier[method], 2)
                == max(round(value, 2) for value in inlier.values()),
                2,
            ),
            bold(times[method], times[method] == min(times.values()), 1),
        ]
        lines.append(f"{method} & {' & '.join(fields)}" + row_end)
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{2pt}",
        "\\parbox{\\columnwidth}{\\scriptsize \\textit{Protocol.} Results are macro-averaged over two disjoint training captures. The close-distance test contains no training frame. Polar bins use the shared initialization camera; angular errors use each method's final camera. Ray RMS measures cross-capture ray-curve disagreement over 1,422 image locations.}",
        "\\end{table}",
        "",
    ]
    (OUTPUT_DIR / "table_close_distance_residual.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    polar_rows = compute_polar_rows(rows)
    ray_rms = compute_cross_train_ray_rms(rows)
    write_csv(OUTPUT_DIR / "close_distance_summary.csv", rows)
    write_csv(OUTPUT_DIR / "close_distance_polar.csv", polar_rows)
    with (OUTPUT_DIR / "close_distance_ray_rms.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "cross_train_ray_rms_deg"))
        writer.writerows(ray_rms.items())
    write_latex(rows, polar_rows, ray_rms)


if __name__ == "__main__":
    main()

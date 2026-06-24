#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SummaryMap = Dict[str, str]


def parse_key_value_file(path: Path) -> SummaryMap:
    data: SummaryMap = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def read_polar_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> Optional[float]:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str) -> Optional[int]:
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def fmt_float(value: Optional[float], digits: int = 5) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.{digits}f}"


def normalize_dataset_name(path: Path) -> str:
    name = path.name
    suffixes = [
        "_polar_diagnostic_only",
        "_ds_baseline_refit_diag",
        "_strict_baseline",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def extract_core_metrics(directory: Path) -> Dict[str, object]:
    backend_opt = parse_key_value_file(directory / "backend_optimization_summary.txt")
    backend_train = parse_key_value_file(directory / "backend_training_summary.txt")
    backend_holdout = parse_key_value_file(directory / "backend_holdout_summary.txt")
    experiment = parse_key_value_file(directory / "experiment_config_summary.txt")
    stage5_bundle = parse_key_value_file(directory / "stage5_bundle_summary.txt")
    return {
        "optimized_backend_rmse": to_float(backend_opt.get("optimized_overall_rmse", "")),
        "training_rmse": to_float(backend_train.get("overall_rmse", "")),
        "holdout_rmse": to_float(backend_holdout.get("overall_rmse", "")),
        "optimized_intrinsics": ",".join(
            filter(
                None,
                [
                    backend_opt.get("optimized_camera_xi", ""),
                    backend_opt.get("optimized_camera_alpha", ""),
                    backend_opt.get("optimized_camera_fu", ""),
                    backend_opt.get("optimized_camera_fv", ""),
                    backend_opt.get("optimized_camera_cu", ""),
                    backend_opt.get("optimized_camera_cv", ""),
                ],
            )
        ),
        "bundle_intrinsics_csv": stage5_bundle.get("camera_intrinsics_csv", ""),
        "selected_frame_count": to_int(experiment.get("round2_selected_frame_count", "")),
        "selected_board_observation_count": to_int(
            experiment.get("round2_selected_board_observation_count", "")
        ),
        "selected_internal_point_count": to_int(
            experiment.get("round2_selected_internal_point_count", "")
        ),
    }


def metrics_equivalent(lhs: Dict[str, object], rhs: Dict[str, object], tol: float = 1e-9) -> Tuple[bool, List[str]]:
    mismatches: List[str] = []
    float_keys = ["optimized_backend_rmse", "training_rmse", "holdout_rmse"]
    exact_keys = [
        "optimized_intrinsics",
        "bundle_intrinsics_csv",
        "selected_frame_count",
        "selected_board_observation_count",
        "selected_internal_point_count",
    ]
    for key in float_keys:
        a = lhs.get(key)
        b = rhs.get(key)
        if a is None or b is None:
            if a != b:
                mismatches.append(key)
            continue
        if not isinstance(a, float) or not isinstance(b, float) or abs(a - b) > tol:
            mismatches.append(key)
    for key in exact_keys:
        if lhs.get(key) != rhs.get(key):
            mismatches.append(key)
    return len(mismatches) == 0, mismatches


def summarize_point_type(rows: List[Dict[str, str]], point_type: str) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for row in rows:
      if row.get("point_type") != point_type:
        continue
      result.append(
          {
              "bin_min_deg": to_float(row.get("bin_min_deg", "")),
              "bin_max_deg": to_float(row.get("bin_max_deg", "")),
              "point_count": to_int(row.get("point_count", "")),
              "rmse": to_float(row.get("rmse", "")),
              "std_x": to_float(row.get("std_x", "")),
              "std_y": to_float(row.get("std_y", "")),
              "median_residual": to_float(row.get("median_residual", "")),
              "p90_residual": to_float(row.get("p90_residual", "")),
              "p95_residual": to_float(row.get("p95_residual", "")),
              "max_residual": to_float(row.get("max_residual", "")),
          }
      )
    result.sort(key=lambda item: (item["bin_min_deg"] or -1.0))
    return result


def analyze_internal_bins(rows: List[Dict[str, object]]) -> List[str]:
    findings: List[str] = []
    valid = [row for row in rows if row.get("point_count") and row.get("point_count", 0) > 0]
    if len(valid) < 2:
        return ["internal bins insufficient for trend analysis"]

    first = valid[0]
    last = valid[-1]
    if (last.get("point_count") or 0) < (first.get("point_count") or 0):
        findings.append("high polar angle internal point count decreases")
    if (
        isinstance(first.get("rmse"), float)
        and isinstance(last.get("rmse"), float)
        and last["rmse"] > first["rmse"]
    ):
        findings.append("high polar angle internal RMSE increases")
    if (
        isinstance(first.get("p90_residual"), float)
        and isinstance(last.get("p90_residual"), float)
        and last["p90_residual"] > first["p90_residual"]
    ):
        findings.append("high polar angle internal p90 increases")
    if (
        isinstance(first.get("p95_residual"), float)
        and isinstance(last.get("p95_residual"), float)
        and last["p95_residual"] > first["p95_residual"]
    ):
        findings.append("high polar angle internal p95 increases")

    for row in valid:
        sx = row.get("std_x")
        sy = row.get("std_y")
        if isinstance(sx, float) and isinstance(sy, float) and sy > 1e-9:
            ratio = max(sx, sy) / max(min(sx, sy), 1e-9)
            if ratio >= 1.5 and (row.get("bin_min_deg") or 0.0) >= 70.0:
                findings.append(
                    f"std imbalance at {fmt_float(row.get('bin_min_deg'), 1)}-{fmt_float(row.get('bin_max_deg'), 1)} deg"
                )
    if not findings:
        findings.append("no strong large-polar-angle degradation trend detected")
    return findings


def write_report(
    output_path: Path,
    rows: List[Tuple[str, Path, Path]],
) -> None:
    lines: List[str] = []
    lines.append("================================================================================")
    lines.append("Stage5 Phase 1 Polar-Angle Diagnostics Report")
    lines.append("================================================================================")
    lines.append("")
    phase2_recommended = False

    for dataset_name, baseline_dir, polar_dir in rows:
        baseline_metrics = extract_core_metrics(baseline_dir)
        polar_metrics = extract_core_metrics(polar_dir)
        equivalent, mismatches = metrics_equivalent(baseline_metrics, polar_metrics)
        polar_summary_exists = (polar_dir / "polar_angle_residual_summary.txt").exists()
        polar_csv_exists = (polar_dir / "polar_angle_residual_bins.csv").exists()
        polar_csv_rows = read_polar_csv(polar_dir / "polar_angle_residual_bins.csv")
        all_bins = summarize_point_type(polar_csv_rows, "all")
        outer_bins = summarize_point_type(polar_csv_rows, "outer")
        internal_bins = summarize_point_type(polar_csv_rows, "internal")
        internal_findings = analyze_internal_bins(internal_bins)

        if any("increases" in finding or "imbalance" in finding for finding in internal_findings):
            phase2_recommended = True

        lines.append("--------------------------------------------------------------------------------")
        lines.append(f"Dataset: {dataset_name}")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(f"baseline_dir: {baseline_dir}")
        lines.append(f"polar_diagnostic_dir: {polar_dir}")
        lines.append(
            f"diagnostic_only_changes_baseline: {'NO' if equivalent else 'YES'}"
        )
        if mismatches:
            lines.append(f"mismatched_fields: {', '.join(mismatches)}")
        lines.append(
            f"polar_angle_residual_summary_exists: {1 if polar_summary_exists else 0}"
        )
        lines.append(
            f"polar_angle_residual_bins_csv_exists: {1 if polar_csv_exists else 0}"
        )
        lines.append("")
        lines.append("Core Result Equivalence Check:")
        lines.append(
            f"  optimized_backend_rmse: baseline={fmt_float(baseline_metrics['optimized_backend_rmse'])} "
            f"polar={fmt_float(polar_metrics['optimized_backend_rmse'])}"
        )
        lines.append(
            f"  training_rmse: baseline={fmt_float(baseline_metrics['training_rmse'])} "
            f"polar={fmt_float(polar_metrics['training_rmse'])}"
        )
        lines.append(
            f"  holdout_rmse: baseline={fmt_float(baseline_metrics['holdout_rmse'])} "
            f"polar={fmt_float(polar_metrics['holdout_rmse'])}"
        )
        lines.append(
            f"  optimized_intrinsics: baseline={baseline_metrics['optimized_intrinsics']} "
            f"polar={polar_metrics['optimized_intrinsics']}"
        )
        lines.append(
            f"  selected_frame_count: baseline={baseline_metrics['selected_frame_count']} "
            f"polar={polar_metrics['selected_frame_count']}"
        )
        lines.append(
            f"  selected_board_observation_count: baseline={baseline_metrics['selected_board_observation_count']} "
            f"polar={polar_metrics['selected_board_observation_count']}"
        )
        lines.append(
            f"  selected_internal_point_count: baseline={baseline_metrics['selected_internal_point_count']} "
            f"polar={polar_metrics['selected_internal_point_count']}"
        )
        lines.append("")

        def append_bin_table(title: str, bin_rows: List[Dict[str, object]]) -> None:
            lines.append(title)
            lines.append(
                "  bin_min_deg | bin_max_deg | point_count | rmse | std_x | std_y | median | p90 | p95 | max"
            )
            for row in bin_rows:
                lines.append(
                    "  {minv} | {maxv} | {count} | {rmse} | {stdx} | {stdy} | {median} | {p90} | {p95} | {maxr}".format(
                        minv=fmt_float(row.get("bin_min_deg"), 1),
                        maxv=fmt_float(row.get("bin_max_deg"), 1),
                        count=row.get("point_count", ""),
                        rmse=fmt_float(row.get("rmse"), 5),
                        stdx=fmt_float(row.get("std_x"), 5),
                        stdy=fmt_float(row.get("std_y"), 5),
                        median=fmt_float(row.get("median_residual"), 5),
                        p90=fmt_float(row.get("p90_residual"), 5),
                        p95=fmt_float(row.get("p95_residual"), 5),
                        maxr=fmt_float(row.get("max_residual"), 5),
                    )
                )
            lines.append("")

        append_bin_table("All points polar-bin summary:", all_bins)
        append_bin_table("Outer only polar-bin summary:", outer_bins)
        append_bin_table("Internal only polar-bin summary:", internal_bins)

        lines.append("Internal large-polar-angle analysis:")
        for finding in internal_findings:
            lines.append(f"  - {finding}")
        lines.append("")

    lines.append("================================================================================")
    lines.append("Phase 1 Verdict")
    lines.append("================================================================================")
    lines.append(
        "diagnostic_only_preserves_baseline: "
        + ("YES" if all(metrics_equivalent(extract_core_metrics(b), extract_core_metrics(p))[0] for _, b, p in rows) else "NO")
    )
    lines.append(
        "phase2_polar_angle_adaptive_weighting_recommended: "
        + ("YES" if phase2_recommended else "NO")
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Stage5 Phase 1 polar-angle diagnostics validation report."
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("BASELINE_DIR", "POLAR_DIR"),
        required=True,
        help="Pair of baseline dir and polar-diagnostic-only dir.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output report path.",
    )
    args = parser.parse_args()

    rows: List[Tuple[str, Path, Path]] = []
    for baseline_str, polar_str in args.pair:
        baseline = Path(baseline_str).resolve()
        polar = Path(polar_str).resolve()
        dataset_name = normalize_dataset_name(baseline)
        rows.append((dataset_name, baseline, polar))

    write_report(Path(args.output).resolve(), rows)
    print(f"report={Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

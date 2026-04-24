#!/usr/bin/env python3
"""Summarize multiple Stage5 backend experiment directories into one table.

Examples:
  python3 tools/summarize_stage5_backend_experiments.py \
      --root result \
      --prefix stage5_backend_first20 \
      --output-dir result/stage5_backend_first20_summary

  python3 tools/summarize_stage5_backend_experiments.py \
      result/stage5_backend_first20_auto \
      result/stage5_backend_first20_ablation_no_round2
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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
    if value is None:
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_delta(value: Optional[float], digits: int = 5) -> str:
    if value is None:
        return ""
    return f"{value:+.{digits}f}"


def fmt_int(value: Optional[int]) -> str:
    return "" if value is None else str(value)


def normalize_name(name: str) -> str:
    if name.startswith("stage5_backend_"):
        return name[len("stage5_backend_") :]
    return name


def collect_directories(args: argparse.Namespace) -> List[Path]:
    directories: List[Path] = []
    if args.directories:
        directories = [Path(entry).resolve() for entry in args.directories]
    else:
        root = Path(args.root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Root does not exist: {root}")
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if args.prefix and not child.name.startswith(args.prefix):
                continue
            if (child / "backend_vs_frontend_summary.txt").exists():
                directories.append(child.resolve())
    return directories


def infer_experiment_label(directory: Path, experiment: SummaryMap, protocol: SummaryMap) -> str:
    if experiment.get("experiment_tag"):
        return experiment["experiment_tag"]
    protocol_label = (
        experiment.get("effective_protocol_label")
        or protocol.get("baseline_protocol_label")
        or directory.name
    )
    if protocol_label == "stage5_backend_auto_v1":
        return "baseline"
    if protocol_label.startswith("stage5_backend_auto_v1__"):
        return protocol_label.split("__", 1)[1]
    return normalize_name(directory.name)


def select_baseline(rows: List[Dict[str, object]], baseline_dir: Optional[str]) -> Optional[Dict[str, object]]:
    if not rows:
        return None
    if baseline_dir:
        baseline_path = str(Path(baseline_dir).resolve())
        for row in rows:
            if row["directory"] == baseline_path:
                return row
    for row in rows:
        if row.get("effective_protocol_label") == "stage5_backend_auto_v1":
            return row
    for row in rows:
        if row.get("experiment_label") == "baseline":
            return row
    for row in rows:
        name = str(row.get("name", ""))
        if "baseline" in name:
            return row
    for row in rows:
        name = str(row.get("name", ""))
        if name.endswith("_auto"):
            return row
    for row in rows:
        protocol_label = str(row.get("effective_protocol_label", ""))
        if protocol_label.startswith("frozen_round2"):
            return row
    return rows[0]


def build_notes(row: Dict[str, object]) -> str:
    notes: List[str] = []
    if row.get("benchmark_success") != 1:
        notes.append("benchmark_failed")
    if row.get("backend_success") != 1:
        notes.append("backend_failed")
    if row.get("fair_protocol_matched") == 0:
        notes.append("not_fair")
    if row.get("diagnostic_only") == 1:
        notes.append("diagnostic_only")
    if row.get("auto_fallback_used") == 1:
        notes.append("auto_fallback")
    failed_iterations = row.get("backend_failed_iterations_total")
    if isinstance(failed_iterations, int) and failed_iterations > 0:
        notes.append(f"failed_iters={failed_iterations}")
    return ",".join(notes)


def load_row(directory: Path) -> Dict[str, object]:
    experiment = parse_key_value_file(directory / "experiment_config_summary.txt")
    backend_vs = parse_key_value_file(directory / "backend_vs_frontend_summary.txt")
    backend_opt = parse_key_value_file(directory / "backend_optimization_summary.txt")
    protocol = parse_key_value_file(directory / "benchmark_protocol_summary.txt")
    auto_init = parse_key_value_file(directory / "auto_camera_initialization_summary.txt")

    row: Dict[str, object] = {
        "directory": str(directory),
        "name": directory.name,
        "experiment_label": infer_experiment_label(directory, experiment, protocol),
        "effective_protocol_label": experiment.get("effective_protocol_label", ""),
        "benchmark_success": to_int(protocol.get("success", "")),
        "backend_success": to_int(backend_opt.get("success", "")),
        "fair_protocol_matched": to_int(protocol.get("fair_protocol_matched", "")),
        "diagnostic_only": to_int(protocol.get("diagnostic_only", "")),
        "split_signature": protocol.get("split_signature", ""),
        "kalibr_source_label": protocol.get("kalibr_source_label", ""),
        "requested_run_second_pass": to_int(experiment.get("requested_run_second_pass", "")),
        "effective_run_second_pass": to_int(experiment.get("effective_run_second_pass", "")),
        "requested_frontend_intrinsics_release_mode": experiment.get(
            "requested_frontend_intrinsics_release_mode", ""
        ),
        "effective_frontend_intrinsics_release_mode": experiment.get(
            "effective_frontend_intrinsics_release_mode", ""
        ),
        "requested_backend_intrinsics_release_mode": experiment.get(
            "requested_backend_intrinsics_release_mode", ""
        ),
        "effective_backend_intrinsics_release_mode": experiment.get(
            "effective_backend_runner_intrinsics_release_mode",
            experiment.get("effective_backend_problem_intrinsics_release_mode", ""),
        ),
        "effective_enable_residual_sanity_gate": to_int(
            experiment.get("effective_enable_residual_sanity_gate", "")
        ),
        "effective_enable_board_pose_fit_gate": to_int(
            experiment.get("effective_enable_board_pose_fit_gate", "")
        ),
        "round1_selected_board_observation_count": to_int(
            experiment.get("round1_selected_board_observation_count", "")
        ),
        "round2_selected_board_observation_count": to_int(
            experiment.get("round2_selected_board_observation_count", "")
        ),
        "round2_selected_internal_point_count": to_int(
            experiment.get("round2_selected_internal_point_count", "")
        ),
        "auto_selected_mode": auto_init.get("selected_mode", ""),
        "auto_selected_source_label": auto_init.get("selected_source_label", ""),
        "auto_fallback_used": to_int(auto_init.get("fallback_used", "")),
        "auto_best_candidate_rmse": to_float(auto_init.get("best_candidate_rmse", "")),
        "auto_accepted_pose_fit_observation_count": to_int(
            auto_init.get("accepted_pose_fit_observation_count", "")
        ),
        "auto_failed_pose_fit_observation_count": to_int(
            auto_init.get("failed_pose_fit_observation_count", "")
        ),
        "training_frontend_overall_rmse": to_float(
            backend_vs.get("training_frontend_overall_rmse", "")
        ),
        "training_backend_overall_rmse": to_float(
            backend_vs.get("training_backend_overall_rmse", "")
        ),
        "training_kalibr_overall_rmse": to_float(
            backend_vs.get("training_kalibr_overall_rmse", "")
        ),
        "holdout_frontend_overall_rmse": to_float(
            backend_vs.get("holdout_frontend_overall_rmse", "")
        ),
        "holdout_backend_overall_rmse": to_float(
            backend_vs.get("holdout_backend_overall_rmse", "")
        ),
        "holdout_kalibr_overall_rmse": to_float(
            backend_vs.get("holdout_kalibr_overall_rmse", "")
        ),
        "training_backend_outer_only_rmse": to_float(
            backend_vs.get("training_backend_outer_only_rmse", "")
        ),
        "training_backend_internal_only_rmse": to_float(
            backend_vs.get("training_backend_internal_only_rmse", "")
        ),
        "holdout_backend_outer_only_rmse": to_float(
            backend_vs.get("holdout_backend_outer_only_rmse", "")
        ),
        "holdout_backend_internal_only_rmse": to_float(
            backend_vs.get("holdout_backend_internal_only_rmse", "")
        ),
        "backend_initial_overall_rmse": to_float(
            backend_vs.get("backend_initial_overall_rmse", "")
        ),
        "backend_optimized_overall_rmse": to_float(
            backend_vs.get("backend_optimized_overall_rmse", "")
        ),
        "backend_training_minus_kalibr": to_float(
            backend_vs.get("training_backend_minus_kalibr", "")
        ),
        "backend_holdout_minus_kalibr": to_float(
            backend_vs.get("holdout_backend_minus_kalibr", "")
        ),
        "backend_initial_cost": to_float(
            backend_opt.get("initial_backend_problem_total_cost", "")
        ),
        "backend_optimized_cost": to_float(
            backend_opt.get("optimized_backend_problem_total_cost", "")
        ),
        "backend_effective_frame_count": to_int(
            backend_opt.get("effective_frame_count", "")
        ),
        "backend_effective_board_observation_count": to_int(
            backend_opt.get("effective_board_observation_count", "")
        ),
        "backend_effective_total_point_count": to_int(
            backend_opt.get("effective_total_point_count", "")
        ),
    }

    failed_total = 0
    for line in (directory / "backend_optimization_summary.txt").read_text(
        encoding="utf-8"
    ).splitlines() if (directory / "backend_optimization_summary.txt").exists() else []:
        if line.startswith("stage_failed_iterations:"):
            failed = to_int(line.split(":", 1)[1].strip())
            if failed is not None:
                failed_total += failed
    row["backend_failed_iterations_total"] = failed_total
    row["notes"] = build_notes(row)
    return row


def compute_baseline_deltas(rows: List[Dict[str, object]], baseline: Optional[Dict[str, object]]) -> None:
    if baseline is None:
        return
    baseline_holdout = baseline.get("holdout_backend_overall_rmse")
    baseline_training = baseline.get("training_backend_overall_rmse")
    baseline_outer = baseline.get("holdout_backend_outer_only_rmse")
    baseline_internal = baseline.get("holdout_backend_internal_only_rmse")
    for row in rows:
        row["delta_vs_baseline_holdout_backend_overall"] = diff(
            row.get("holdout_backend_overall_rmse"), baseline_holdout
        )
        row["delta_vs_baseline_training_backend_overall"] = diff(
            row.get("training_backend_overall_rmse"), baseline_training
        )
        row["delta_vs_baseline_holdout_outer"] = diff(
            row.get("holdout_backend_outer_only_rmse"), baseline_outer
        )
        row["delta_vs_baseline_holdout_internal"] = diff(
            row.get("holdout_backend_internal_only_rmse"), baseline_internal
        )


def diff(lhs: object, rhs: object) -> Optional[float]:
    if not isinstance(lhs, float) or not isinstance(rhs, float):
        return None
    return lhs - rhs


CSV_COLUMNS: Sequence[str] = (
    "experiment_label",
    "name",
    "effective_protocol_label",
    "requested_run_second_pass",
    "effective_run_second_pass",
    "effective_frontend_intrinsics_release_mode",
    "effective_backend_intrinsics_release_mode",
    "effective_enable_residual_sanity_gate",
    "effective_enable_board_pose_fit_gate",
    "round1_selected_board_observation_count",
    "round2_selected_board_observation_count",
    "round2_selected_internal_point_count",
    "auto_selected_mode",
    "auto_selected_source_label",
    "auto_fallback_used",
    "auto_best_candidate_rmse",
    "training_backend_overall_rmse",
    "holdout_backend_overall_rmse",
    "holdout_backend_outer_only_rmse",
    "holdout_backend_internal_only_rmse",
    "training_kalibr_overall_rmse",
    "holdout_kalibr_overall_rmse",
    "backend_initial_overall_rmse",
    "backend_optimized_overall_rmse",
    "backend_initial_cost",
    "backend_optimized_cost",
    "backend_failed_iterations_total",
    "delta_vs_baseline_training_backend_overall",
    "delta_vs_baseline_holdout_backend_overall",
    "delta_vs_baseline_holdout_outer",
    "delta_vs_baseline_holdout_internal",
    "fair_protocol_matched",
    "diagnostic_only",
    "notes",
    "directory",
)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def write_markdown(path: Path, rows: List[Dict[str, object]], baseline: Optional[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Stage5 Backend Experiment Summary")
    lines.append("")
    lines.append(f"experiment_count: {len(rows)}")
    if baseline is not None:
        lines.append(f"baseline: {baseline['name']}")
    lines.append("")
    lines.append("## Main Table")
    lines.append("")
    header = (
        "| experiment | second_pass | frontend_mode | backend_mode | residual_gate | "
        "board_pose_gate | train_backend | holdout_backend | holdout_outer | "
        "holdout_internal | delta_holdout_vs_baseline | notes |"
    )
    divider = (
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    lines.append(header)
    lines.append(divider)
    for row in rows:
        lines.append(
            "| {experiment} | {second_pass} | {frontend_mode} | {backend_mode} | "
            "{residual_gate} | {board_gate} | {train_backend} | {holdout_backend} | "
            "{holdout_outer} | {holdout_internal} | {delta_holdout} | {notes} |".format(
                experiment=row["experiment_label"],
                second_pass=fmt_int(row.get("effective_run_second_pass")),
                frontend_mode=row.get("effective_frontend_intrinsics_release_mode", ""),
                backend_mode=row.get("effective_backend_intrinsics_release_mode", ""),
                residual_gate=fmt_int(row.get("effective_enable_residual_sanity_gate")),
                board_gate=fmt_int(row.get("effective_enable_board_pose_fit_gate")),
                train_backend=fmt_float(row.get("training_backend_overall_rmse")),
                holdout_backend=fmt_float(row.get("holdout_backend_overall_rmse")),
                holdout_outer=fmt_float(row.get("holdout_backend_outer_only_rmse")),
                holdout_internal=fmt_float(row.get("holdout_backend_internal_only_rmse")),
                delta_holdout=fmt_delta(
                    row.get("delta_vs_baseline_holdout_backend_overall")
                ),
                notes=row.get("notes", ""),
            )
        )
    lines.append("")
    lines.append("## Selection Scale")
    lines.append("")
    lines.append(
        "| experiment | round1_boards | round2_boards | round2_internal_points | "
        "auto_rmse | auto_fallback |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {experiment} | {r1} | {r2} | {r2_internal} | {auto_rmse} | {fallback} |".format(
                experiment=row["experiment_label"],
                r1=fmt_int(row.get("round1_selected_board_observation_count")),
                r2=fmt_int(row.get("round2_selected_board_observation_count")),
                r2_internal=fmt_int(row.get("round2_selected_internal_point_count")),
                auto_rmse=fmt_float(row.get("auto_best_candidate_rmse")),
                fallback=fmt_int(row.get("auto_fallback_used")),
            )
        )
    lines.append("")
    lines.append("## Optimization")
    lines.append("")
    lines.append(
        "| experiment | initial_backend_rmse | optimized_backend_rmse | "
        "initial_cost | optimized_cost | failed_iterations_total |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {experiment} | {initial_rmse} | {optimized_rmse} | {initial_cost} | "
            "{optimized_cost} | {failed} |".format(
                experiment=row["experiment_label"],
                initial_rmse=fmt_float(row.get("backend_initial_overall_rmse")),
                optimized_rmse=fmt_float(row.get("backend_optimized_overall_rmse")),
                initial_cost=fmt_float(row.get("backend_initial_cost")),
                optimized_cost=fmt_float(row.get("backend_optimized_cost")),
                failed=fmt_int(row.get("backend_failed_iterations_total")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Stage5 backend experiment directories into one table."
    )
    parser.add_argument("directories", nargs="*", help="Experiment directories to summarize.")
    parser.add_argument("--root", default="result", help="Root directory to scan.")
    parser.add_argument(
        "--prefix",
        default="",
        help="Only include experiment directories whose names start with this prefix.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary CSV/Markdown will be written.",
    )
    parser.add_argument(
        "--baseline-dir",
        default="",
        help="Optional explicit baseline directory. Otherwise inferred automatically.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    directories = collect_directories(args)
    if not directories:
        raise SystemExit("No experiment directories found.")

    rows = [load_row(directory) for directory in directories]
    rows.sort(
        key=lambda row: (
            row["experiment_label"] != "baseline",
            row.get("holdout_backend_overall_rmse") is None,
            row.get("holdout_backend_overall_rmse") or float("inf"),
            row["name"],
        )
    )
    baseline = select_baseline(rows, args.baseline_dir or None)
    compute_baseline_deltas(rows, baseline)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "stage5_backend_experiment_summary.csv", rows)
    write_markdown(output_dir / "stage5_backend_experiment_summary.md", rows, baseline)

    print(f"summarized_experiments={len(rows)}")
    if baseline is not None:
        print(f"baseline={baseline['name']}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize Stage6 stereo extrinsic experiment directories into one table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence


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


def parse_stereo_extrinsic_yaml(path: Path) -> SummaryMap:
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


def fmt_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_int(value: Optional[int]) -> str:
    return "" if value is None else str(value)


def collect_directories(args: argparse.Namespace) -> List[Path]:
    if args.directories:
        return [Path(entry).resolve() for entry in args.directories]
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    directories: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if args.prefix and not child.name.startswith(args.prefix):
            continue
        if (child / "stereo_extrinsic_summary.txt").exists():
            directories.append(child.resolve())
    return directories


def infer_subset_label(directory: Path) -> str:
    name = directory.name
    for token in ("first10", "first20", "first50", "full"):
        if token in name:
            return token
    return name


def parse_vector3(value: str) -> List[Optional[float]]:
    stripped = value.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return [None, None, None]
    parts = [part.strip() for part in stripped[1:-1].split(",")]
    values: List[Optional[float]] = [to_float(part) for part in parts[:3]]
    while len(values) < 3:
        values.append(None)
    return values


def parse_vector4(value: str) -> List[Optional[float]]:
    stripped = value.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return [None, None, None, None]
    parts = [part.strip() for part in stripped[1:-1].split(",")]
    values: List[Optional[float]] = [to_float(part) for part in parts[:4]]
    while len(values) < 4:
        values.append(None)
    return values


def load_row(directory: Path) -> Dict[str, object]:
    extrinsic = parse_key_value_file(directory / "stereo_extrinsic_summary.txt")
    extrinsic_yaml = parse_stereo_extrinsic_yaml(directory / "stereo_extrinsic.yaml")
    initialization = parse_key_value_file(directory / "stereo_initialization_summary.txt")
    graph = parse_key_value_file(directory / "stereo_graph_summary.txt")
    reprojection = parse_key_value_file(directory / "stereo_reprojection_summary.txt")
    runtime = parse_key_value_file(directory / "stage6_runtime_summary.txt")
    pair_selection = parse_key_value_file(directory / "stereo_pair_selection_summary.txt")
    intrinsics_sanity = parse_key_value_file(
        directory / "stereo_intrinsics_sanity_summary.txt"
    )
    global_sparse_ba = parse_key_value_file(
        directory / "stereo_global_sparse_ba_summary.txt"
    )
    translation = parse_vector3(
        extrinsic.get("translation_xyz", extrinsic_yaml.get("translation_xyz", ""))
    )
    quaternion = parse_vector4(
        extrinsic.get("quaternion_wxyz", extrinsic_yaml.get("quaternion_wxyz", ""))
    )
    rotation_angle_deg = to_float(
        extrinsic.get("rotation_angle_deg", extrinsic_yaml.get("rotation_angle_deg", ""))
    )
    if rotation_angle_deg is None and quaternion[0] is not None:
        w = max(-1.0, min(1.0, float(quaternion[0])))
        rotation_angle_deg = 2.0 * math.acos(w) * 180.0 / math.pi

    row: Dict[str, object] = {
        "subset_label": infer_subset_label(directory),
        "directory_name": directory.name,
        "directory": str(directory),
        "success": to_int(extrinsic.get("success", "")),
        "paired_frame_count": to_int(extrinsic.get("paired_frame_count", "")),
        "graph_propagation_iteration_count": to_int(
            initialization.get("graph_propagation_iteration_count", "")
        ),
        "graph_propagation_stopped_by_no_progress": to_int(
            initialization.get("graph_propagation_stopped_by_no_progress", "")
        ),
        "graph_propagation_stopped_by_iteration_limit": to_int(
            initialization.get("graph_propagation_stopped_by_iteration_limit", "")
        ),
        "reachable_training_pair_count": to_int(
            initialization.get("reachable_training_pair_count", "")
        ),
        "initialized_training_pair_count": to_int(
            initialization.get("initialized_training_pair_count", "")
        ),
        "excluded_training_pair_count": to_int(
            initialization.get("excluded_training_pair_count", "")
        ),
        "unreachable_training_pair_count": to_int(
            initialization.get("unreachable_training_pair_count", "")
        ),
        "training_shared_board_pair_count": to_int(
            reprojection.get("training_shared_board_pair_count", "")
        ),
        "training_single_camera_only_pair_count": to_int(
            reprojection.get("training_single_camera_only_pair_count", "")
        ),
        "training_total_stereo_rmse": to_float(
            reprojection.get("training_total_stereo_rmse", "")
        ),
        "training_cam0_rmse": to_float(reprojection.get("training_cam0_rmse", "")),
        "training_cam1_rmse": to_float(reprojection.get("training_cam1_rmse", "")),
        "training_cam1_over_cam0_rmse_ratio": to_float(
            reprojection.get("training_cam1_over_cam0_rmse_ratio", "")
        ),
        "training_shared_total_rmse": to_float(
            reprojection.get("training_shared_total_rmse", "")
        ),
        "training_shared_cam0_rmse": to_float(
            reprojection.get("training_shared_cam0_rmse", "")
        ),
        "training_shared_cam1_rmse": to_float(
            reprojection.get("training_shared_cam1_rmse", "")
        ),
        "training_shared_cam1_over_cam0_rmse_ratio": to_float(
            intrinsics_sanity.get("training_shared_cam1_over_cam0_rmse_ratio", "")
        )
        if intrinsics_sanity
        else (
            to_float(reprojection.get("training_shared_cam1_rmse", ""))
            / to_float(reprojection.get("training_shared_cam0_rmse", ""))
            if to_float(reprojection.get("training_shared_cam0_rmse", "")) not in (None, 0)
            and to_float(reprojection.get("training_shared_cam1_rmse", "")) is not None
            else None
        ),
        "same_intrinsics_path": to_int(
            intrinsics_sanity.get("same_intrinsics_path", "")
        ),
        "same_intrinsics_parameters": to_int(
            intrinsics_sanity.get("same_intrinsics_parameters", "")
        ),
        "same_resolution": to_int(intrinsics_sanity.get("same_resolution", "")),
        "likely_intrinsics_shared_scale_issue": to_int(
            intrinsics_sanity.get("likely_intrinsics_shared_scale_issue", "")
        ),
        "solver_mode": extrinsic.get(
            "solver_mode", reprojection.get("solver_mode", "")
        ),
        "eligible_pair_count": to_int(
            extrinsic.get(
                "eligible_pair_count",
                pair_selection.get("eligible_pair_count", ""),
            )
        ),
        "selected_pair_count": to_int(
            extrinsic.get(
                "selected_pair_count",
                pair_selection.get("selected_pair_count", ""),
            )
        ),
        "selection_reachable_pair_count": to_int(
            pair_selection.get("reachable_pair_count", "")
        ),
        "selection_initialized_pair_count": to_int(
            pair_selection.get("initialized_pair_count", "")
        ),
        "selection_selected_covered_board_count": to_int(
            pair_selection.get("selected_covered_board_count", "")
        ),
        "selection_selected_pose_fit_rmse_min": to_float(
            pair_selection.get("selected_pose_fit_rmse_min", "")
        ),
        "selection_selected_pose_fit_rmse_median": to_float(
            pair_selection.get("selected_pose_fit_rmse_median", "")
        ),
        "selection_selected_pose_fit_rmse_max": to_float(
            pair_selection.get("selected_pose_fit_rmse_max", "")
        ),
        "translation_x": translation[0],
        "translation_y": translation[1],
        "translation_z": translation[2],
        "rotation_angle_deg": rotation_angle_deg,
        "baseline_length": to_float(
            extrinsic.get("baseline_length", extrinsic_yaml.get("baseline_length", ""))
        ),
        "global_sparse_ba_initial_rmse": to_float(
            global_sparse_ba.get("initial_selected_rmse", "")
        ),
        "global_sparse_ba_final_rmse": to_float(
            global_sparse_ba.get("final_selected_rmse", "")
        ),
        "global_sparse_ba_shared_observation_count": to_int(
            global_sparse_ba.get("shared_observation_count", "")
        ),
        "global_sparse_ba_cam0_only_observation_count": to_int(
            global_sparse_ba.get("cam0_only_observation_count", "")
        ),
        "global_sparse_ba_cam1_only_observation_count": to_int(
            global_sparse_ba.get("cam1_only_observation_count", "")
        ),
        "global_sparse_ba_shared_observation_weight_sum": to_float(
            global_sparse_ba.get("shared_observation_weight_sum", "")
        ),
        "global_sparse_ba_cam0_only_observation_weight_sum": to_float(
            global_sparse_ba.get("cam0_only_observation_weight_sum", "")
        ),
        "global_sparse_ba_cam1_only_observation_weight_sum": to_float(
            global_sparse_ba.get("cam1_only_observation_weight_sum", "")
        ),
        "global_sparse_ba_shared_observation_weight_scale": to_float(
            global_sparse_ba.get("shared_observation_weight_scale", "")
        ),
        "global_sparse_ba_single_camera_only_observation_weight_scale": to_float(
            global_sparse_ba.get("single_camera_only_observation_weight_scale", "")
        ),
        "global_sparse_ba_single_camera_only_weight_mode": global_sparse_ba.get(
            "single_camera_only_weight_mode", ""
        ),
        "global_sparse_ba_single_camera_only_base_scale": to_float(
            global_sparse_ba.get("single_camera_only_base_scale", "")
        ),
        "global_sparse_ba_single_camera_only_per_side_budget_ratio": to_float(
            global_sparse_ba.get("single_camera_only_per_side_budget_ratio", "")
        ),
        "global_sparse_ba_shared_total_base_weight": to_float(
            global_sparse_ba.get("shared_total_base_weight", "")
        ),
        "global_sparse_ba_cam0_only_total_base_weight": to_float(
            global_sparse_ba.get("cam0_only_total_base_weight", "")
        ),
        "global_sparse_ba_cam1_only_total_base_weight": to_float(
            global_sparse_ba.get("cam1_only_total_base_weight", "")
        ),
        "global_sparse_ba_per_side_budget_limit": to_float(
            global_sparse_ba.get("per_side_budget_limit", "")
        ),
        "global_sparse_ba_adaptive_single_camera_only_per_side_cap_ratio": to_float(
            global_sparse_ba.get(
                "adaptive_single_camera_only_per_side_cap_ratio", ""
            )
        ),
        "global_sparse_ba_cam0_only_cap": to_float(
            global_sparse_ba.get("cam0_only_cap", "")
        ),
        "global_sparse_ba_cam1_only_cap": to_float(
            global_sparse_ba.get("cam1_only_cap", "")
        ),
        "global_sparse_ba_cam0_only_effective_scale": to_float(
            global_sparse_ba.get("cam0_only_effective_scale", "")
        ),
        "global_sparse_ba_cam1_only_effective_scale": to_float(
            global_sparse_ba.get("cam1_only_effective_scale", "")
        ),
        "global_sparse_ba_cam0_only_budget_clamped": to_int(
            global_sparse_ba.get("cam0_only_budget_clamped", "")
        ),
        "global_sparse_ba_cam1_only_budget_clamped": to_int(
            global_sparse_ba.get("cam1_only_budget_clamped", "")
        ),
        "connected_component_count": to_int(
            graph.get("connected_component_count", "")
        ),
        "gauge_connected_pair_count": to_int(
            graph.get("gauge_connected_pair_count", "")
        ),
        "gauge_connected_board_count": to_int(
            graph.get("gauge_connected_board_count", "")
        ),
        "pairing_build_dataset_runtime_seconds": to_float(
            runtime.get("pairing_build_dataset_runtime_seconds", "")
        ),
        "initialization_runtime_seconds": to_float(
            runtime.get("initialization_runtime_seconds", "")
        ),
        "training_optimization_runtime_seconds": to_float(
            runtime.get("training_optimization_runtime_seconds", "")
        ),
        "total_runtime_seconds": to_float(runtime.get("total_runtime_seconds", "")),
    }
    return row


CSV_COLUMNS: Sequence[str] = (
    "subset_label",
    "directory_name",
    "success",
    "paired_frame_count",
    "graph_propagation_iteration_count",
    "graph_propagation_stopped_by_no_progress",
    "graph_propagation_stopped_by_iteration_limit",
    "reachable_training_pair_count",
    "initialized_training_pair_count",
    "excluded_training_pair_count",
    "unreachable_training_pair_count",
    "training_shared_board_pair_count",
    "training_single_camera_only_pair_count",
    "training_total_stereo_rmse",
    "training_cam0_rmse",
    "training_cam1_rmse",
    "training_cam1_over_cam0_rmse_ratio",
    "training_shared_total_rmse",
    "training_shared_cam0_rmse",
    "training_shared_cam1_rmse",
    "training_shared_cam1_over_cam0_rmse_ratio",
    "same_intrinsics_path",
    "same_intrinsics_parameters",
    "same_resolution",
    "likely_intrinsics_shared_scale_issue",
    "solver_mode",
    "eligible_pair_count",
    "selected_pair_count",
    "selection_reachable_pair_count",
    "selection_initialized_pair_count",
    "selection_selected_covered_board_count",
    "selection_selected_pose_fit_rmse_min",
    "selection_selected_pose_fit_rmse_median",
    "selection_selected_pose_fit_rmse_max",
    "translation_x",
    "translation_y",
    "translation_z",
    "rotation_angle_deg",
    "baseline_length",
    "global_sparse_ba_initial_rmse",
    "global_sparse_ba_final_rmse",
    "global_sparse_ba_shared_observation_count",
    "global_sparse_ba_cam0_only_observation_count",
    "global_sparse_ba_cam1_only_observation_count",
    "global_sparse_ba_shared_observation_weight_sum",
    "global_sparse_ba_cam0_only_observation_weight_sum",
    "global_sparse_ba_cam1_only_observation_weight_sum",
    "global_sparse_ba_shared_observation_weight_scale",
    "global_sparse_ba_single_camera_only_observation_weight_scale",
    "global_sparse_ba_single_camera_only_weight_mode",
    "global_sparse_ba_single_camera_only_base_scale",
    "global_sparse_ba_single_camera_only_per_side_budget_ratio",
    "global_sparse_ba_shared_total_base_weight",
    "global_sparse_ba_cam0_only_total_base_weight",
    "global_sparse_ba_cam1_only_total_base_weight",
    "global_sparse_ba_per_side_budget_limit",
    "global_sparse_ba_adaptive_single_camera_only_per_side_cap_ratio",
    "global_sparse_ba_cam0_only_cap",
    "global_sparse_ba_cam1_only_cap",
    "global_sparse_ba_cam0_only_effective_scale",
    "global_sparse_ba_cam1_only_effective_scale",
    "global_sparse_ba_cam0_only_budget_clamped",
    "global_sparse_ba_cam1_only_budget_clamped",
    "connected_component_count",
    "gauge_connected_pair_count",
    "gauge_connected_board_count",
    "pairing_build_dataset_runtime_seconds",
    "initialization_runtime_seconds",
    "training_optimization_runtime_seconds",
    "total_runtime_seconds",
    "directory",
)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            formatted: Dict[str, object] = {}
            for key in CSV_COLUMNS:
                value = row.get(key, "")
                if isinstance(value, float):
                    formatted[key] = fmt_float(value)
                elif isinstance(value, int):
                    formatted[key] = fmt_int(value)
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Stage6 Stereo Experiment Summary")
    lines.append("")
    lines.append(f"experiment_count: {len(rows)}")
    lines.append("")
    lines.append(
        "| subset | pairs | graph_iter | no_progress | init_pairs | excluded | "
        "unreachable | shared_pairs | single_cam_pairs | train_total | cam0 | cam1 | "
        "cam1/cam0 | baseline | rot_deg |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in rows:
        lines.append(
            "| {subset} | {pairs} | {graph_iter} | {no_progress} | {init_pairs} | "
            "{excluded} | {unreachable} | {shared_pairs} | {single_pairs} | "
            "{train_total} | {cam0} | {cam1} | {ratio} | {baseline} | {rot} |".format(
                subset=row.get("subset_label", ""),
                pairs=fmt_int(row.get("paired_frame_count")),
                graph_iter=fmt_int(row.get("graph_propagation_iteration_count")),
                no_progress=fmt_int(row.get("graph_propagation_stopped_by_no_progress")),
                init_pairs=fmt_int(row.get("initialized_training_pair_count")),
                excluded=fmt_int(row.get("excluded_training_pair_count")),
                unreachable=fmt_int(row.get("unreachable_training_pair_count")),
                shared_pairs=fmt_int(row.get("training_shared_board_pair_count")),
                single_pairs=fmt_int(row.get("training_single_camera_only_pair_count")),
                train_total=fmt_float(row.get("training_total_stereo_rmse")),
                cam0=fmt_float(row.get("training_cam0_rmse")),
                cam1=fmt_float(row.get("training_cam1_rmse")),
                ratio=fmt_float(row.get("training_cam1_over_cam0_rmse_ratio")),
                baseline=fmt_float(row.get("baseline_length")),
                rot=fmt_float(row.get("rotation_angle_deg")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Stage6 stereo experiment directories into one table."
    )
    parser.add_argument("directories", nargs="*", help="Experiment directories to summarize.")
    parser.add_argument("--root", default="result", help="Root directory to scan.")
    parser.add_argument(
        "--prefix",
        default="stage6_stereo_",
        help="Only include experiment directories whose names start with this prefix.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary CSV/Markdown will be written.",
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
            str(row.get("subset_label", "")),
            row.get("paired_frame_count") is None,
            row.get("paired_frame_count") or 0,
            row.get("directory_name", ""),
        )
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "stage6_stereo_experiment_summary.csv", rows)
    write_markdown(output_dir / "stage6_stereo_experiment_summary.md", rows)

    print(f"summarized_experiments={len(rows)}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def parse_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> Optional[float]:
    if value == "":
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
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


def polar_metric(
    rows: List[Dict[str, str]],
    split: str,
    point_type: str,
    bucket: str,
    metric: str,
    camera_index: str = "-1",
    board_id: Optional[str] = None,
) -> Optional[float]:
    values: List[float] = []
    weights: List[int] = []
    for row in rows:
        if row.get("split") != split:
            continue
        if row.get("point_type") != point_type:
            continue
        if row.get("polar_bucket") != bucket:
            continue
        if row.get("camera_index") != camera_index:
            continue
        if board_id is not None and row.get("board_id") != board_id:
            continue
        value = to_float(row.get(metric, ""))
        count = to_int(row.get("point_count", ""))
        if value is None or count is None or count <= 0:
            continue
        values.append(value)
        weights.append(count)
    if not values:
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def polar_point_count(
    rows: List[Dict[str, str]],
    split: str,
    point_type: str,
    bucket: str,
    camera_index: str = "-1",
) -> Optional[int]:
    total = 0
    found = False
    for row in rows:
        if row.get("split") != split:
            continue
        if row.get("point_type") != point_type:
            continue
        if row.get("polar_bucket") != bucket:
            continue
        if row.get("camera_index") != camera_index:
            continue
        count = to_int(row.get("point_count", ""))
        if count is None:
            continue
        total += count
        found = True
    return total if found else None


def layout_drift_stats(rows: List[Dict[str, str]]) -> Dict[str, Optional[float]]:
    gaps: List[float] = []
    global_rmse: List[float] = []
    local_rmse: List[float] = []
    translations: List[float] = []
    rotations: List[float] = []
    for row in rows:
        global_value = to_float(row.get("global_outer_rmse_px", ""))
        local_value = to_float(row.get("local_outer_rmse_px", ""))
        translation = to_float(row.get("translation_drift_m", ""))
        rotation = to_float(row.get("rotation_drift_deg", ""))
        if global_value is not None and local_value is not None:
            gaps.append(global_value - local_value)
            global_rmse.append(global_value)
            local_rmse.append(local_value)
        if translation is not None:
            translations.append(translation)
        if rotation is not None:
            rotations.append(rotation)

    def mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def maximum(values: List[float]) -> Optional[float]:
        return max(values) if values else None

    return {
        "layout_drift_row_count": float(len(rows)) if rows else None,
        "layout_drift_mean_outer_gap_px": mean(gaps),
        "layout_drift_max_outer_gap_px": maximum(gaps),
        "layout_drift_mean_global_outer_rmse_px": mean(global_rmse),
        "layout_drift_mean_local_outer_rmse_px": mean(local_rmse),
        "layout_drift_mean_translation_m": mean(translations),
        "layout_drift_max_translation_m": maximum(translations),
        "layout_drift_mean_rotation_deg": mean(rotations),
        "layout_drift_max_rotation_deg": maximum(rotations),
    }


def selected_board_distribution(rows: List[Dict[str, str]]) -> Dict[str, object]:
    board_counts: Dict[int, int] = {}
    pair_ids = set()
    for row in rows:
        pair_id = to_int(row.get("pair_index", ""))
        board_id = to_int(row.get("board_id", ""))
        if pair_id is not None:
            pair_ids.add(pair_id)
        if board_id is None:
            continue
        board_counts[board_id] = board_counts.get(board_id, 0) + 1
    distribution = ",".join(
        f"{board_id}:{board_counts[board_id]}" for board_id in sorted(board_counts)
    )
    return {
        "persistent_selected_pair_count_from_csv": len(pair_ids),
        "persistent_selected_board_distribution": distribution,
    }


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
    persistent = parse_key_value_file(
        directory / "stage6_persistent_incremental_selection_summary.txt"
    )
    reference_holdout = parse_key_value_file(
        directory / "stereo_reference_holdout_summary.txt"
    )
    polar_rows = parse_csv_rows(directory / "stereo_holdout_board_polar_rmse.csv")
    selected_board_rows = parse_csv_rows(
        directory / "stage6_persistent_incremental_selected_boards.csv"
    )
    if not selected_board_rows:
        selected_board_rows = parse_csv_rows(
            directory / "stereo_pair_board_trial_selected_boards.csv"
        )
    drift = layout_drift_stats(
        parse_csv_rows(directory / "stereo_holdout_local_layout_drift.csv")
    )
    selected_distribution = selected_board_distribution(selected_board_rows)
    persistent_selected_pair_board_count = to_int(
        persistent.get("final_selected_pair_board_count", "")
    )
    if persistent_selected_pair_board_count is None and selected_board_rows:
        persistent_selected_pair_board_count = len(selected_board_rows)
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
        "holdout_extrinsic_only_total_stereo_rmse": to_float(
            reprojection.get("holdout_extrinsic_only_total_stereo_rmse", "")
        ),
        "holdout_extrinsic_only_outer_only_rmse": to_float(
            reprojection.get("holdout_extrinsic_only_outer_only_rmse", "")
        ),
        "holdout_extrinsic_only_internal_only_rmse": to_float(
            reprojection.get("holdout_extrinsic_only_internal_only_rmse", "")
        ),
        "holdout_extrinsic_only_used_pair_count": to_int(
            reprojection.get("holdout_extrinsic_only_used_pair_count", "")
        ),
        "reference_extrinsic_only_holdout_total_stereo_rmse": to_float(
            reference_holdout.get(
                "reference_extrinsic_only_holdout_total_stereo_rmse", ""
            )
        ),
        "ours_minus_reference_extrinsic_only_holdout_total_stereo_rmse": to_float(
            reference_holdout.get(
                "ours_minus_reference_extrinsic_only_holdout_total_stereo_rmse",
                "",
            )
        ),
        "persistent_incremental_estimator_used": to_int(
            persistent.get("persistent_incremental_estimator_used", "")
        ),
        "persistent_incremental_default_main_path": to_int(
            persistent.get("persistent_incremental_default_main_path", "")
        ),
        "persistent_incremental_seed_pair_count": to_int(
            persistent.get("persistent_incremental_seed_pair_count", "")
        ),
        "persistent_incremental_seed_pair_board_count": to_int(
            persistent.get("persistent_incremental_seed_pair_board_count", "")
        ),
        "persistent_incremental_seed_information_gain": to_float(
            persistent.get("persistent_incremental_seed_information_gain", "")
        ),
        "persistent_attempted_count": to_int(persistent.get("attempted_count", "")),
        "persistent_accepted_count": to_int(persistent.get("accepted_count", "")),
        "persistent_rejected_count": to_int(persistent.get("rejected_count", "")),
        "persistent_final_selected_pair_board_count": persistent_selected_pair_board_count,
        **selected_distribution,
        "persistent_rmse_delta_diagnostics_only": to_int(
            persistent.get("rmse_delta_diagnostics_only", "")
        ),
        "persistent_catastrophic_rejected_count": to_int(
            persistent.get("batch_acceptance_rejected_catastrophic_residual_count", "")
        ),
        "extrinsic_only_polar_0_30_rmse": polar_metric(
            polar_rows, "holdout_extrinsic_only", "all", "polar_0_30",
            "pixel_rmse_px"
        ),
        "extrinsic_only_polar_30_50_rmse": polar_metric(
            polar_rows, "holdout_extrinsic_only", "all", "polar_30_50",
            "pixel_rmse_px"
        ),
        "extrinsic_only_polar_50_70_rmse": polar_metric(
            polar_rows, "holdout_extrinsic_only", "all", "polar_50_70",
            "pixel_rmse_px"
        ),
        "extrinsic_only_polar_70_plus_rmse": polar_metric(
            polar_rows, "holdout_extrinsic_only", "all", "polar_70_plus",
            "pixel_rmse_px"
        ),
        "extrinsic_only_polar_50_70_point_count": polar_point_count(
            polar_rows, "holdout_extrinsic_only", "all", "polar_50_70"
        ),
        "extrinsic_only_polar_70_plus_point_count": polar_point_count(
            polar_rows, "holdout_extrinsic_only", "all", "polar_70_plus"
        ),
        **drift,
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
    "holdout_extrinsic_only_total_stereo_rmse",
    "holdout_extrinsic_only_outer_only_rmse",
    "holdout_extrinsic_only_internal_only_rmse",
    "holdout_extrinsic_only_used_pair_count",
    "reference_extrinsic_only_holdout_total_stereo_rmse",
    "ours_minus_reference_extrinsic_only_holdout_total_stereo_rmse",
    "persistent_incremental_estimator_used",
    "persistent_incremental_default_main_path",
    "persistent_incremental_seed_pair_count",
    "persistent_incremental_seed_pair_board_count",
    "persistent_incremental_seed_information_gain",
    "persistent_attempted_count",
    "persistent_accepted_count",
    "persistent_rejected_count",
    "persistent_final_selected_pair_board_count",
    "persistent_selected_pair_count_from_csv",
    "persistent_selected_board_distribution",
    "persistent_rmse_delta_diagnostics_only",
    "persistent_catastrophic_rejected_count",
    "extrinsic_only_polar_0_30_rmse",
    "extrinsic_only_polar_30_50_rmse",
    "extrinsic_only_polar_50_70_rmse",
    "extrinsic_only_polar_70_plus_rmse",
    "extrinsic_only_polar_50_70_point_count",
    "extrinsic_only_polar_70_plus_point_count",
    "layout_drift_row_count",
    "layout_drift_mean_outer_gap_px",
    "layout_drift_max_outer_gap_px",
    "layout_drift_mean_global_outer_rmse_px",
    "layout_drift_mean_local_outer_rmse_px",
    "layout_drift_mean_translation_m",
    "layout_drift_max_translation_m",
    "layout_drift_mean_rotation_deg",
    "layout_drift_max_rotation_deg",
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
        "| subset | pairs | selected | pair-board | accepted | rejected | "
        "boards | train RMSE | ours extrinsic-only | reference | ours-ref | "
        "polar 50-70 | polar 70+ | baseline | rot_deg |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in rows:
        lines.append(
            "| {subset} | {pairs} | {selected} | {pair_board} | {accepted} | "
            "{rejected} | {boards} | {train_total} | {extrinsic_only} | "
            "{reference} | {delta} | {polar_50_70} | {polar_70_plus} | {baseline} | "
            "{rot} |".format(
                subset=row.get("subset_label", ""),
                pairs=fmt_int(row.get("paired_frame_count")),
                selected=fmt_int(row.get("selected_pair_count")),
                pair_board=fmt_int(
                    row.get("persistent_final_selected_pair_board_count")
                ),
                accepted=fmt_int(row.get("persistent_accepted_count")),
                rejected=fmt_int(row.get("persistent_rejected_count")),
                boards=row.get("persistent_selected_board_distribution", ""),
                train_total=fmt_float(row.get("training_total_stereo_rmse")),
                extrinsic_only=fmt_float(
                    row.get("holdout_extrinsic_only_total_stereo_rmse")
                ),
                reference=fmt_float(
                    row.get("reference_extrinsic_only_holdout_total_stereo_rmse")
                ),
                delta=fmt_float(
                    row.get(
                        "ours_minus_reference_extrinsic_only_holdout_total_stereo_rmse"
                    )
                ),
                polar_50_70=fmt_float(row.get("extrinsic_only_polar_50_70_rmse")),
                polar_70_plus=fmt_float(
                    row.get("extrinsic_only_polar_70_plus_rmse")
                ),
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

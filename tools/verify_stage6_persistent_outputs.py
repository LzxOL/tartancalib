#!/usr/bin/env python3
"""Verify Stage6 persistent incremental stereo output artifacts."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_key_value_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def parse_numeric_vector(value: str) -> List[float] | None:
    match = re.fullmatch(r"\s*\[([^\]]*)\]\s*", value)
    if not match:
        return None
    try:
        values = [float(token.strip()) for token in match.group(1).split(",") if token.strip()]
    except ValueError:
        return None
    return values if all(math.isfinite(item) for item in values) else None


def parse_final_camera_yaml(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    values: Dict[str, str] = {}
    for key in ("camera_model", "distortion_model", "intrinsics", "resolution"):
        match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+)", text)
        if match:
            values[key] = match.group(1).strip()
    return values


def verify_final_camera_yaml(directory: Path, side: str, intrinsics: Dict[str, str],
                             results: List[Tuple[str, str, str]]) -> None:
    filename = f"stereo_final_{side}_intrinsics.yaml"
    if not require_file(directory, filename, results):
        return
    exported = parse_final_camera_yaml(directory / filename)
    prefix = "left" if side == "left" else "right"
    for field in ("camera_model", "distortion_model"):
        expected = intrinsics.get(f"{prefix}_{field}", "")
        if exported.get(field) == expected:
            add_result(results, "PASS", f"final {side} YAML {field}")
        else:
            add_result(results, "FAIL", f"final {side} YAML {field}",
                       f"got {exported.get(field, '')!r}, expected {expected!r}")
    for field in ("intrinsics", "resolution"):
        actual = parse_numeric_vector(exported.get(field, ""))
        expected = parse_numeric_vector(intrinsics.get(f"{prefix}_{field}", ""))
        if (actual is not None and expected is not None and len(actual) == len(expected)
                # The summary is intentionally human-readable and rounds values;
                # the exported YAML is the full-precision source of truth.
                and all(abs(lhs - rhs) <= max(1e-5, 5e-6 * max(1.0, abs(rhs)))
                        for lhs, rhs in zip(actual, expected))):
            add_result(results, "PASS", f"final {side} YAML {field} matches summary")
        else:
            add_result(results, "FAIL", f"final {side} YAML {field} matches summary",
                       f"got {exported.get(field, '')!r}, expected {intrinsics.get(f'{prefix}_{field}', '')!r}")


def parse_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def to_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def to_int(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def add_result(results: List[Tuple[str, str, str]], status: str, name: str,
               detail: str = "") -> None:
    results.append((status, name, detail))


def require_file(directory: Path, relative_path: str,
                 results: List[Tuple[str, str, str]]) -> bool:
    path = directory / relative_path
    if path.exists():
        add_result(results, "PASS", f"artifact:{relative_path}")
        return True
    add_result(results, "FAIL", f"artifact:{relative_path}", "missing")
    return False


def require_kv(data: Dict[str, str], key: str, expected: str,
               results: List[Tuple[str, str, str]]) -> None:
    value = data.get(key, "")
    if value == expected:
        add_result(results, "PASS", f"{key} == {expected}")
    else:
        add_result(results, "FAIL", f"{key} == {expected}", f"got {value!r}")


def require_positive_float(data: Dict[str, str], key: str,
                           results: List[Tuple[str, str, str]]) -> None:
    value = to_float(data.get(key, ""))
    if value is not None and value > 0.0:
        add_result(results, "PASS", f"{key} positive", f"{value:.6g}")
    else:
        add_result(results, "FAIL", f"{key} positive", f"got {data.get(key, '')!r}")


def require_positive_int(data: Dict[str, str], key: str,
                         results: List[Tuple[str, str, str]]) -> None:
    value = to_int(data.get(key, ""))
    if value is not None and value > 0:
        add_result(results, "PASS", f"{key} positive", str(value))
    else:
        add_result(results, "FAIL", f"{key} positive", f"got {data.get(key, '')!r}")


def require_close_float(lhs_data: Dict[str, str], lhs_key: str,
                        rhs_data: Dict[str, str], rhs_key: str,
                        tolerance: float,
                        results: List[Tuple[str, str, str]]) -> None:
    lhs = to_float(lhs_data.get(lhs_key, ""))
    rhs = to_float(rhs_data.get(rhs_key, ""))
    if lhs is not None and rhs is not None and abs(lhs - rhs) <= tolerance:
        add_result(
            results,
            "PASS",
            f"{lhs_key} matches {rhs_key}",
            f"{lhs:.6g} vs {rhs:.6g}",
        )
    else:
        add_result(
            results,
            "FAIL",
            f"{lhs_key} matches {rhs_key}",
            f"got {lhs_data.get(lhs_key, '')!r} vs {rhs_data.get(rhs_key, '')!r}",
        )


def require_columns(columns: Sequence[str], required: Iterable[str],
                    csv_name: str,
                    results: List[Tuple[str, str, str]]) -> None:
    missing = [column for column in required if column not in columns]
    if not missing:
        add_result(results, "PASS", f"{csv_name} required columns")
    else:
        add_result(results, "FAIL", f"{csv_name} required columns",
                   "missing " + ",".join(missing))


def selected_board_distribution(rows: List[Dict[str, str]]) -> str:
    counts: Dict[int, int] = {}
    for row in rows:
        board_id = to_int(row.get("board_id", ""))
        if board_id is None:
            continue
        counts[board_id] = counts.get(board_id, 0) + 1
    return ",".join(f"{board}:{counts[board]}" for board in sorted(counts))


def require_nonempty_kv(data: Dict[str, str], key: str,
                        results: List[Tuple[str, str, str]]) -> None:
    value = data.get(key, "")
    if value:
        add_result(results, "PASS", f"{key} present", value)
    else:
        add_result(results, "FAIL", f"{key} present", "missing")


def verify_polar_rows(rows: List[Dict[str, str]],
                      results: List[Tuple[str, str, str]]) -> None:
    if not rows:
        add_result(results, "FAIL", "polar bucket csv nonempty")
        return
    add_result(results, "PASS", "polar bucket csv nonempty", f"rows={len(rows)}")
    splits = {row.get("split", "") for row in rows}
    if "holdout_extrinsic_only" in splits:
        add_result(results, "PASS", "polar split holdout_extrinsic_only")
    else:
        add_result(results, "FAIL", "polar split holdout_extrinsic_only",
                   f"got {sorted(splits)}")
    buckets = {row.get("polar_bucket", "") for row in rows}
    if buckets:
        add_result(results, "PASS", "polar buckets present", ",".join(sorted(buckets)))
    else:
        add_result(results, "FAIL", "polar buckets present")
    has_positive_count = any((to_int(row.get("point_count", "")) or 0) > 0
                             for row in rows)
    if has_positive_count:
        add_result(results, "PASS", "polar positive point counts")
    else:
        add_result(results, "FAIL", "polar positive point counts")


def verify_frontend_prefilter(runtime: Dict[str, str],
                              results: List[Tuple[str, str, str]]) -> None:
    require_kv(runtime, "frontend_pairing_prefilter_enabled", "1", results)
    for side in ("left", "right"):
        original_key = f"frontend_original_{side}_frame_count"
        processed_key = f"frontend_processed_{side}_frame_count"
        skipped_key = f"frontend_skipped_unpaired_{side}_frame_count"
        original = to_int(runtime.get(original_key, ""))
        processed = to_int(runtime.get(processed_key, ""))
        skipped = to_int(runtime.get(skipped_key, ""))
        if (original is not None and processed is not None and skipped is not None and
                original > 0 and processed > 0 and processed <= original and
                skipped == original - processed):
            add_result(
                results,
                "PASS",
                f"frontend {side} prefilter counts",
                f"original={original} processed={processed} skipped={skipped}",
            )
        else:
            add_result(
                results,
                "FAIL",
                f"frontend {side} prefilter counts",
                f"original={runtime.get(original_key, '')!r} "
                f"processed={runtime.get(processed_key, '')!r} "
                f"skipped={runtime.get(skipped_key, '')!r}",
            )


def verify_directory(directory: Path,
                     require_rejection: bool,
                     expected_pose_structure: str,
                     require_visualizations: bool,
                     require_final_camera_yamls: bool) -> Tuple[bool, List[Tuple[str, str, str]]]:
    results: List[Tuple[str, str, str]] = []
    required_artifacts = [
        "stage6_init_summary.txt",
        "stage6_persistent_incremental_selection_summary.txt",
        "stage6_persistent_incremental_batch_decisions.csv",
        "stage6_persistent_incremental_pair_board_decisions.csv",
        "stage6_persistent_incremental_selected_boards.csv",
        "stereo_extrinsic.yaml",
        "stereo_intrinsics_sanity_summary.txt",
        "stereo_reprojection_summary.txt",
        "stereo_holdout_board_polar_rmse.csv",
        "stage6_runtime_summary.txt",
    ]
    for relative_path in required_artifacts:
        require_file(directory, relative_path, results)

    init = parse_key_value_file(directory / "stage6_init_summary.txt")
    persistent = parse_key_value_file(
        directory / "stage6_persistent_incremental_selection_summary.txt"
    )
    reprojection = parse_key_value_file(directory / "stereo_reprojection_summary.txt")
    extrinsic = parse_key_value_file(directory / "stereo_extrinsic.yaml")
    intrinsics = parse_key_value_file(directory / "stereo_intrinsics_sanity_summary.txt")
    pairing = parse_key_value_file(directory / "stereo_pairing_summary.txt")
    runtime = parse_key_value_file(directory / "stage6_runtime_summary.txt")

    require_kv(init, "stage6_initialization_role", "seed_only_no_selection", results)
    for key in (
        "candidate_count",
        "reachable_training_pair_count",
        "initialized_training_pair_count",
        "initialized_board_count",
    ):
        require_positive_int(init, key, results)
    require_positive_float(init, "medoid_score", results)
    require_positive_float(init, "pair_only_before_shared_rmse", results)
    require_positive_float(init, "pair_only_after_shared_rmse", results)

    expected_flags = {
        "enabled": "1",
        "persistent_incremental_estimator_used": "1",
        "persistent_incremental_default_main_path": "1",
        "persistent_incremental_uses_real_incremental_estimator": "1",
        "persistent_incremental_batch_unit": "pair_cohesive",
        "pair_board_selection_role": "ablation_fallback_diagnostic",
        "rmse_delta_diagnostics_only": "1",
        "batch_acceptance_policy": "persistent_incremental_estimator",
    }
    for key, expected in expected_flags.items():
        require_kv(persistent, key, expected, results)
    require_kv(
        persistent,
        "persistent_incremental_pose_structure",
        expected_pose_structure,
        results,
    )
    require_kv(
        persistent,
        "persistent_incremental_layout_updates_extrinsic",
        "0" if expected_pose_structure == "independent_pair_board" else "1",
        results,
    )
    for key in (
        "attempted_count",
        "accepted_count",
        "persistent_incremental_seed_pair_count",
        "persistent_incremental_seed_pair_board_count",
        "persistent_incremental_seed_information_gain",
    ):
        require_positive_float(persistent, key, results)

    if any(key.startswith("holdout_") and not key.startswith("holdout_extrinsic_only_")
           for key in reprojection):
        add_result(results, "FAIL", "reprojection uses extrinsic-only holdout only",
                   "found non-extrinsic holdout keys")
    else:
        add_result(results, "PASS", "reprojection uses extrinsic-only holdout only")
    require_positive_float(reprojection, "training_total_stereo_rmse", results)
    if expected_pose_structure == "independent_pair_board":
        metric_name = persistent.get("persistent_incremental_residual_metric_name", "")
        if metric_name == "pixel_px":
            require_close_float(
                persistent,
                "final_selected_rmse",
                reprojection,
                "training_total_stereo_rmse",
                1e-4,
                results,
            )
        elif metric_name in {"tangent_plane_rad", "pixel_tangent_px_equivalent"}:
            # The persistent selector reports its active objective domain while
            # the reprojection summary is always an independent pixel-space
            # evaluation. Comparing these values would mix units.
            require_positive_float(persistent, "final_selected_rmse", results)
            add_result(
                results,
                "PASS",
                "final_selected_rmse uses a non-pixel persistent metric",
                metric_name,
            )
        else:
            add_result(
                results,
                "FAIL",
                "persistent_incremental_residual_metric_name supported",
                f"got {metric_name!r}",
            )
    require_positive_float(
        reprojection, "holdout_extrinsic_only_total_stereo_rmse", results
    )

    require_kv(extrinsic, "cam0_is_reference", "1", results)
    require_nonempty_kv(extrinsic, "translation_xyz", results)
    require_nonempty_kv(extrinsic, "quaternion_wxyz", results)
    require_positive_float(extrinsic, "baseline_length", results)
    require_positive_int(extrinsic, "selected_pair_count", results)

    require_kv(intrinsics, "stage6_uses_external_intrinsics", "0", results)
    require_kv(
        intrinsics,
        "left_camera_seed_source",
        "stage6_auto_left_monocular_frontend",
        results,
    )
    require_kv(
        intrinsics,
        "right_camera_seed_source",
        "stage6_auto_right_monocular_frontend",
        results,
    )
    require_kv(intrinsics, "same_intrinsics_parameters", "0", results)
    require_kv(intrinsics, "same_resolution", "1", results)
    require_kv(intrinsics, "likely_intrinsics_shared_scale_issue", "0", results)
    require_nonempty_kv(intrinsics, "stage6_intrinsics_mode", results)
    if require_final_camera_yamls:
        verify_final_camera_yaml(directory, "left", intrinsics, results)
        verify_final_camera_yaml(directory, "right", intrinsics, results)
    requested_intrinsics_mode = intrinsics.get("stage6_requested_intrinsics_mode", "")
    if requested_intrinsics_mode in {
        "regularized_joint_projection",
        "adaptive_regularized_joint_projection",
    }:
        require_nonempty_kv(
            intrinsics, "stage6_effective_intrinsics_mode", results
        )
        require_nonempty_kv(
            intrinsics, "stage6_projection_release_reason", results
        )
        require_nonempty_kv(
            persistent,
            "persistent_incremental_projection_prior_enabled",
            results,
        )
        for key in (
            "persistent_incremental_projection_policy_training_pair_count",
            "persistent_incremental_projection_policy_shared_pair_board_count",
            "persistent_incremental_projection_policy_distinct_board_count",
            "persistent_incremental_projection_policy_observation_point_count",
        ):
            require_positive_int(persistent, key, results)
    require_nonempty_kv(pairing, "measurement_source_mode", results)
    require_nonempty_kv(
        pairing, "inherits_stage5_persistent_accepted_set", results
    )
    verify_frontend_prefilter(runtime, results)

    batch_columns, batch_rows = parse_csv(
        directory / "stage6_persistent_incremental_batch_decisions.csv"
    )
    required_batch_columns = [
        "batch_index",
        "pair_index",
        "batch_type",
        "accepted",
        "batchAccepted",
        "committed_or_rollback",
        "selected_board_ids",
        "JStart",
        "JFinal",
        "information_gain",
        "rankTheta",
        "rankPsi",
        "num_iterations",
        "accept_reason",
        "reject_reason",
    ]
    require_columns(batch_columns, required_batch_columns, "batch_decisions.csv",
                    results)
    if batch_rows:
        add_result(results, "PASS", "batch_decisions.csv nonempty",
                   f"rows={len(batch_rows)}")
    else:
        add_result(results, "FAIL", "batch_decisions.csv nonempty")
    if any(row.get("seed") == "1" and row.get("accepted") == "1"
           for row in batch_rows):
        add_result(results, "PASS", "seed batch accepted")
    else:
        add_result(results, "FAIL", "seed batch accepted")
    if all(row.get("committed_or_rollback") in ("committed", "rollback")
           for row in batch_rows):
        add_result(results, "PASS", "all batches commit or rollback")
    else:
        add_result(results, "FAIL", "all batches commit or rollback")
    rejected_rows = [
        row for row in batch_rows
        if row.get("accepted") == "0" or
        row.get("batchAccepted") == "0" or
        row.get("committed_or_rollback") == "rollback"
    ]
    if rejected_rows:
        add_result(results, "PASS", "rejected rollback batches present",
                   f"rows={len(rejected_rows)}")
    elif require_rejection:
        add_result(results, "FAIL", "rejected rollback batches present",
                   "required but none found")
    else:
        add_result(results, "INFO", "rejected rollback batches present",
                   "none in this run")

    selected_columns, selected_rows = parse_csv(
        directory / "stage6_persistent_incremental_selected_boards.csv"
    )
    require_columns(selected_columns, ["pair_index", "board_id"],
                    "selected_boards.csv", results)
    distribution = selected_board_distribution(selected_rows)
    if distribution:
        add_result(results, "PASS", "selected board distribution", distribution)
    else:
        add_result(results, "FAIL", "selected board distribution", "empty")

    polar_columns, polar_rows = parse_csv(
        directory / "stereo_holdout_board_polar_rmse.csv"
    )
    require_columns(
        polar_columns,
        [
            "split",
            "board_id",
            "camera_index",
            "point_type",
            "polar_bucket",
            "point_count",
            "pixel_rmse_px",
            "angular_rmse_deg",
        ],
        "stereo_holdout_board_polar_rmse.csv",
        results,
    )
    verify_polar_rows(polar_rows, results)

    if require_visualizations:
        viz_dir = directory / "stereo_backend_input_visualizations"
        image_count = len(list(viz_dir.glob("*.png"))) if viz_dir.exists() else 0
        if image_count > 0:
            add_result(results, "PASS", "backend input visualizations",
                       f"png_count={image_count}")
        else:
            add_result(results, "FAIL", "backend input visualizations",
                       "no png files")
    else:
        add_result(results, "INFO", "backend input visualizations",
                   "not required")

    success = not any(status == "FAIL" for status, _, _ in results)
    return success, results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Stage6 persistent incremental output directories."
    )
    parser.add_argument("directories", nargs="+", help="Stage6 output directories")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write stage6_persistent_output_verification.txt in each directory.",
    )
    parser.add_argument(
        "--require-rejection",
        action="store_true",
        help="Fail unless at least one persistent batch was rejected/rolled back.",
    )
    parser.add_argument(
        "--expected-pose-structure",
        choices=("independent_pair_board", "shared_frame_layout"),
        default="independent_pair_board",
        help="Require the persistent estimator to use this pose structure.",
    )
    parser.add_argument(
        "--require-visualizations",
        action="store_true",
        help="Fail unless backend-input PNG visualizations are present.",
    )
    parser.add_argument(
        "--require-final-camera-yamls",
        action="store_true",
        help="Fail unless exported Stage6 final camera YAML files match the final summary.",
    )
    args = parser.parse_args()

    overall_success = True
    for entry in args.directories:
        directory = Path(entry).resolve()
        success, results = verify_directory(
            directory,
            args.require_rejection,
            args.expected_pose_structure,
            args.require_visualizations,
            args.require_final_camera_yamls,
        )
        overall_success = overall_success and success
        lines = [f"directory: {directory}", f"success: {1 if success else 0}"]
        for status, name, detail in results:
            suffix = f" - {detail}" if detail else ""
            lines.append(f"{status}: {name}{suffix}")
        text = "\n".join(lines) + "\n"
        print(text, end="")
        if args.write_report:
            (directory / "stage6_persistent_output_verification.txt").write_text(
                text, encoding="utf-8"
            )
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())

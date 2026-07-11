#!/usr/bin/env python3
"""Verify Stage6 persistent incremental stereo output artifacts."""

from __future__ import annotations

import argparse
import csv
import math
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


def verify_directory(directory: Path,
                     require_rejection: bool,
                     require_reference: bool) -> Tuple[bool, List[Tuple[str, str, str]]]:
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
        "stereo_backend_input_visualizations",
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
    reference = parse_key_value_file(directory / "stereo_reference_holdout_summary.txt")

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
    require_positive_float(
        reprojection, "holdout_extrinsic_only_total_stereo_rmse", results
    )

    require_kv(extrinsic, "cam0_is_reference", "1", results)
    require_nonempty_kv(extrinsic, "translation_xyz", results)
    require_nonempty_kv(extrinsic, "quaternion_wxyz", results)
    require_positive_float(extrinsic, "baseline_length", results)
    require_positive_int(extrinsic, "selected_pair_count", results)

    require_nonempty_kv(intrinsics, "left_intrinsics_path", results)
    require_nonempty_kv(intrinsics, "right_intrinsics_path", results)
    require_kv(intrinsics, "same_intrinsics_path", "0", results)
    require_kv(intrinsics, "same_intrinsics_parameters", "0", results)
    require_kv(intrinsics, "same_resolution", "1", results)
    require_kv(intrinsics, "likely_intrinsics_shared_scale_issue", "0", results)

    if reference:
        require_kv(reference, "comparison_metric", "extrinsic_only_holdout", results)
        require_positive_float(
            reference, "ours_extrinsic_only_holdout_total_stereo_rmse", results
        )
        require_positive_float(
            reference, "reference_extrinsic_only_holdout_total_stereo_rmse", results
        )
        require_positive_int(reference, "extrinsic_only_holdout_used_pair_count", results)
        require_positive_int(
            reference, "reference_extrinsic_only_holdout_used_pair_count", results
        )
    elif require_reference:
        add_result(results, "FAIL", "reference holdout summary present", "missing")
    else:
        add_result(results, "INFO", "reference holdout summary present",
                   "not required")

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

    viz_dir = directory / "stereo_backend_input_visualizations"
    image_count = len(list(viz_dir.glob("*.png"))) if viz_dir.exists() else 0
    if image_count > 0:
        add_result(results, "PASS", "backend input visualizations",
                   f"png_count={image_count}")
    else:
        add_result(results, "FAIL", "backend input visualizations",
                   "no png files")

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
        "--require-reference",
        action="store_true",
        help="Fail unless stereo_reference_holdout_summary.txt is present and valid.",
    )
    args = parser.parse_args()

    overall_success = True
    for entry in args.directories:
        directory = Path(entry).resolve()
        success, results = verify_directory(
            directory, args.require_rejection, args.require_reference
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

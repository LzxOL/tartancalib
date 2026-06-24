#!/usr/bin/env python3
"""Archive reusable calibration artifacts from an experiment output directory.

The full Stage5/Stage6 result directories are useful for analysis, but they are
too bulky as a stable input for later validation reruns.  This helper keeps a
small, explicit artifact bundle with calibration parameters and key summaries.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


CALIBRATION_FILE_NAMES = {
    "stereo_extrinsic.yaml",
    "stereo_extrinsic_summary.txt",
    "backend_optimization_summary.txt",
    "backend_optimization_cost_parity_optimized_summary.txt",
    "backend_vs_kalibr_summary.txt",
    "benchmark_intrinsics_compare.csv",
    "stage5_bundle_summary.txt",
    "stage5_backend_problem_summary.txt",
}

CALIBRATION_SUFFIXES = (
    "camchain.yaml",
    "intrinsic.yaml",
    "intrinsics.yaml",
    "calibration.yaml",
)

SUMMARY_FILE_NAMES = {
    "runtime_summary.txt",
    "stage6_runtime_summary.txt",
    "stereo_reprojection_summary.txt",
    "stereo_reference_holdout_summary.txt",
    "stereo_pair_board_trial_selection_summary.txt",
    "stereo_holdout_board_polar_rmse.csv",
    "stereo_per_camera_residuals.csv",
    "stereo_per_board_residuals.csv",
    "backend_training_summary.txt",
    "backend_holdout_summary.txt",
    "experiment_config_summary.txt",
}


def is_calibration_file(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    if name in CALIBRATION_FILE_NAMES:
        return True
    return lower.endswith(CALIBRATION_SUFFIXES)


def should_copy(path: Path) -> bool:
    return is_calibration_file(path) or path.name in SUMMARY_FILE_NAMES


def copy_file(src: Path, dst_root: Path, run_dir: Path) -> dict[str, str]:
    relative = src.relative_to(run_dir)
    dst = dst_root / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    kind = "calibration" if is_calibration_file(src) else "summary"
    return {
        "kind": kind,
        "source": str(src),
        "artifact": str(dst),
    }


def infer_stage(run_dir: Path) -> str:
    if (run_dir / "stereo_extrinsic.yaml").exists():
        return "stage6"
    if (run_dir / "backend_optimization_summary.txt").exists():
        return "stage5"
    return "unknown"


def archive_run(run_dir: Path, archive_root: Path, name: str | None) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    artifact_name = name or run_dir.name
    artifact_dir = (archive_root / artifact_name).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and should_copy(path):
            copied.append(copy_file(path, artifact_dir, run_dir))

    manifest = {
        "artifact_name": artifact_name,
        "stage": infer_stage(run_dir),
        "source_run_dir": str(run_dir),
        "artifact_dir": str(artifact_dir),
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_files": [row for row in copied if row["kind"] == "calibration"],
        "summary_files": [row for row in copied if row["kind"] == "summary"],
    }
    manifest_path = artifact_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive reusable intrinsics/extrinsics artifacts from a Stage5/Stage6 result."
    )
    parser.add_argument("--run-dir", required=True, help="Experiment output directory")
    parser.add_argument(
        "--archive-root",
        default="report/calibration_artifacts",
        help="Root directory for reusable calibration artifacts",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional artifact bundle name; defaults to run directory name",
    )
    args = parser.parse_args()

    artifact_dir = archive_run(Path(args.run_dir), Path(args.archive_root), args.name)
    print(artifact_dir)


if __name__ == "__main__":
    main()

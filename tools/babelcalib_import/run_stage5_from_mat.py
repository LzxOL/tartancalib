#!/usr/bin/env python3
"""Run the native Stage5 pipeline from BabelCalib MAT observations."""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path

from scipy.io import loadmat


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Convert BabelCalib MAT observations and run Stage5.",
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-mat", type=Path)
    source.add_argument("--mat", type=Path, help="all.mat to split deterministically")
    parser.add_argument("--test-mat", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models", required=True)
    parser.add_argument(
        "--target-mode",
        choices=("auto", "single_board", "multi_board"),
        default="auto",
    )
    parser.add_argument("--kalibr-camchain", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--holdout-ratio", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument(
        "--all-training",
        action="store_true",
        help=(
            "Use every view in --mat for calibration. The same observations are "
            "provided to Stage5 only as a non-independent diagnostic dataset because "
            "the current benchmark entry point requires a holdout input."
        ),
    )
    parser.add_argument("--backend", type=Path, default=Path("build/run_stage5_backend"))
    parser.add_argument("--converter", type=Path)
    args, backend_args = parser.parse_known_args()
    if args.train_mat is not None and args.test_mat is None:
        parser.error("--train-mat requires --test-mat")
    if args.mat is not None and args.test_mat is not None:
        parser.error("--test-mat cannot be combined with --mat")
    if args.all_training and args.mat is None:
        parser.error("--all-training requires --mat")
    if not args.all_training and not 0.0 < args.holdout_ratio < 1.0:
        parser.error("--holdout-ratio must be in (0, 1)")
    return args, backend_args


def mat_view_count(path: Path) -> int:
    data = loadmat(path, variable_names=["corners"], squeeze_me=False)
    if "corners" not in data:
        raise RuntimeError(f"{path} is missing 'corners'")
    return int(data["corners"].size)


def run(command: list[str], cwd: Path) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    args, backend_args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    interchange_root = output / "precomputed_input"
    train_dir = interchange_root / "training"
    holdout_dir = interchange_root / (
        "training_diagnostic" if args.all_training else "holdout"
    )
    converter = (
        args.converter.resolve()
        if args.converter is not None
        else Path(__file__).with_name("convert_babelcalib_mat.py")
    )
    backend = args.backend if args.backend.is_absolute() else repo / args.backend

    train_mat = args.train_mat.resolve() if args.train_mat is not None else args.mat.resolve()
    holdout_mat = args.test_mat.resolve() if args.test_mat is not None else args.mat.resolve()
    train_index_file: Path | None = None
    holdout_index_file: Path | None = None
    if args.mat is not None and not args.all_training:
        view_count = mat_view_count(args.mat.resolve())
        indices = list(range(view_count))
        random.Random(args.split_seed).shuffle(indices)
        holdout_count = min(
            view_count - 3,
            max(1, int(round(args.holdout_ratio * view_count))),
        )
        if holdout_count < 1:
            raise RuntimeError("all.mat has too few views for a 70/30 Stage5 split")
        holdout_indices = sorted(indices[:holdout_count])
        train_indices = sorted(indices[holdout_count:])
        interchange_root.mkdir(parents=True, exist_ok=True)
        train_index_file = interchange_root / "training_view_indices.json"
        holdout_index_file = interchange_root / "holdout_view_indices.json"
        train_index_file.write_text(json.dumps(train_indices, indent=2) + "\n")
        holdout_index_file.write_text(json.dumps(holdout_indices, indent=2) + "\n")

    diagnostic_label = (
        "training_reprojection_diagnostic"
        if args.all_training
        else "holdout"
    )
    for mat_path, directory, label, index_file in (
        (train_mat, train_dir, "training", train_index_file),
        (holdout_mat, holdout_dir, diagnostic_label, holdout_index_file),
    ):
        command = [
            sys.executable,
            str(converter),
            "--mat",
            str(mat_path),
            "--output",
            str(directory),
            "--split-label",
            label,
        ]
        if index_file is not None:
            command.extend(["--view-indices-file", str(index_file)])
        run(command, repo)

    command = [
        str(backend),
        "--config",
        str(args.config.resolve()),
        "--models",
        args.models,
        "--kalibr-camchain",
        str(args.kalibr_camchain.resolve()),
        "--output",
        str(output),
        "--stage5-precomputed-observations-dir",
        str(train_dir),
        "--stage5-precomputed-holdout-observations-dir",
        str(holdout_dir),
        "--stage5-precomputed-target-mode",
        args.target_mode,
        "--runtime-mode",
        "research",
        "--stage5-disable-selected-case-visualizations",
        "--stage5-enable-polar-angle-diagnostics",
    ]
    command.extend(backend_args)
    interchange_root.mkdir(parents=True, exist_ok=True)
    (interchange_root / "stage5_command.txt").write_text(
        shlex.join(command) + "\n"
    )
    manifest = {
        "input_mode": "babelcalib_mat_precomputed",
        "calibration_protocol": (
            "all_views_training" if args.all_training else "random_holdout_ratio"
        ),
        "train_mat": str(train_mat),
        "holdout_mat": str(holdout_mat),
        "all_views_used_for_training": args.all_training,
        "independent_holdout": not args.all_training,
        "holdout_role": (
            "same_observations_reprojection_diagnostic_only"
            if args.all_training
            else "independent_evaluation"
        ),
        "split_seed": (
            args.split_seed
            if args.mat is not None and not args.all_training
            else None
        ),
        "holdout_ratio": (
            args.holdout_ratio
            if args.mat is not None and not args.all_training
            else None
        ),
        "boards_rt_used_to_initialize_layout": False,
        "target_mode_requested": args.target_mode,
        "image_detection_or_internal_regeneration_run": False,
    }
    (interchange_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    run(command, repo)
    if args.all_training:
        view_count = mat_view_count(train_mat)
        (output / "calibration_protocol_summary.txt").write_text(
            "\n".join([
                "success: 1",
                "calibration_protocol: all_views_training",
                f"training_view_count: {view_count}",
                "all_views_used_for_training: 1",
                "independent_holdout: 0",
                "holdout_role: same_observations_reprojection_diagnostic_only",
                "generalization_metrics_available: 0",
                "warning: The diagnostic copy contains the training observations and must not be reported as held-out performance.",
                "",
            ]),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

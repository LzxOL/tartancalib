#!/usr/bin/env python3
"""Run Stage5 from BabelCalib MAT observations and compare a camera catalog.

This is a reproducible wrapper around run_stage5_from_mat.py:

  train.mat -> Stage5 precomputed observations -> Stage5 selection BA
            -> frozen-test evaluation -> current-baseline YAML + catalog table

The MATLAB file is the source of the 2D/3D observations. No image detector is
called by this workflow. The reference camchain is passed only because the
Stage5 entry point requires one; --models selects the optimized model family.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
DEFAULT_SCIPY_PYTHON = Path(
    "/Users/linzhaoxian/.cache/codex-runtimes/"
    "codex-primary-runtime/dependencies/python/bin/python3"
)

MODEL_SPECS = {
    "ds": {"stage5": "ds-none", "suffix": "ds", "canonical_dir": "ds"},
    "kb": {"stage5": "pinhole-equi", "suffix": "kb", "canonical_dir": "kb"},
    "eucm": {"stage5": "eucm-none", "suffix": "eucm", "canonical_dir": "eucm"},
    "ucm": {"stage5": "omni-none", "suffix": "ucm", "canonical_dir": "ucm"},
    # Backward-compatible spelling used by earlier experiment scripts.
    "omni": {"stage5": "omni-none", "suffix": "omni", "canonical_dir": "ucm"},
}

DEFAULT_ALL_MODELS = ("ds", "kb", "eucm", "ucm")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mat", type=Path, help="Single all.mat input (legacy spelling).")
    source.add_argument("--train-mat", type=Path, help="MAT used for calibration.")
    parser.add_argument(
        "--test-mat",
        type=Path,
        default=None,
        help="Independent frozen-test MAT. It is never used for selection or BA.",
    )
    parser.add_argument("--dataset-id", required=True, help="Stable catalog id, e.g. checkerboard-3-25-all.")
    parser.add_argument("--camera", choices=("left", "right"), required=True)
    parser.add_argument(
        "--models", nargs="+", default=["all"], choices=("all", *MODEL_SPECS),
        help=(
            "One or more model aliases. Use --models all to run DS, KB, EUCM, "
            "and UCM sequentially in one experiment directory."
        ),
    )
    parser.add_argument("--catalog-subdir", default="checkerboard")
    parser.add_argument("--catalog-root", type=Path, default=REPO / "intrintic/catalog/current_baseline")
    parser.add_argument(
        "--canonical-root",
        "--reference-root",
        dest="canonical_root",
        type=Path,
        default=REPO / "intrintic/catalog/canonical",
        help="Canonical reference catalog root; auto-searches <root>/<camera>/<model>.",
    )
    parser.add_argument(
        "--reference-yaml",
        action="append",
        default=[],
        metavar="[LABEL:]PATH",
        help="Explicit comparison YAML; repeatable. Auto-discovered references are also used by default.",
    )
    parser.add_argument(
        "--disable-auto-canonical-references",
        action="store_true",
        help="Use only --reference-yaml instead of searching canonical-root.",
    )
    parser.add_argument("--run-root", type=Path, default=REPO / "result_may")
    parser.add_argument("--config", type=Path, default=REPO / "aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml")
    parser.add_argument(
        "--reference-camchain",
        "--kalibr-camchain",
        dest="reference_camchain",
        type=Path,
        default=None,
        help="Reference camchain passed to Stage5; auto-selects canonical Kalibr YAML when omitted.",
    )
    parser.add_argument("--holdout-ratio", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all MAT views for calibration; do not create a train/test split.",
    )
    parser.add_argument("--target-mode", choices=("auto", "single_board", "multi_board"), default="single_board")
    parser.add_argument("--stage5-python", type=Path, default=DEFAULT_SCIPY_PYTHON)
    parser.add_argument("--backend", type=Path, default=REPO / "build/run_stage5_backend")
    parser.add_argument("--tag", default=None, help="Optional non-overwriting run/catalog filename suffix.")
    parser.add_argument("--dry-run", action="store_true")
    args, stage5_args = parser.parse_known_args()
    if args.train_mat is not None and args.test_mat is None and not args.all:
        parser.error("--train-mat requires --test-mat, or use --mat for an internal split")
    if args.test_mat is not None and args.mat is not None:
        parser.error("--test-mat must be used with --train-mat, not --mat")
    if not args.all and not 0.0 < args.holdout_ratio < 1.0:
        parser.error("--holdout-ratio must be in (0, 1)")
    return args, stage5_args


def selected_models(values: list[str]) -> list[str]:
    if "all" in values:
        return list(DEFAULT_ALL_MODELS)
    return list(dict.fromkeys(values))


def default_reference_camchain(camera: str) -> Path:
    return REPO / "config" / f"mono_fisheye_calib_3_25_{camera}-camchain.yaml"


def model_reference_directory(root: Path, camera: str, spec: dict[str, str]) -> Path:
    return root / camera / spec["canonical_dir"]


def reference_candidates(
    root: Path, camera: str, spec: dict[str, str], dataset_id: str
) -> list[Path]:
    """Find normalized reference YAMLs without assuming one dataset directory name."""
    camera_root = root / camera
    if not camera_root.is_dir():
        return []
    model_dir = spec["canonical_dir"].lower()
    suffix = spec["suffix"].lower()
    direct = model_reference_directory(root, camera, spec)
    candidates = sorted(direct.rglob("*.yaml")) if direct.is_dir() else []
    if not candidates:
        candidates = sorted(camera_root.rglob("*.yaml"))
    dataset_hint = dataset_id.lower().replace("_", "-")
    for suffix in ("-all", "-clear"):
        if dataset_hint.endswith(suffix):
            dataset_hint = dataset_hint[: -len(suffix)]
    if dataset_hint:
        dataset_matches = [
            path for path in candidates
            if dataset_hint in str(path).lower().replace("_", "-")
        ]
        if dataset_matches:
            candidates = dataset_matches
    selected: list[Path] = []
    for path in candidates:
        parts = {part.lower() for part in path.parts}
        stem = path.stem.lower()
        in_model_dir = model_dir in parts
        suffix_match = stem.endswith(f"__{suffix}")
        # UCM is the native omni-none catalog. Do not accidentally include
        # omni-radtan files from the legacy camera-root layout.
        if spec["canonical_dir"] == "ucm":
            suffix_match = suffix_match or stem.endswith("__omni")
            if "omni-radtan" in stem and not in_model_dir:
                suffix_match = False
        if in_model_dir or suffix_match:
            selected.append(path.resolve())
    return list(dict.fromkeys(selected))


def reference_label(path: Path) -> str:
    return path.stem


def split_reference_spec(spec: str) -> tuple[str, Path]:
    separator = spec.find(":")
    if separator > 0:
        return spec[:separator], Path(spec[separator + 1:])
    path = Path(spec)
    return reference_label(path), path


def reference_spec(label: str, path: Path) -> str:
    return f"{label}:{path}"


def choose_kalibr_reference(
    candidates: list[Path], explicit: Path | None, camera: str
) -> Path:
    if explicit is not None:
        return explicit.resolve()
    kalibr = [path for path in candidates if "kalibr" in path.stem.lower()]
    if kalibr:
        # Prefer the ordinary ``__kalibr__`` file over the optional
        # ``kalibr-stereo`` reference when both variants exist.
        def kalibr_priority(path: Path) -> tuple[int, str]:
            stem = path.stem.lower()
            if "__kalibr__" in stem:
                return (0, str(path))
            if "kalibr-stereo" in stem:
                return (1, str(path))
            return (2, str(path))

        kalibr.sort(key=kalibr_priority)
        return kalibr[0]
    return default_reference_camchain(camera).resolve()


def sanitize_reference_key(label: str) -> str:
    sanitized: list[str] = []
    last_was_underscore = False
    for character in label:
        if character.isalnum():
            sanitized.append(character.lower())
            last_was_underscore = False
        elif not last_was_underscore:
            sanitized.append("_")
            last_was_underscore = True
    return "".join(sanitized).strip("_")


def metric(values: dict[str, str], prefixes: list[str], names: tuple[str, ...]) -> str:
    for prefix in prefixes:
        for name in names:
            value = values.get(prefix + name, "")
            if value and value.lower() not in {"nan", "inf", "-inf"}:
                return value
    return ""


def comparison_rows(
    output: Path,
    references: list[tuple[str, Path]],
    kalibr_reference: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split, ours_filename, reference_filename in (
        ("training", "backend_training_summary.txt", "benchmark_training_summary.txt"),
        ("frozen_test", "backend_holdout_summary.txt", "benchmark_holdout_summary.txt"),
    ):
        ours_path = output / ours_filename
        reference_path = output / reference_filename
        if not ours_path.is_file():
            continue
        ours_values = read_key_value(ours_path)
        reference_values = read_key_value(
            reference_path if reference_path.is_file() else ours_path
        )
        rows.append({
            "split": split,
            "method": "ours-baseline",
            "source_yaml": "",
            "rmse_px": metric(ours_values, ["", "our_"], ("overall_rmse", "test_rmse_all_control_points")),
            "p95_px": metric(ours_values, ["", "our_"], ("p95_reprojection_error",)),
            "pose_success_rate": metric(ours_values, ["", "our_"], ("test_pose_refit_success_rate", "pose_only_refit_success_rate")),
            "point_count": metric(ours_values, ["", "our_"], ("point_count",)),
        })
        rows.append({
            "split": split,
            "method": "kalibr",
            "source_yaml": str(kalibr_reference),
            "rmse_px": metric(reference_values, ["kalibr_"], ("overall_rmse", "test_rmse_all_control_points")),
            "p95_px": metric(reference_values, ["kalibr_"], ("p95_reprojection_error",)),
            "pose_success_rate": metric(reference_values, ["kalibr_"], ("test_pose_refit_success_rate", "pose_only_refit_success_rate")),
            "point_count": metric(reference_values, ["kalibr_"], ("point_count",)),
        })
        for label, yaml_path in references:
            prefix = f"reference_{sanitize_reference_key(label)}_"
            rows.append({
                "split": split,
                "method": label,
                "source_yaml": str(yaml_path),
                "rmse_px": metric(reference_values, [prefix], ("overall_rmse", "test_rmse_all_control_points")),
                "p95_px": metric(reference_values, [prefix], ("p95_reprojection_error",)),
                "pose_success_rate": metric(reference_values, [prefix], ("test_pose_refit_success_rate", "pose_only_refit_success_rate")),
                "point_count": metric(reference_values, [prefix], ("point_count",)),
            })
    return rows


def write_comparison_outputs(
    run_base: Path,
    model_output: Path,
    references: list[tuple[str, Path]],
    kalibr_reference: Path,
) -> list[dict[str, str]]:
    rows = comparison_rows(model_output, references, kalibr_reference)
    import csv

    fieldnames = ["split", "method", "source_yaml", "rmse_px", "p95_px", "pose_success_rate", "point_count"]
    model_output.mkdir(parents=True, exist_ok=True)
    with (model_output / "canonical_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_all_model_comparison(run_base: Path, rows: list[dict[str, str]]) -> None:
    import csv

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    run_base.mkdir(parents=True, exist_ok=True)
    with (run_base / "canonical_comparison_all_models.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_key_value(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def csv_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def yaml_floats(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.17g}" for value in values) + "]"


def run_resolution(source_run: Path) -> tuple[int, int]:
    for name in ("stage5_bundle_summary.txt", "stage5_round1_bundle_summary.txt"):
        path = source_run / name
        if not path.is_file():
            continue
        values = read_key_value(path)
        if "camera_resolution_width" in values and "camera_resolution_height" in values:
            return int(float(values["camera_resolution_width"])), int(float(values["camera_resolution_height"]))
    raise RuntimeError(f"Stage5 output has no camera resolution summary: {source_run}")


def canonical_yaml(
    summary: dict[str, str],
    source_run: Path,
    source_mat: Path,
    camera: str,
    calibration_protocol: str,
) -> str:
    camera_model = summary.get("camera_model", "")
    distortion_model = summary.get("camera_distortion_model", "")
    intrinsics = csv_floats(summary.get("camera_intrinsics_csv", ""))
    distortion = csv_floats(summary.get("camera_distortion_csv", ""))
    width, height = run_resolution(source_run)
    if not camera_model or not distortion_model or not intrinsics:
        raise RuntimeError(f"missing final camera fields in {source_run}")
    return "\n".join([
        f"# Canonical current-baseline camchain. Generated from: {source_run}",
        f"# Input observations: {source_mat}",
        f"# Calibration protocol: {calibration_protocol}",
        "# Holdout metrics are diagnostic-only when calibration_protocol=all_views_training.",
        "# Parameters are the final Stage5 persistent incremental BA state.",
        f"# Requested model family: {summary.get('camera_model_family', '')}",
        "cam0:",
        "  cam_overlaps: []",
        f"  camera_model: {camera_model}",
        f"  distortion_model: {distortion_model}",
        f"  intrinsics: {yaml_floats(intrinsics)}",
        f"  distortion_coeffs: {yaml_floats(distortion)}",
        f"  resolution: [{width}, {height}]",
        f'  rostopic: "/vimbax_camera_{"left" if camera == "left" else "37086"}/image_raw"',
        "",
    ])


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    args, stage5_args = parse_args()
    train_mat = (args.train_mat or args.mat).resolve()
    if not train_mat.is_file():
        raise FileNotFoundError(f"training MAT does not exist: {train_mat}")
    test_mat = args.test_mat.resolve() if args.test_mat is not None else None
    if test_mat is not None and not test_mat.is_file():
        raise FileNotFoundError(f"--test-mat does not exist: {test_mat}")
    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(f"--config does not exist: {config}")
    canonical_root = args.canonical_root.resolve()
    spec_candidates: dict[str, list[Path]] = {
        alias: reference_candidates(canonical_root, args.camera, spec, args.dataset_id)
        for alias, spec in MODEL_SPECS.items()
    }
    selected = selected_models(args.models)
    explicit_reference = (
        args.reference_camchain.resolve()
        if args.reference_camchain is not None
        else None
    )
    if explicit_reference is not None and not explicit_reference.is_file():
        raise FileNotFoundError(
            f"--reference-camchain does not exist: {explicit_reference}"
        )
    python = args.stage5_python.resolve() if args.stage5_python.is_file() else Path(sys.executable)
    backend = args.backend.resolve()
    if not backend.is_file() and not args.dry_run:
        raise FileNotFoundError(f"--backend does not exist: {backend}")

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_base = args.run_root.resolve() / f"stage5_mat_{args.dataset_id}_{args.camera}_{tag}"
    catalog_base = args.catalog_root.resolve() / args.catalog_subdir
    runner = Path(__file__).with_name("run_stage5_from_mat.py")
    protocol = (
        "explicit_train_frozen_test" if test_mat is not None
        else "all_views_training" if args.all
        else "random_holdout_ratio"
    )

    all_rows: list[dict[str, str]] = []
    manifest: dict[str, Any] = {
        "workflow": "babelcalib_mat_to_stage5_current_baseline_catalog_and_comparison",
        "mat": str(train_mat),
        "train_mat": str(train_mat),
        "test_mat": str(test_mat) if test_mat is not None else None,
        "config": str(config),
        "reference_camchain": str(explicit_reference) if explicit_reference else None,
        "canonical_root": str(canonical_root),
        "models": [],
        "stage5_extra_args": stage5_args,
        "target_mode": args.target_mode,
        "calibration_protocol": protocol,
        "all_views_used_for_training": bool(args.all or test_mat is not None),
        "independent_holdout": test_mat is not None or not args.all,
        "holdout_role": (
            "independent_frozen_precomputed_test"
            if test_mat is not None
            else "same_observations_reprojection_diagnostic_only"
            if args.all
            else "independent_evaluation"
        ),
        "canonical_references_by_model": {},
        "reference_camchain_by_model": {},
    }
    for alias in selected:
        spec = MODEL_SPECS[alias]
        candidates = spec_candidates[alias]
        reference = choose_kalibr_reference(
            candidates, explicit_reference, args.camera
        )
        if not reference.is_file():
            raise FileNotFoundError(
                "No usable Kalibr/reference camchain found for model "
                f"{alias}; pass --reference-camchain explicitly or populate "
                f"{canonical_root / args.camera / spec['canonical_dir']}."
            )
        explicit_refs: list[tuple[str, Path]] = []
        for raw_spec in args.reference_yaml:
            label, path = split_reference_spec(raw_spec)
            path = path.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"--reference-yaml does not exist: {path}")
            explicit_refs.append((label, path))
        discovered = [] if args.disable_auto_canonical_references else [
            (reference_label(path), path) for path in candidates
        ]
        references: list[tuple[str, Path]] = []
        seen_paths: set[Path] = set()
        for label, path in discovered + explicit_refs:
            if path.resolve() == reference.resolve() or path.resolve() in seen_paths:
                continue
            seen_paths.add(path.resolve())
            references.append((label, path.resolve()))

        output = run_base / alias
        command = [
            str(python), str(runner),
            "--config", str(config),
            "--models", spec["stage5"],
            "--target-mode", args.target_mode,
            "--kalibr-camchain", str(reference),
            "--output", str(output),
            "--backend", str(backend),
        ]
        if test_mat is not None:
            command.extend(["--train-mat", str(train_mat), "--test-mat", str(test_mat)])
        elif args.all:
            command.extend(["--mat", str(train_mat), "--all-training"])
        else:
            command.extend(["--mat", str(train_mat)])
            command.extend([
                "--holdout-ratio", str(args.holdout_ratio),
                "--split-seed", str(args.split_seed),
            ])
        for label, path in references:
            command.extend(["--reference-intrinsics-yaml", reference_spec(label, path)])
        command.extend(stage5_args)
        run(command, REPO, args.dry_run)
        entry: dict[str, Any] = {
            "alias": alias,
            "stage5_model": spec["stage5"],
            "run_output": str(output),
            "reference_camchain": str(reference),
            "reference_yamls": [
                {"label": label, "path": str(path)} for label, path in references
            ],
        }
        if not args.dry_run:
            summary_path = output / "backend_training_summary.txt"
            if not summary_path.is_file():
                raise RuntimeError(f"Stage5 completed without {summary_path}")
            summary = read_key_value(summary_path)
            filename = f"{args.dataset_id}__{args.camera}__ours-baseline__{spec['suffix']}.yaml"
            catalog_path = catalog_base / filename
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            catalog_path.write_text(
                canonical_yaml(
                    summary,
                    output,
                    train_mat,
                    args.camera,
                    protocol,
                ),
                encoding="utf-8",
            )
            rows = write_comparison_outputs(run_base, output, references, reference)
            all_rows.extend([{**row, "model": alias} for row in rows])
            entry.update({
                "summary": str(summary_path),
                "catalog_yaml": str(catalog_path),
                "comparison_csv": str(output / "canonical_comparison.csv"),
            })
        manifest["canonical_references_by_model"][alias] = [
            {"label": label, "path": str(path)} for label, path in references
        ]
        manifest["reference_camchain_by_model"][alias] = str(reference)
        manifest["models"].append(entry)
    run_base.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        write_all_model_comparison(run_base, all_rows)
    (run_base / "catalog_publish_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

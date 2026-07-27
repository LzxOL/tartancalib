#!/usr/bin/env python3
"""Run Stage5 Spherical BA and tabulate canonical camera baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    canonical_group: str
    compatible_models: tuple[tuple[str, str], ...]
    label: str


MODEL_ALIASES = {
    "ds": "ds-none",
    "double-sphere": "ds-none",
    "double_sphere": "ds-none",
    "kb": "pinhole-equi",
    "pinhole-equi": "pinhole-equi",
    "pinhole_equi": "pinhole-equi",
    "ucm": "omni-none",
    "mei": "omni-none",
    "omni": "omni-none",
    "omni-none": "omni-none",
    "eucm": "eucm-none",
    "eucm-none": "eucm-none",
}

MODEL_SPECS = {
    "ds-none": ModelSpec("ds", (("ds", "none"),), "DS"),
    "pinhole-equi": ModelSpec(
        "kb", (("pinhole", "equidistant"),), "KB"
    ),
    "omni-none": ModelSpec("ucm", (("omni", "none"),), "UCM / Mei"),
    "eucm-none": ModelSpec("eucm", (("eucm", "none"),), "EUCM"),
}

METHOD_ORDER = {
    "ours": 0,
    "kalibr": 1,
    "tartancalib": 2,
    "camodocal": 3,
    "babelcalib": 4,
    "opencv": 5,
}


@dataclass(frozen=True)
class CanonicalCamera:
    method: str
    label: str
    path: Path
    camera_model: str
    distortion_model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage5 Spherical BA and export a canonical-baseline table."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Image directory, for example stereo_dataset_20260430_144928-clear/right.",
    )
    parser.add_argument(
        "--models",
        "--model",
        dest="models",
        default="ds,kb,ucm,eucm",
        help="Comma-separated models: ds, kb, ucm/mei/omni, eucm.",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("intrintic/catalog/canonical/right"),
        help="Canonical camera catalog root, relative to the repository by default.",
    )
    parser.add_argument(
        "--canonical-target",
        help="Target directory under canonical-root. Required for multiple targets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-timestamp",
        help="Optional YYYYMMDD_HHMMSS suffix. Defaults to the current Asia/Shanghai time.",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--current-baseline-root",
        type=Path,
        default=Path("intrintic/catalog/current_baseline/mul-board"),
        help="Directory where this run's final Ours camera YAML files are saved.",
    )
    parser.add_argument(
        "--save-current-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save each final Ours camera to current_baseline/mul-board.",
    )
    parser.add_argument("--inlier-threshold-px", type=float, default=1.0)
    parser.add_argument(
        "--include-omni-radtan-references",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include canonical Omni-radtan references in the UCM/Mei group.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the default run_stage5_backend build step.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print resolved commands without running."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow an existing output directory."
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def normalize_models(value: str) -> list[str]:
    normalized: list[str] = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in MODEL_ALIASES:
            raise ValueError(f"Unsupported model '{raw}'.")
        model = MODEL_ALIASES[key]
        if model not in normalized:
            normalized.append(model)
    if not normalized:
        raise ValueError("--models must contain at least one model.")
    return normalized


def parse_camera_yaml(path: Path) -> tuple[str, str]:
    camera_model = ""
    distortion_model = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r'\s*camera_model:\s*"?([^\s"]+)', line)
        if match:
            camera_model = match.group(1).strip().lower()
        match = re.match(r'\s*distortion_model:\s*"?([^\s"]+)', line)
        if match:
            distortion_model = match.group(1).strip().lower()
    if not camera_model or not distortion_model:
        raise ValueError(f"Invalid canonical camera YAML: {path}")
    return camera_model, distortion_model


def method_from_filename(path: Path) -> tuple[str, str]:
    name = path.stem.lower()
    for method, label in (
        ("kalibr", "Kalibr"),
        ("tartancalib", "TartanCalib"),
        ("camodocal", "CamOdoCal"),
        ("babelcalib", "BabelCalib"),
        ("opencv", "OpenCV"),
    ):
        if method in name:
            return method, label
    raise ValueError(f"Cannot determine baseline method from canonical file: {path}")


def resolve_target_dir(root: Path, target: str | None) -> Path:
    if target:
        target_path = root / target
        if not target_path.is_dir():
            raise ValueError(f"Canonical target does not exist: {target_path}")
        return target_path
    target_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(target_dirs) == 1:
        return target_dirs[0]
    if not target_dirs:
        raise ValueError(f"No target directory found under canonical root: {root}")
    names = ", ".join(path.name for path in target_dirs)
    raise ValueError(f"Multiple canonical targets found ({names}); pass --canonical-target.")


def discover_cameras(
    target_dir: Path, model: str, include_omni_radtan: bool
) -> list[CanonicalCamera]:
    spec = MODEL_SPECS[model]
    candidate_paths = list((target_dir / spec.canonical_group).glob("*.yaml"))
    if model == "omni-none" and include_omni_radtan:
        candidate_paths.extend(target_dir.glob("*omni-radtan*.yaml"))
    cameras: list[CanonicalCamera] = []
    for path in sorted(set(candidate_paths)):
        camera_model, distortion_model = parse_camera_yaml(path)
        compatible = (camera_model, distortion_model) in spec.compatible_models
        if model == "omni-none" and include_omni_radtan:
            compatible = compatible or (camera_model, distortion_model) == (
                "omni",
                "radtan",
            )
        if not compatible:
            continue
        method, label = method_from_filename(path)
        if distortion_model == "radtan":
            label += " [radtan]"
        cameras.append(CanonicalCamera(method, label, path, camera_model, distortion_model))
    if not cameras:
        raise ValueError(
            f"No compatible canonical YAML files for {model} in {target_dir}."
        )
    cameras.sort(key=lambda camera: (METHOD_ORDER[camera.method], camera.path.name))
    if not any(camera.method == "kalibr" for camera in cameras):
        raise ValueError(f"{model} requires a Kalibr canonical YAML for --kalibr-camchain.")
    return cameras


def parse_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def inlier_rates(
    points_csv: Path, threshold_px: float
) -> dict[str, tuple[int, int, float]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with points_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            residual = float(row["residual_norm"])
            if not math.isfinite(residual):
                continue
            counts[row["method"]][1] += 1
            if residual <= threshold_px:
                counts[row["method"]][0] += 1
    return {
        method: (inliers, total, 100.0 * inliers / total)
        for method, (inliers, total) in counts.items()
        if total > 0
    }


def metric_prefix(method: str) -> str:
    return "kalibr" if method == "kalibr" else f"reference_{method}"


def parse_canonical_metadata(path: Path) -> tuple[list[int], str]:
    resolution: list[int] = []
    rostopic = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        resolution_match = re.match(r"\s*resolution:\s*\[([^\]]+)\]", line)
        if resolution_match:
            resolution = [
                int(value.strip()) for value in resolution_match.group(1).split(",")
            ]
        topic_match = re.match(r'\s*rostopic:\s*"?([^"]*)"?\s*$', line)
        if topic_match:
            rostopic = topic_match.group(1).strip()
    if len(resolution) != 2:
        raise ValueError(f"Canonical YAML lacks a valid resolution: {path}")
    return resolution, rostopic


def summary_camera_values(
    summary: dict[str, str], labels_key: str, values_key: str
) -> tuple[list[str], list[str]]:
    labels = [value.strip() for value in summary.get(labels_key, "").split(",") if value.strip()]
    values = [value.strip() for value in summary.get(values_key, "").split(",") if value.strip()]
    if len(labels) != len(values):
        raise ValueError(
            f"Mismatched camera labels and values in {labels_key}/{values_key}."
        )
    return labels, values


def yaml_list(values: list[str]) -> str:
    return "[" + ", ".join(values) + "]"


def save_current_baseline_camera(
    result_dir: Path,
    canonical_camera: CanonicalCamera,
    destination: Path,
    image: Path,
    model: str,
) -> None:
    summary = parse_summary(result_dir / "backend_holdout_summary.txt")
    if summary.get("success") != "1":
        raise ValueError(f"Cannot save failed Stage5 run: {result_dir}")
    intrinsics_labels, intrinsics = summary_camera_values(
        summary, "camera_intrinsics_labels", "camera_intrinsics_csv"
    )
    distortion_labels, distortion = summary_camera_values(
        summary, "camera_distortion_labels", "camera_distortion_csv"
    )
    if not intrinsics_labels:
        raise ValueError(f"Missing final intrinsics in {result_dir}")
    camera_model = summary["camera_model"]
    distortion_model = summary["camera_distortion_model"]
    if distortion_model == "equi":
        distortion_model = "equidistant"
    resolution, rostopic = parse_canonical_metadata(canonical_camera.path)
    relative_result = result_dir.relative_to(ROOT)
    source_image = image.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Current Stage5 Spherical baseline camera.",
        f"# Generated from: {relative_result}",
        f"# Image input: {source_image}",
        f"# Requested model family: {model}",
        "# Residual model: sphere_angular (tangent-plane component-wise angular residual).",
        "# This camera is the final persistent incremental-selection BA state.",
        "# Calibration uses every input frame; no train/test split is applied.",
        "# Metrics use a frozen frontend prepass on the same input frames and are self-evaluation only.",
        "# Self-evaluation metrics: RMSE={} px, P95={} px.".format(
            summary["overall_rmse"], summary["p95_reprojection_error"]
        ),
        "cam0:",
        "  cam_overlaps: []",
        f"  camera_model: {camera_model}",
        f"  distortion_model: {distortion_model}",
        f"  intrinsics: {yaml_list(intrinsics)}",
        f"  distortion_coeffs: {yaml_list(distortion)}",
        f"  resolution: {yaml_list([str(value) for value in resolution])}",
        f"  rostopic: {json.dumps(rostopic)}",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def read_model_rows(
    model: str,
    result_dir: Path,
    references: list[CanonicalCamera],
    threshold_px: float,
    current_baseline_yaml: Path | None,
) -> list[dict[str, object]]:
    summary = parse_summary(result_dir / "benchmark_holdout_summary.txt")
    rates = inlier_rates(result_dir / "benchmark_holdout_points.csv", threshold_px)
    method_specs: list[tuple[str, str, str, CanonicalCamera | None]] = [
        ("ours", "Ours", "our", None)
    ]
    method_specs.extend(
        (camera.method, camera.label, metric_prefix(camera.method), camera)
        for camera in references
    )
    rows: list[dict[str, object]] = []
    for method, label, prefix, camera in method_specs:
        point_method = "ours" if method == "ours" else method
        if point_method not in rates:
            raise ValueError(
                f"Missing point-wise evaluation for {model} / {method} in {result_dir}."
            )
        inliers, points, ratio = rates[point_method]
        rows.append(
            {
                "model": MODEL_SPECS[model].label,
                "method": label,
                "method_key": method,
                "camera_model": "ours" if camera is None else camera.camera_model,
                "distortion_model": "ours" if camera is None else camera.distortion_model,
                "canonical_yaml": ""
                if camera is None
                else str(camera.path.relative_to(ROOT)),
                "ours_parameter_yaml": ""
                if current_baseline_yaml is None
                else str(current_baseline_yaml.relative_to(ROOT)),
                "rmse_px": float(summary[f"{prefix}_overall_rmse"]),
                "p95_px": float(summary[f"{prefix}_p95_reprojection_error"]),
                "inlier_at_threshold_percent": ratio,
                "inlier_count": inliers,
                "point_count": points,
                "result_dir": str(result_dir.relative_to(ROOT)),
            }
        )
    return rows


def best_values(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        "rmse_px": min(float(row["rmse_px"]) for row in rows),
        "p95_px": min(float(row["p95_px"]) for row in rows),
        "inlier_at_threshold_percent": max(
            float(row["inlier_at_threshold_percent"]) for row in rows
        ),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_best(value: float, best: float) -> bool:
    return math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-12)


def bold(value: str, condition: bool) -> str:
    return f"**{value}**" if condition else value


def write_markdown(
    rows_by_model: dict[str, list[dict[str, object]]], path: Path, threshold: float
) -> None:
    lines = [
        "# Canonical Spherical Baseline Comparison (All-Training Self-Evaluation)",
        "",
        f"Inl.@{threshold:g}px is computed from the same frozen self-evaluation point CSV for every method.",
        "",
        "| Model | Method | RMSE [px] | P95 [px] | Inl. [%] | Canonical YAML |",
        "|---|---|---:|---:|---:|---|",
    ]
    for model, rows in rows_by_model.items():
        best = best_values(rows)
        for row in rows:
            rmse = float(row["rmse_px"])
            p95 = float(row["p95_px"])
            inlier = float(row["inlier_at_threshold_percent"])
            lines.append(
                "| {model} | {method} | {rmse} | {p95} | {inlier} | {yaml} |".format(
                    model=model,
                    method=row["method"],
                    rmse=bold(f"{rmse:.3f}", is_best(rmse, best["rmse_px"])),
                    p95=bold(f"{p95:.3f}", is_best(p95, best["p95_px"])),
                    inlier=bold(
                        f"{inlier:.2f}",
                        is_best(inlier, best["inlier_at_threshold_percent"]),
                    ),
                    yaml=row["canonical_yaml"] or "--",
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def write_latex(
    rows_by_model: dict[str, list[dict[str, object]]], path: Path, threshold: float
) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Canonical-baseline comparison with Spherical BA using all input frames for calibration. "
        f"Inl. is the percentage of same-set self-evaluation points below {threshold:.2f} px. "
        "Best values within each camera model are bold.}",
        "  \\label{tab:canonical_spherical_baselines}",
        "  \\setlength{\\tabcolsep}{4.0pt}",
        "  \\renewcommand{\\arraystretch}{1.03}",
        "  \\resizebox{\\columnwidth}{!}{%",
        "  \\begin{tabular}{llccc}",
        "    \\toprule",
        "    Model & Method & RMSE $\\downarrow$ & P95 $\\downarrow$ "
        f"& Inl.@{threshold:g}px [\\%] $\\uparrow$ \\\\",
        "    \\midrule",
    ]
    groups = list(rows_by_model.items())
    for group_index, (model, rows) in enumerate(groups):
        best = best_values(rows)
        for index, row in enumerate(rows):
            model_cell = (
                f"\\multirow{{{len(rows)}}}{{*}}{{{latex_escape(model)}}}"
                if index == 0
                else ""
            )
            method = latex_escape(str(row["method"]))
            if row["method_key"] == "ours":
                method = f"\\textbf{{{method}}}"
            values = []
            for key, decimals in (
                ("rmse_px", 3),
                ("p95_px", 3),
                ("inlier_at_threshold_percent", 2),
            ):
                value = float(row[key])
                text = f"{value:.{decimals}f}"
                if is_best(value, best[key]):
                    text = f"\\textbf{{{text}}}"
                values.append(text)
            lines.append(f"    {model_cell} & {method} & " + " & ".join(values) + " \\\\")
        if group_index + 1 < len(groups):
            lines.append("    \\midrule")
    lines.extend(["    \\bottomrule", "  \\end{tabular}%", "  }", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(command: list[str], dry_run: bool) -> None:
    print("+ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    models = normalize_models(args.models)
    image = resolve_repo_path(args.image)
    config = resolve_repo_path(args.config)
    canonical_root = resolve_repo_path(args.canonical_root)
    if not image.is_dir():
        raise ValueError(f"Image directory does not exist: {image}")
    if not config.is_file():
        raise ValueError(f"Config does not exist: {config}")
    target_dir = resolve_target_dir(canonical_root, args.canonical_target)
    slug = re.sub(r"[^a-z0-9_]+", "_", f"{image.parent.name}_{image.name}".lower())
    output_base = (
        resolve_repo_path(args.output)
        if args.output
        else ROOT / "result_may" / f"stage5_canonical_sphere_{slug}"
    )
    timestamp = args.run_timestamp or datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", timestamp):
        raise ValueError("--run-timestamp must use YYYYMMDD_HHMMSS.")
    output_root = output_base.parent / f"{output_base.name}_{timestamp}"
    cache_root = (
        resolve_repo_path(args.cache_dir)
        if args.cache_dir
        else ROOT / "result" / f".stage5_canonical_sphere_{slug}_cache"
    )
    current_baseline_root = resolve_repo_path(args.current_baseline_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise ValueError(
            f"Output exists: {output_root}. Pass --overwrite or choose --output."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    binary = ROOT / "build/run_stage5_backend"
    if not args.skip_build:
        run_command(
            ["cmake", "--build", "build", "--target", "run_stage5_backend", "-j", "8"],
            args.dry_run,
        )
    if not args.dry_run and not binary.is_file():
        raise ValueError(f"Missing backend binary: {binary}. Do not pass --skip-build.")

    all_rows: list[dict[str, object]] = []
    rows_by_model: dict[str, list[dict[str, object]]] = {}
    manifest: dict[str, object] = {
        "image": str(image),
        "models": models,
        "canonical_root": str(canonical_root),
        "canonical_target": str(target_dir),
        "output_base": str(output_base),
        "run_timestamp": timestamp,
        "residual_model": "sphere_angular",
        "calibration_scope": "all_input_frames",
        "evaluation_scope": "same_input_frozen_frontend_prepass",
        "inlier_threshold_px": args.inlier_threshold_px,
        "current_baseline_root": str(current_baseline_root),
        "runs": [],
    }

    for model in models:
        references = discover_cameras(
            target_dir, model, args.include_omni_radtan_references
        )
        kalibr = next(camera for camera in references if camera.method == "kalibr")
        result_dir = output_root / model
        model_folder = {
            "ds-none": "ds",
            "pinhole-equi": "kb",
            "omni-none": "omni",
            "eucm-none": "eucm",
        }[model]
        current_baseline_yaml = (
            current_baseline_root
            / image.parent.name
            / model_folder
            / f"{image.parent.name}__{image.name}__ours-spherical__{model_folder}.yaml"
        )
        command = [
            str(binary),
            "--config",
            str(config),
            "--runtime-mode",
            "research",
            "--all",
            "--stage5-disable-selected-case-visualizations",
            "--stage5-enable-polar-angle-diagnostics",
            "--image",
            str(image),
            "--test-image",
            str(image),
            "--stage5-external-holdout-self-frontend-prepass",
            "--models",
            model,
            "--kalibr-camchain",
            str(kalibr.path),
            "--backend-residual-model",
            "sphere_angular",
            "--output",
            str(result_dir),
            "--cache-dir",
            str(cache_root / model),
        ]
        for camera in references:
            if camera.method != "kalibr":
                command.extend(
                    ["--reference-intrinsics-yaml", f"{camera.method}:{camera.path}"]
                )
        manifest["runs"].append(
            {
                "model": model,
                "result_dir": str(result_dir),
                "kalibr_camchain": str(kalibr.path),
                "references": [str(camera.path) for camera in references],
                "current_baseline_yaml": str(current_baseline_yaml),
                "command": command,
            }
        )
        run_command(command, args.dry_run)
        if args.dry_run:
            continue
        if args.save_current_baseline:
            save_current_baseline_camera(
                result_dir,
                kalibr,
                current_baseline_yaml,
                image,
                model,
            )
        rows = read_model_rows(
            model,
            result_dir,
            references,
            args.inlier_threshold_px,
            current_baseline_yaml if args.save_current_baseline else None,
        )
        rows_by_model[MODEL_SPECS[model].label] = rows
        all_rows.extend(rows)

    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(f"Dry run complete. Manifest: {output_root / 'run_manifest.json'}")
        return 0
    write_csv(all_rows, output_root / "canonical_spherical_table.csv")
    write_markdown(
        rows_by_model, output_root / "canonical_spherical_table.md", args.inlier_threshold_px
    )
    write_latex(
        rows_by_model, output_root / "canonical_spherical_table.tex", args.inlier_threshold_px
    )
    print(f"Table CSV: {output_root / 'canonical_spherical_table.csv'}")
    print(f"Table Markdown: {output_root / 'canonical_spherical_table.md'}")
    print(f"Table LaTeX: {output_root / 'canonical_spherical_table.tex'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

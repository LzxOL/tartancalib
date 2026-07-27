#!/usr/bin/env python3
"""Create an auditable, inseparable stereo calibration bundle.

The three YAML inputs are copied together and accompanied by checksums and the
exact Stage6/data provenance.  This prevents downstream evaluation from
accidentally combining an extrinsic from one run with intrinsics from another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def require_existing(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        fail(f"missing {label}: {resolved}")
    return resolved


def yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+)", text)
    if not match:
        fail(f"missing '{key}' in stereo extrinsic YAML")
    return match.group(1).strip()


def parse_positive_int(values: dict[str, str], key: str) -> int:
    try:
        value = int(float(values[key]))
    except (KeyError, ValueError):
        fail(f"missing/invalid '{key}' in Stage6 pairing summary")
    if value <= 0:
        fail(f"'{key}' must be positive, got {value}")
    return value


def parse_nonnegative_int(values: dict[str, str], key: str) -> int:
    try:
        value = int(float(values[key]))
    except (KeyError, ValueError):
        fail(f"missing/invalid '{key}' in Stage6 pairing summary")
    if value < 0:
        fail(f"'{key}' must be non-negative, got {value}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage6-output", type=Path, required=True)
    parser.add_argument("--left-intrinsics", type=Path, required=True)
    parser.add_argument("--right-intrinsics", type=Path, required=True)
    parser.add_argument("--training-left-dir", type=Path, required=True)
    parser.add_argument("--training-right-dir", type=Path, required=True)
    parser.add_argument("--holdout-left-dir", type=Path, required=True)
    parser.add_argument("--holdout-right-dir", type=Path, required=True)
    parser.add_argument("--max-pair-delta-ms", type=float, default=1.0)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.max_pair_delta_ms < 0.0:
        fail("--max-pair-delta-ms must be non-negative")
    stage6 = args.stage6_output.resolve()
    if not stage6.is_dir():
        fail(f"missing Stage6 output directory: {stage6}")
    left = require_existing(args.left_intrinsics, "left intrinsics")
    right = require_existing(args.right_intrinsics, "right intrinsics")
    extrinsic = require_existing(stage6 / "stereo_extrinsic.yaml", "Stage6 stereo extrinsic")
    pairing = parse_key_values(stage6 / "stereo_pairing_summary.txt")
    extrinsic_summary = parse_key_values(stage6 / "stereo_extrinsic_summary.txt")
    reprojection = parse_key_values(stage6 / "stereo_reprojection_summary.txt")
    training_pair_count = parse_positive_int(extrinsic_summary, "reachable_training_pair_count")
    holdout_pair_count = parse_positive_int(reprojection, "holdout_extrinsic_only_pair_count")
    max_delta_ns = parse_nonnegative_int(pairing, "max_pair_timestamp_delta_ns")
    allowed_delta_ns = int(round(args.max_pair_delta_ms * 1e6))
    if max_delta_ns > allowed_delta_ns:
        fail(f"Stage6 max timestamp delta {max_delta_ns} ns exceeds {allowed_delta_ns} ns")
    if pairing.get("pairing_mode", "") not in {"exact_timestamp", "filename_timestamp_exact"}:
        fail(f"unexpected Stage6 pairing mode: {pairing.get('pairing_mode', '')!r}")

    extrinsic_text = extrinsic.read_text(encoding="utf-8")
    if yaml_scalar(extrinsic_text, "cam0_is_reference") != "1":
        fail("Stage6 extrinsic does not declare cam0_is_reference: 1")
    if not re.search(r"(?m)^\s*rotation_matrix:\s*$", extrinsic_text):
        fail("Stage6 extrinsic lacks rotation_matrix")
    if not re.search(r"(?m)^\s*translation_xyz:\s*\[[^\]]+\]", extrinsic_text):
        fail("Stage6 extrinsic lacks translation_xyz")

    for label, directory in {
        "training left directory": args.training_left_dir,
        "training right directory": args.training_right_dir,
        "holdout left directory": args.holdout_left_dir,
        "holdout right directory": args.holdout_right_dir,
    }.items():
        if not directory.resolve().is_dir():
            fail(f"missing {label}: {directory.resolve()}")

    bundle = args.bundle_dir.resolve()
    if bundle.exists() and any(bundle.iterdir()):
        fail(f"bundle directory already exists and is non-empty: {bundle}")
    bundle.mkdir(parents=True, exist_ok=True)
    copies = {
        "left_intrinsics.yaml": left,
        "right_intrinsics.yaml": right,
        "stereo_extrinsic.yaml": extrinsic,
    }
    for name, source in copies.items():
        shutil.copy2(source, bundle / name)

    manifest = {
        "bundle_schema_version": 1,
        "model": "ds-none",
        "coordinate_convention": "p_cam1 = R_cam1_from_cam0 * p_cam0 + t_cam1_from_cam0",
        "stage6_source_output": str(stage6),
        "files": {
            name: {
                "source": str(source),
                "sha256": sha256(bundle / name),
            }
            for name, source in copies.items()
        },
        "training": {
            "left_dir": str(args.training_left_dir.resolve()),
            "right_dir": str(args.training_right_dir.resolve()),
            "strict_timestamp_pair_count": training_pair_count,
            "max_pair_timestamp_delta_ns": max_delta_ns,
            "configured_max_pair_delta_ms": args.max_pair_delta_ms,
        },
        "holdout": {
            "left_dir": str(args.holdout_left_dir.resolve()),
            "right_dir": str(args.holdout_right_dir.resolve()),
            "strict_timestamp_pair_count": holdout_pair_count,
            "role": "external_validation_only",
            "excluded_from_stage6_optimization": True,
        },
        "stage6": {
            "intrinsics_mode": "fixed_stage5",
            "persistent_pose_structure": "independent_pair_board",
            "final_global_ba_skipped": True,
            "frame_pairing_mode": pairing.get("pairing_mode"),
        },
    }
    (bundle / "stereo_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"created provenance-locked bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

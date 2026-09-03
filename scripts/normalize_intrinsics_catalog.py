#!/usr/bin/env python3
"""Generate and validate the canonical intrinsic-parameter catalog."""

import argparse
import csv
import json
import math
from pathlib import Path


EXPECTED = {
    ("ds", "none"): (6, 0),
    ("eucm", "none"): (6, 0),
    ("pinhole", "equidistant"): (4, 4),
    ("omni", "none"): (5, 0),
    ("omni", "radtan"): (5, 4),
}


def yaml_scalar(value):
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        # repr(float) preserves enough digits for an exact binary64 round trip.
        return "[" + ", ".join(repr(item) if isinstance(item, float) else str(item) for item in value) + "]"
    return str(value)


def validate(entry, root):
    source = root / entry["source"]
    if not source.is_file():
        raise ValueError(f"{entry['id']}: missing source file {entry['source']}")
    key = (entry["camera_model"], entry["distortion_model"])
    if key not in EXPECTED:
        raise ValueError(f"{entry['id']}: unsupported model pair {key}")
    ni, nd = EXPECTED[key]
    if len(entry["intrinsics"]) != ni or len(entry["distortion_coeffs"]) != nd:
        raise ValueError(f"{entry['id']}: expected {ni} intrinsics and {nd} distortion coefficients")
    if entry["resolution"] != [4512, 4512]:
        raise ValueError(f"{entry['id']}: unexpected resolution {entry['resolution']}")
    values = entry["intrinsics"] + entry["distortion_coeffs"]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{entry['id']}: non-finite parameter")


def render(entry):
    lines = [
        f"# Canonical evaluation camchain. Generated from: {entry['source']}",
        "# Parameters are reference-only and must not initialize Stage5.",
        "cam0:",
        "  cam_overlaps: []",
        f"  camera_model: {entry['camera_model']}",
        f"  distortion_model: {entry['distortion_model']}",
        f"  intrinsics: {yaml_scalar(entry['intrinsics'])}",
        f"  distortion_coeffs: {yaml_scalar(entry['distortion_coeffs'])}",
        f"  resolution: {yaml_scalar(entry['resolution'])}",
        f"  rostopic: {yaml_scalar(entry['rostopic'])}",
    ]
    if entry.get("conversion_note"):
        lines.insert(2, f"# Conversion: {entry['conversion_note']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate without rewriting files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    catalog = root / "intrintic" / "catalog"
    manifest_path = catalog / "manifest.json"
    entries = json.loads(manifest_path.read_text())["entries"]
    rows = []
    expected_outputs = set()
    for entry in entries:
        validate(entry, root)
        canonical_subdir = entry.get("canonical_subdir", entry["camera"])
        output = catalog / "canonical" / canonical_subdir / entry["filename"]
        expected_outputs.add(output.resolve())
        expected = render(entry)
        if args.check:
            if not output.exists() or output.read_text() != expected:
                raise SystemExit(f"stale or missing: {output.relative_to(root)}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(expected)
        rows.append({
            "calibration_dataset": entry["calibration_dataset"],
            "camera": entry["camera"],
            "method": entry["method"],
            "model": entry["model"],
            "camera_model": entry["camera_model"],
            "distortion_model": entry["distortion_model"],
            "canonical_path": str(output.relative_to(root)),
            "source_path": entry["source"],
        })
    index_path = catalog / "catalog.csv"
    fieldnames = list(rows[0])
    if args.check:
        if not index_path.exists():
            raise SystemExit(f"missing: {index_path.relative_to(root)}")
    else:
        with index_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        for stale in (catalog / "canonical").rglob("*.yaml"):
            if stale.resolve() not in expected_outputs:
                stale.unlink()
    print(f"validated {len(entries)} canonical camera files")


if __name__ == "__main__":
    main()

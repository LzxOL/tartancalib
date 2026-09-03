#!/usr/bin/env python3
"""Derive a strictly paired Outer4-only BabelCalib export from a full export.

The output retains the full board geometry and original MATLAB correspondences;
only columns whose source point_type is not ``outer`` are removed.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def outer_mask_by_frame(full_export, split):
    """Build masks from the source rows, not global point ids.

    Some historical frontend outputs assign different source point ids to the
    same physical outer corner. The exporter preserves this per-observation
    correspondence in the MAT; filtering must therefore follow its exact row
    order for every frame rather than a global fiducial allowlist.
    """
    rows_by_frame = {}
    with (full_export / f"points_{split}.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["frame_index"]), row["frame_label"])
            rows_by_frame.setdefault(key, []).append(row)
    frames = read_jsonl(full_export / f"frames_{split}.jsonl")
    masks = []
    for frame in frames:
        key = (int(frame["frame_index"]), frame["frame_label"])
        rows = sorted(
            rows_by_frame[key],
            key=lambda row: (
                int(row["board_id"]), int(row["point_id"]), row["point_type"],
                float(row["observed_x"]), float(row["observed_y"]),
            ),
        )
        masks.append(np.array([row["point_type"] == "outer" for row in rows], dtype=bool))
    return masks


def filter_mat(source, destination, masks):
    # Keep NumPy structured arrays so savemat writes MATLAB structs, not cells.
    data = loadmat(source, struct_as_record=True, squeeze_me=False)
    corners = data["corners"]
    if corners.size != len(masks):
        raise RuntimeError(f"{source}: MAT/image metadata length mismatch")
    for corner, keep in zip(corners.ravel(), masks):
        if np.asarray(corner["x"]).shape[1] != keep.size:
            raise RuntimeError(f"{source}: MAT/source point order length mismatch")
        corner["x"] = np.asarray(corner["x"])[:, keep]
        corner["cspond"] = np.asarray(corner["cspond"])[:, keep]
    payload = {
        "corners": corners,
        "boards": data["boards"],
        "imgsize": data["imgsize"],
        "export_metadata_json": json.dumps(
            {
                "point_types": ["outer"],
                "derivation": "filtered from full export; boards and outer fiducial indices retained",
                "source_export": str(source.parent.resolve()),
            },
            sort_keys=True,
        ),
    }
    savemat(destination, payload, do_compression=True, long_field_names=True)


def write_outer_points(source, destination):
    with source.open(newline="") as fin, destination.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row.get("point_type") == "outer":
                writer.writerow(row)


def main():
    args = parse_args()
    source = args.full_export.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split in ("all", "train", "test", "backend"):
        masks = outer_mask_by_frame(source, split)
        filter_mat(source / f"{split}.mat", output / f"{split}.mat", masks)
        write_outer_points(source / f"points_{split}.csv", output / f"points_{split}.csv")
        frames = source / f"frames_{split}.jsonl"
        if frames.exists():
            (output / frames.name).write_text(frames.read_text())
    for name in ("fiducial_map.jsonl", "split.json", "failed_frames.txt"):
        (output / name).write_text((source / name).read_text())
    report = {
        "derivation": "strictly paired Outer4-only filter",
        "source_full_export": str(source),
        "point_types": ["outer"],
        "retains_full_boards_X": True,
        "retains_full_boards_Rt": True,
        "retains_full_outer_correspondence_indices": True,
        "filter_basis": "per-frame point_type in source CSV, matched to exporter row order",
    }
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a deterministic view subset of a BabelCalib MAT file."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("--ratio must be in (0, 1]")
    # Keep MATLAB struct arrays as structured ndarrays so downstream MAT
    # importers see corners/boards as real structs after savemat.
    data = loadmat(args.input, squeeze_me=False, struct_as_record=True)
    corners = data.get("corners")
    if corners is None or corners.ndim != 2 or corners.shape[0] != 1:
        raise ValueError(f"expected corners as 1xN MATLAB struct array, got {getattr(corners, 'shape', None)}")
    count = corners.shape[1]
    subset_count = max(1, min(count, int(round(count * args.ratio))))
    indices = list(range(count))
    random.Random(args.seed).shuffle(indices)
    selected = sorted(indices[:subset_count])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corners": corners[:, selected],
        "boards": data["boards"],
        "imgsize": data["imgsize"],
    }
    if "export_metadata_json" in data:
        payload["export_metadata_json"] = data["export_metadata_json"]
    savemat(output, payload, do_compression=True, long_field_names=True)
    output.with_suffix(".json").write_text(json.dumps({
        "source_mat": str(args.input.resolve()),
        "output_mat": str(output),
        "source_view_count": count,
        "selected_view_count": subset_count,
        "ratio": args.ratio,
        "seed": args.seed,
        "selected_view_indices_zero_based": selected,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "selected_view_count": subset_count, "indices": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

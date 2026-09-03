#!/usr/bin/env python3
"""Export reproducible prefixes of a persistent Backend candidate schedule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def fingerprint(keys: list[tuple[int, int]]) -> str:
    payload = json.dumps(keys, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--prefix-batches", default="1,2,4,8")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    requested = sorted({int(value) for value in args.prefix_batches.split(",")})
    if not requested or requested[0] <= 0:
        raise SystemExit("--prefix-batches must contain positive integers")

    batches: dict[int, set[tuple[int, int]]] = {}
    labels: dict[int, set[str]] = {}
    with args.decisions.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_order = row.get("persistent_incremental_attempt_order", "")
            if not raw_order or int(raw_order) < 0:
                continue
            order = int(raw_order)
            batches.setdefault(order, set()).add(
                (int(row["frame_index"]), int(row["board_id"]))
            )
            labels.setdefault(order, set()).add(row.get("frame_label", ""))

    ordered_batches = sorted(batches)
    if len(ordered_batches) < requested[-1]:
        raise RuntimeError(
            f"schedule has {len(ordered_batches)} batches, need {requested[-1]}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source_decisions": str(args.decisions.resolve()),
        "available_batch_count": len(ordered_batches),
        "prefixes": [],
    }
    for count in requested:
        selected_orders = ordered_batches[:count]
        keys = sorted({key for order in selected_orders for key in batches[order]})
        path = output_dir / f"prefix_{count:02d}_batches.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "board_id"])
            writer.writerows(keys)
        manifest["prefixes"].append(
            {
                "batch_count": count,
                "attempt_orders": selected_orders,
                "frame_board_count": len(keys),
                "frame_count": len({frame for frame, _ in keys}),
                "frame_labels": sorted(
                    {label for order in selected_orders for label in labels[order]}
                ),
                "frame_board_set_fingerprint": fingerprint(keys),
                "path": str(path),
            }
        )

    manifest_path = output_dir / "prefix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

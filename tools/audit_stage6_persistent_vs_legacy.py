#!/usr/bin/env python3
"""Audit Stage6 persistent-vs-legacy experiment summary tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


Row = Dict[str, str]


def to_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_board_distribution(value: str) -> Dict[int, int]:
    result: Dict[int, int] = {}
    if not value:
        return result
    for token in value.split(","):
        if ":" not in token:
            continue
        board_text, count_text = token.split(":", 1)
        try:
            result[int(board_text)] = int(float(count_text))
        except ValueError:
            continue
    return result


def split_key(directory_name: str) -> str:
    if "1444190clear_to_144928clear" in directory_name:
        return "1444190clear_to_144928clear"
    if "144928clear_to_1444190clear" in directory_name:
        return "144928clear_to_1444190clear"
    if "144419_to_144928" in directory_name:
        return "144419_to_144928"
    if "144928_to_144419" in directory_name:
        return "144928_to_144419"
    return directory_name


def method_key(directory_name: str) -> str:
    if "persistent" in directory_name:
        return "persistent"
    if "legacy" in directory_name:
        return "legacy"
    return "other"


def load_rows(path: Path) -> List[Row]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def add_result(results: List[Tuple[str, str]], status: str, message: str) -> None:
    results.append((status, message))


def audit_pair(split: str, persistent: Row, legacy: Row, rmse_tolerance: float,
               results: List[Tuple[str, str]]) -> None:
    persistent_rmse = to_float(
        persistent.get("holdout_extrinsic_only_total_stereo_rmse", "")
    )
    legacy_rmse = to_float(legacy.get("holdout_extrinsic_only_total_stereo_rmse", ""))
    reference_rmse = to_float(
        persistent.get("reference_extrinsic_only_holdout_total_stereo_rmse", "")
    )
    persistent_boards = parse_board_distribution(
        persistent.get("persistent_selected_board_distribution", "")
    )
    legacy_boards = parse_board_distribution(
        legacy.get("persistent_selected_board_distribution", "")
    )

    if persistent_rmse is None or legacy_rmse is None:
        add_result(results, "FAIL", f"{split}: missing persistent/legacy RMSE")
    else:
        delta = persistent_rmse - legacy_rmse
        if delta <= rmse_tolerance:
            add_result(
                results,
                "PASS",
                f"{split}: persistent RMSE within tolerance of legacy "
                f"({persistent_rmse:.6f} vs {legacy_rmse:.6f}, delta={delta:.6f})",
            )
        else:
            add_result(
                results,
                "FAIL",
                f"{split}: persistent RMSE regressed beyond tolerance "
                f"({persistent_rmse:.6f} vs {legacy_rmse:.6f}, delta={delta:.6f})",
            )

    if persistent_rmse is None or reference_rmse is None:
        add_result(results, "FAIL", f"{split}: missing persistent/reference RMSE")
    elif persistent_rmse < reference_rmse:
        add_result(
            results,
            "PASS",
            f"{split}: persistent beats reference "
            f"({persistent_rmse:.6f} vs {reference_rmse:.6f})",
        )
    else:
        add_result(
            results,
            "FAIL",
            f"{split}: persistent does not beat reference "
            f"({persistent_rmse:.6f} vs {reference_rmse:.6f})",
        )

    if not persistent_boards or not legacy_boards:
        add_result(results, "FAIL", f"{split}: missing board distributions")
        return

    persistent_min = min(persistent_boards.values())
    legacy_min = min(legacy_boards.values())
    if persistent_min >= legacy_min:
        add_result(
            results,
            "PASS",
            f"{split}: persistent min board coverage >= legacy "
            f"({persistent_min} vs {legacy_min})",
        )
    else:
        add_result(
            results,
            "FAIL",
            f"{split}: persistent min board coverage lower than legacy "
            f"({persistent_min} vs {legacy_min})",
        )

    persistent_total = sum(persistent_boards.values())
    legacy_total = sum(legacy_boards.values())
    if persistent_total >= legacy_total:
        add_result(
            results,
            "PASS",
            f"{split}: persistent pair-board count >= legacy "
            f"({persistent_total} vs {legacy_total})",
        )
    else:
        add_result(
            results,
            "FAIL",
            f"{split}: persistent pair-board count lower than legacy "
            f"({persistent_total} vs {legacy_total})",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Stage6 persistent-vs-legacy summary CSV."
    )
    parser.add_argument("summary_csv", help="stage6_stereo_experiment_summary.csv")
    parser.add_argument(
        "--rmse-tolerance",
        type=float,
        default=0.05,
        help="Allowed persistent-minus-legacy extrinsic-only RMSE delta.",
    )
    parser.add_argument("--output", help="Optional text report path.")
    args = parser.parse_args()

    rows = load_rows(Path(args.summary_csv))
    grouped: Dict[str, Dict[str, Row]] = {}
    for row in rows:
        directory_name = row.get("directory_name", "")
        grouped.setdefault(split_key(directory_name), {})[
            method_key(directory_name)
        ] = row

    results: List[Tuple[str, str]] = []
    for split in sorted(grouped):
        methods = grouped[split]
        if "persistent" not in methods or "legacy" not in methods:
            add_result(
                results,
                "FAIL",
                f"{split}: missing persistent or legacy row "
                f"(methods={sorted(methods)})",
            )
            continue
        audit_pair(
            split,
            methods["persistent"],
            methods["legacy"],
            args.rmse_tolerance,
            results,
        )

    success = not any(status == "FAIL" for status, _ in results)
    lines = [f"success: {1 if success else 0}"]
    lines.extend(f"{status}: {message}" for status, message in results)
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

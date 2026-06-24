#!/usr/bin/env python3
"""Combine Stage6 ours/Kalibr holdout visualization PNGs.

The Stage6 exporter writes ours and reference/Kalibr overlays into sibling
directories with matching filenames. This script stacks matching images into a
single comparison figure and places the aggregate RMSE values in the output
filename and title bars.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def rmse_token(value: str) -> str:
    try:
        return f"{float(value):.5f}".replace(".", "p")
    except ValueError:
        return value.replace(".", "p").replace("/", "_")


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    height, width = image.shape[:2]
    bar_height = max(52, height // 28)
    titled = np.full((height + bar_height, width, 3), 245, dtype=np.uint8)
    titled[bar_height:, :, :] = image
    cv2.putText(
        titled,
        title,
        (18, min(bar_height - 15, 38)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return titled


def pad_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    out = np.full((image.shape[0], width, 3), 255, dtype=np.uint8)
    out[:, : image.shape[1], :] = image
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument(
        "--output-subdir",
        default="stereo_reprojection_visualizations/combined_ours_vs_kalibr_rmse",
    )
    args = parser.parse_args()

    result_dir: Path = args.result_dir
    summary = read_summary(result_dir / "stereo_reference_holdout_summary.txt")
    ours_rmse = summary.get("ours_extrinsic_only_holdout_total_stereo_rmse", "nan")
    kalibr_rmse = summary.get(
        "reference_extrinsic_only_holdout_total_stereo_rmse", "nan"
    )
    ours_token = rmse_token(ours_rmse)
    kalibr_token = rmse_token(kalibr_rmse)

    viz_root = result_dir / "stereo_reprojection_visualizations"
    ours_dir = viz_root / "ours_holdout_extrinsic_only_top_bad_pair_boards"
    kalibr_dir = viz_root / "reference_holdout_extrinsic_only_top_bad_pair_boards"
    output_dir = result_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    count = 0
    for ours_path in sorted(ours_dir.glob("*.png")):
        kalibr_path = kalibr_dir / ours_path.name
        if not kalibr_path.exists():
            continue
        ours_img = cv2.imread(str(ours_path), cv2.IMREAD_COLOR)
        kalibr_img = cv2.imread(str(kalibr_path), cv2.IMREAD_COLOR)
        if ours_img is None or kalibr_img is None:
            continue

        width = max(ours_img.shape[1], kalibr_img.shape[1])
        ours_titled = pad_to_width(
            add_title(ours_img, f"Ours extrinsic-only holdout RMSE = {ours_rmse}"),
            width,
        )
        kalibr_titled = pad_to_width(
            add_title(
                kalibr_img,
                f"Kalibr/reference extrinsic-only holdout RMSE = {kalibr_rmse}",
            ),
            width,
        )
        divider = np.full((10, width, 3), 35, dtype=np.uint8)
        combined = np.vstack([ours_titled, divider, kalibr_titled])

        output_name = (
            f"ours_rmse_{ours_token}_kalibr_rmse_{kalibr_token}_{ours_path.name}"
        )
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), combined)
        rows.append(
            {
                "combined_png": str(output_path),
                "ours_png": str(ours_path),
                "kalibr_png": str(kalibr_path),
                "ours_extrinsic_only_rmse": ours_rmse,
                "kalibr_extrinsic_only_rmse": kalibr_rmse,
            }
        )
        count += 1

    with (output_dir / "combined_index.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "combined_png",
            "ours_png",
            "kalibr_png",
            "ours_extrinsic_only_rmse",
            "kalibr_extrinsic_only_rmse",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"combined_count={count}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

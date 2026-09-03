#!/usr/bin/env python3
"""Render holdout internal-point diagnostics in Stage5 filter-overlay style.

Holdout observations are intentionally not passed through the training
pre-backend filter.  This tool is therefore read-only: it visualizes the
actual holdout observations and flags large final-evaluation residuals without
changing which points participated in calibration.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-index", type=int, action="append", required=True)
    parser.add_argument("--board-id", type=int, default=5)
    parser.add_argument("--warn-residual-px", type=float, default=3.0)
    parser.add_argument("--bad-residual-px", type=float, default=10.0)
    return parser.parse_args()


def image_for(image_dir: Path, label: str) -> Path:
    candidates = [image_dir / label]
    candidates.extend(image_dir / f"{label}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(image_dir.glob(f"{label}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"image not found for {label}")


def as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    args = parse_args()
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV Python module is required") from exc

    target_frames = set(args.frame_index)
    selected: dict[int, list[dict[str, str]]] = defaultdict(list)
    with open(args.points_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "backend" or row.get("split") != "holdout":
                continue
            if as_int(row, "frame_index") not in target_frames:
                continue
            if as_int(row, "board_id") != args.board_id:
                continue
            selected[as_int(row, "frame_index")].append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for frame_index in sorted(target_frames):
        rows = selected.get(frame_index, [])
        if not rows:
            raise RuntimeError(f"no backend holdout points for frame={frame_index}, board={args.board_id}")
        label = rows[0]["frame_label"]
        image_path = image_for(Path(args.image_dir), label)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {image_path}")

        outer_rows = [row for row in rows if row["point_type"] == "outer"]
        internal_rows = [row for row in rows if row["point_type"] == "internal"]
        warn_count = 0
        bad_count = 0
        for row in rows:
            ox, oy = as_float(row, "observed_x"), as_float(row, "observed_y")
            px, py = as_float(row, "predicted_x"), as_float(row, "predicted_y")
            point = (round(ox), round(oy))
            predicted = (round(px), round(py))
            if row["point_type"] == "outer":
                cv2.drawMarker(image, point, (255, 220, 0), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
                cv2.drawMarker(image, predicted, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 13, 1, cv2.LINE_AA)
                cv2.line(image, predicted, point, (0, 220, 255), 1, cv2.LINE_AA)
                continue

            residual = as_float(row, "residual_norm")
            if residual >= args.bad_residual_px:
                color = (0, 0, 255)  # red: severe diagnostic flag
                bad_count += 1
            elif residual >= args.warn_residual_px:
                color = (0, 165, 255)  # orange: warning diagnostic flag
                warn_count += 1
            else:
                color = (60, 230, 60)  # green: low final residual
            cv2.circle(image, point, 6, color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(image, point, 8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(image, predicted, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)
            cv2.line(image, predicted, point, (0, 220, 255), 1, cv2.LINE_AA)
            point_id = row["point_id"]
            cv2.putText(image, f"{point_id}:{residual:.1f}", (point[0] + 9, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        panel_width = min(image.shape[1] - 20, 1530)
        cv2.rectangle(image, (10, 10), (panel_width, 133), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, f"Holdout internal-point filter diagnostic | frame {frame_index} | Board {args.board_id}",
                    (22, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "cyan cross=Outer4 observation  green=internal <3px  orange=3-10px  red=>=10px",
                    (22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "magenta x=final model projection  yellow line=residual  holdout: read-only, no point was filtered",
                    (22, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"outer={len(outer_rows)} internal={len(internal_rows)} warn={warn_count} severe={bad_count}",
                    (22, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        filename = f"frame_{frame_index:06d}_{label}_board_{args.board_id}_holdout_internal_filter_diagnostic.png"
        output_path = output_dir / filename
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"failed to write {output_path}")
        summary_rows.append({
            "frame_index": frame_index,
            "frame_label": label,
            "board_id": args.board_id,
            "outer_count": len(outer_rows),
            "internal_count": len(internal_rows),
            "warn_count": warn_count,
            "severe_count": bad_count,
            "image_file": filename,
        })

    with open(output_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {len(summary_rows)} overlays to {output_dir}")


if __name__ == "__main__":
    main()

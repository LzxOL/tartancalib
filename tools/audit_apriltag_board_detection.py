#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_bool_value(text):
    value = text.strip().lower()
    if value == "yes":
        return "yes"
    if value == "no":
        return "no"
    return text.strip()


def parse_log(image_path, text):
    rows = []
    current = None
    overlay = ""

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("combined overlay:"):
            overlay = stripped.split(":", 1)[1].strip()
            continue

        match = re.match(r"Board\s+(\d+)$", stripped)
        if match:
            if current is not None:
                current["combined_overlay"] = overlay
                rows.append(current)
            current = {
                "image": image_path.name,
                "image_path": str(image_path),
                "board_id": int(match.group(1)),
                "tag_detected": "",
                "valid_observation": "",
                "outer_success": "",
                "outer_failure_reason": "",
                "local_patch_rescue_attempted": "",
                "local_patch_rescue_used": "",
                "local_patch_rescue_summary": "",
                "valid_corners": 0,
                "valid_internal_points": 0,
                "raw_detection_debug": "",
            }
            continue

        if current is None:
            continue

        if stripped.startswith("tag detected:"):
            current["tag_detected"] = parse_bool_value(stripped.split(":", 1)[1])
        elif stripped.startswith("valid observation:"):
            current["valid_observation"] = parse_bool_value(stripped.split(":", 1)[1])
        elif stripped.startswith("outer wrapper success:"):
            current["outer_success"] = parse_bool_value(stripped.split(":", 1)[1])
        elif stripped.startswith("outer failure reason:"):
            current["outer_failure_reason"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("local patch rescue attempted:"):
            current["local_patch_rescue_attempted"] = parse_bool_value(stripped.split(":", 1)[1])
        elif stripped.startswith("local patch rescue used:"):
            current["local_patch_rescue_used"] = parse_bool_value(stripped.split(":", 1)[1])
        elif stripped.startswith("local patch rescue summary:"):
            current["local_patch_rescue_summary"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("valid corners:"):
            current["valid_corners"] = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("valid internal points:"):
            current["valid_internal_points"] = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("raw detections:"):
            raw = stripped.split(":", 1)[1].strip()
            if raw != "(none)":
                current["raw_detection_debug"] = raw

    if current is not None:
        current["combined_overlay"] = overlay
        rows.append(current)
    return rows


def is_suspect(row):
    if row["valid_observation"] != "yes":
        return True
    if row["outer_success"] != "yes":
        return True
    if row["local_patch_rescue_used"] == "yes":
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="./build/detect_apriltag_internal")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    log_dir = output_dir / "per_image_logs"
    suspect_dir = output_dir / "suspect_board_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    suspect_dir.mkdir(parents=True, exist_ok=True)

    images = [
        path for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if args.limit > 0:
        images = images[:args.limit]

    all_rows = []
    for index, image_path in enumerate(images, start=1):
        overlay_path = overlay_dir / f"{image_path.stem}_detected.png"
        cmd = [
            args.detector,
            "--image", str(image_path),
            "--config", args.config,
            "--output", str(overlay_path),
        ]
        proc = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_text = proc.stdout
        (log_dir / f"{image_path.stem}.log").write_text(log_text)
        rows = parse_log(image_path, log_text)
        for row in rows:
            row["detector_return_code"] = proc.returncode
            row["requested_overlay"] = str(overlay_path)
            all_rows.append(row)

        if index % 10 == 0 or index == len(images):
            print(f"processed {index}/{len(images)}: {image_path.name}", flush=True)

    fields = [
        "image", "image_path", "board_id",
        "tag_detected", "valid_observation", "outer_success",
        "outer_failure_reason", "local_patch_rescue_attempted",
        "local_patch_rescue_used", "local_patch_rescue_summary",
        "valid_corners", "valid_internal_points",
        "raw_detection_debug", "combined_overlay", "requested_overlay",
        "detector_return_code",
    ]
    csv_path = output_dir / "board_detection_audit.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    suspect_rows = [row for row in all_rows if is_suspect(row)]
    copied = set()
    for row in suspect_rows:
        overlay_path = Path(row.get("requested_overlay", ""))
        if overlay_path.exists() and overlay_path not in copied:
            shutil.copy2(overlay_path, suspect_dir / f"{overlay_path.stem}_suspect_overlay.png")
            copied.add(overlay_path)

    status_counter = Counter()
    for row in all_rows:
        key = (
            row["board_id"],
            row["tag_detected"],
            row["valid_observation"],
            row["outer_success"],
            row["outer_failure_reason"],
        )
        status_counter[key] += 1

    per_board_valid = defaultdict(int)
    per_board_invalid = defaultdict(int)
    for row in all_rows:
        if row["valid_observation"] == "yes" and row["outer_success"] == "yes":
            per_board_valid[row["board_id"]] += 1
        else:
            per_board_invalid[row["board_id"]] += 1

    summary_path = output_dir / "board_detection_audit_summary.txt"
    with summary_path.open("w") as output:
        output.write(f"images={len(images)}\n")
        output.write(f"board_rows={len(all_rows)}\n")
        output.write(f"suspect_board_rows={len(suspect_rows)}\n")
        output.write(f"suspect_overlay_images={len(copied)}\n\n")

        output.write("valid_count_by_board:\n")
        for board_id in sorted(set(per_board_valid) | set(per_board_invalid)):
            output.write(
                f"board={board_id} valid={per_board_valid[board_id]} "
                f"invalid={per_board_invalid[board_id]}\n")

        output.write("\nstatus_by_board:\n")
        for key, count in sorted(status_counter.items()):
            board_id, tag, valid, outer, reason = key
            output.write(
                f"board={board_id} tag_detected={tag} valid={valid} "
                f"outer={outer} reason={reason} count={count}\n")

        output.write("\nsuspect_rows:\n")
        for row in suspect_rows:
            output.write(
                f"{row['image']}, board={row['board_id']}, "
                f"tag={row['tag_detected']}, valid={row['valid_observation']}, "
                f"outer={row['outer_success']}, reason={row['outer_failure_reason']}, "
                f"corners={row['valid_corners']}, "
                f"internal={row['valid_internal_points']}, "
                f"rescue_used={row['local_patch_rescue_used']}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote suspect overlays: {suspect_dir}")


if __name__ == "__main__":
    main()

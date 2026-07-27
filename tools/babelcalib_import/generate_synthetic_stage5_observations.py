#!/usr/bin/env python3
"""Generate reproducible semi-synthetic Stage5 observations.

The board geometry and optimized training scene are taken from an existing
run, while image measurements are reprojected from one explicitly chosen DS
camera.  This removes detector/layout noise from a convergence-basin test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from run_ds_weak_mode_perturbation import parse_scene, project_ds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-training-dir", type=Path, required=True)
    parser.add_argument("--source-holdout-dir", type=Path, required=True)
    parser.add_argument("--reference-scene", type=Path, required=True)
    parser.add_argument("--output-training-dir", type=Path, required=True)
    parser.add_argument("--output-holdout-dir", type=Path, required=True)
    parser.add_argument("--noise-sigma-px", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.noise_sigma_px < 0.0:
        raise ValueError("--noise-sigma-px must be nonnegative")

    scene = parse_scene(args.reference_scene.resolve())
    rng = np.random.default_rng(args.seed)
    generated_points = 0
    invalid_points = 0

    def prepare(source: Path, output: Path, synthetic: bool) -> None:
        nonlocal generated_points, invalid_points
        output = output.resolve()
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        for source_file in source.iterdir():
            if source_file.name != "points.csv":
                shutil.copy2(source_file, output / source_file.name)
        with (source / "points.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if synthetic:
            for row in rows:
                frame_id = int(row["frame_index"])
                board_id = int(row["board_id"])
                if frame_id not in scene.frames or board_id not in scene.boards:
                    invalid_points += 1
                    continue
                point = np.asarray([float(row["target_x"]), float(row["target_y"]), float(row["target_z"])], dtype=np.float64)
                point_camera = (scene.frames[frame_id] @ scene.boards[board_id] @ np.r_[point, 1.0])[:3]
                pixels, valid = project_ds(scene.camera, point_camera[None, :])
                if not bool(valid[0]):
                    invalid_points += 1
                    continue
                noise = rng.normal(0.0, args.noise_sigma_px, size=2)
                row["observed_x"] = f"{pixels[0, 0] + noise[0]:.12f}"
                row["observed_y"] = f"{pixels[0, 1] + noise[1]:.12f}"
                row["quality"] = "1.0"
                generated_points += 1
        with (output / "points.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    prepare(args.source_training_dir, args.output_training_dir, True)
    # Holdout pose priors are not part of the training scene. Preserve the
    # real holdout measurements and use them only as an auxiliary metric.
    prepare(args.source_holdout_dir, args.output_holdout_dir, False)

    report = {
        "schema": "stage5_semi_synthetic_observations_v1",
        "reference_scene": str(args.reference_scene.resolve()),
        "reference_camera": [
            scene.camera.xi, scene.camera.alpha, scene.camera.fu,
            scene.camera.fv, scene.camera.cu, scene.camera.cv,
        ],
        "training_noise_sigma_px": args.noise_sigma_px,
        "noise_seed": args.seed,
        "generated_training_point_count": generated_points,
        "invalid_training_point_count": invalid_points,
        "holdout_measurements": "copied_from_real_dataset_for_auxiliary_evaluation",
    }
    payload = json.dumps(report, sort_keys=True).encode("utf-8")
    report["training_observation_fingerprint"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    output_root = args.output_training_dir.resolve().parent
    (output_root / "synthetic_generation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

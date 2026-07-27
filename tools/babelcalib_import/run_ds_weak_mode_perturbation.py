#!/usr/bin/env python3
"""Run paired DS recovery experiments along outer-only weak information modes.

The weak modes are computed from the clean reference scene and the exact
frame-board schedule used by the persistent backend.  A dense joint Jacobian
is built for DS intrinsics, shared frame poses, and non-reference board poses.
The pose/layout variables are marginalized with a Schur complement.  Each
weak mode is then scaled to a requested initial peripheral ray P95, so the
perturbation severity is comparable across directions.

Stage5 itself is not modified by this tool.  Each generated scene is loaded by
both branches, and Stage5's scale-zero perturbation boundary is used only to
freeze the same observations and remove internal residuals in the outer-only
branch at the correct point in the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

import run_ds_perturbation_sweep as sweep


@dataclass(frozen=True)
class Scene:
    camera: sweep.Camera
    frames: dict[int, np.ndarray]
    boards: dict[int, np.ndarray]
    raw_lines: tuple[str, ...]


@dataclass(frozen=True)
class WeakMode:
    index: int
    eigenvalue: float
    direction: np.ndarray
    label: str
    subspace_angle_deg: float = math.nan


@dataclass(frozen=True)
class CalibratedPerturbation:
    mode_index: int
    mode_label: str
    subspace_angle_deg: float
    sign: int
    target_peripheral_ray_p95_deg: float
    amplitude: float
    camera: sweep.Camera
    initial_full_ray_p95_deg: float
    initial_peripheral_ray_p95_deg: float
    valid_grid_ratio: float


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-mat", type=Path, required=True)
    parser.add_argument("--test-mat", type=Path, required=True)
    parser.add_argument("--train-precomputed-dir", type=Path)
    parser.add_argument("--test-precomputed-dir", type=Path)
    parser.add_argument("--camera", choices=("left", "right"), required=True)
    parser.add_argument("--reference-camchain", type=Path, required=True)
    parser.add_argument("--clean-reference-run", type=Path, required=True)
    parser.add_argument("--fixed-backend-seed-list", type=Path, required=True)
    parser.add_argument("--fixed-backend-input-list", type=Path, required=True)
    parser.add_argument(
        "--fisher-backend-input-list",
        type=Path,
        help=(
            "Optional common candidate list used only to define the weak "
            "mode. This keeps the perturbation identical while the actual "
            "backend input prefix changes."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--config", type=Path,
        default=repo / "aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml",
    )
    parser.add_argument("--backend", type=Path, default=repo / "build/run_stage5_backend")
    parser.add_argument("--python", type=Path, default=sweep.DEFAULT_SCIPY_PYTHON)
    parser.add_argument("--model", default="ds-none", choices=("ds-none",))
    parser.add_argument("--fixed-backend-shuffle-seed", type=int, default=1337)
    parser.add_argument("--reject-unlisted-fixed-backend-input", action="store_true", default=True)
    parser.add_argument("--persistent-fix-board-layout", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--exact-fixed-backend-input",
        action="store_true",
        help=(
            "Use the supplied force-include frame-board list as the exact "
            "paired backend input in both branches."
        ),
    )
    parser.add_argument(
        "--release-intrinsics-immediately",
        action="store_true",
        help=(
            "Release DS intrinsics in the first incremental trial; this is "
            "needed when testing recovery from a perturbed camera seed."
        ),
    )
    parser.add_argument("--grid-size", type=int, default=121)
    parser.add_argument("--weak-mode-indices", default="0,1,2")
    parser.add_argument(
        "--weak-subspace-angles-deg",
        default="",
        help=(
            "Optional comma-separated angles in span(W1,W2). When set, "
            "these preregistered directions replace --weak-mode-indices."
        ),
    )
    parser.add_argument("--signs", default="-1,1")
    parser.add_argument("--target-peripheral-p95-deg", default="0.5,1,2,4")
    parser.add_argument("--minimum-valid-grid-ratio", type=float, default=0.98)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Expected finite positive values")
    return values


def parse_scene(path: Path) -> Scene:
    raw_lines = tuple(path.read_text(encoding="utf-8").splitlines())
    camera: sweep.Camera | None = None
    frames: dict[int, np.ndarray] = {}
    boards: dict[int, np.ndarray] = {}
    for line in raw_lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "camera" and len(fields) == 7:
            camera = sweep.Camera(*(float(value) for value in fields[1:]), family="ds-none")
        elif fields[0] in {"frame", "board"} and len(fields) >= 21:
            identifier = int(fields[1])
            matrix = np.asarray([float(value) for value in fields[-16:]], dtype=np.float64).reshape(4, 4)
            (frames if fields[0] == "frame" else boards)[identifier] = matrix
    if camera is None or not frames or not boards:
        raise RuntimeError(f"Malformed Stage5 reference scene: {path}")
    return Scene(camera, frames, boards, raw_lines)


def write_scene(path: Path, scene: Scene, camera: sweep.Camera) -> None:
    camera_line = "camera " + " ".join(
        f"{value:.17g}" for value in
        (camera.xi, camera.alpha, camera.fu, camera.fv, camera.cu, camera.cv)
    )
    lines = [camera_line if line.startswith("camera ") else line for line in scene.raw_lines]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_frame_board_keys(paths: Iterable[Path]) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                keys.add((int(row["frame_index"]), int(row["board_id"])))
    if not keys:
        raise RuntimeError("The fixed backend schedule is empty")
    return keys


def read_points(
    path: Path,
    keys: set[tuple[int, int]],
    point_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["frame_index"]), int(row["board_id"]))
            if key not in keys or (
                point_types is not None and row["point_type"] not in point_types
            ):
                continue
            points.append({
                "frame": key[0],
                "board": key[1],
                "point": np.asarray(
                    [float(row["target_x"]), float(row["target_y"]), float(row["target_z"]), 1.0],
                    dtype=np.float64,
                ),
            })
    if not points:
        raise RuntimeError(f"No requested points found in fixed backend schedule: {path}")
    return points


def project_ds(camera: sweep.Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    z1 = camera.xi * d1 + z
    d2 = np.sqrt(x * x + y * y + z1 * z1)
    denominator = camera.alpha * d2 + (1.0 - camera.alpha) * z1
    valid = (
        np.isfinite(denominator) & (np.abs(denominator) > 1e-12) &
        np.isfinite(d1) & np.isfinite(d2)
    )
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = camera.fu * x[valid] / denominator[valid] + camera.cu
    pixels[valid, 1] = camera.fv * y[valid] / denominator[valid] + camera.cv
    valid &= np.all(np.isfinite(pixels), axis=1)
    return pixels, valid


def camera_to_coordinates(camera: sweep.Camera, width: int, height: int) -> np.ndarray:
    return np.asarray([
        camera.xi,
        camera.alpha,
        math.log(camera.fu),
        math.log(camera.fv),
        camera.cu / width,
        camera.cv / height,
    ], dtype=np.float64)


def camera_from_coordinates(values: np.ndarray, width: int, height: int) -> sweep.Camera:
    return sweep.Camera(
        float(values[0]), float(values[1]),
        float(math.exp(values[2])), float(math.exp(values[3])),
        float(values[4] * width), float(values[5] * height),
        family="ds-none",
    )


def perturb_transform_left(transform: np.ndarray, axis: int, delta: float) -> np.ndarray:
    perturbation = np.eye(4, dtype=np.float64)
    if axis < 3:
        perturbation[axis, 3] = delta
    else:
        vector = np.zeros(3, dtype=np.float64)
        vector[axis - 3] = delta
        perturbation[:3, :3] = Rotation.from_rotvec(vector).as_matrix()
    return perturbation @ transform


def observation_pixels(
    scene: Scene,
    camera: sweep.Camera,
    points: list[dict[str, Any]],
    frame_overrides: dict[int, np.ndarray] | None = None,
    board_overrides: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    frame_overrides = frame_overrides or {}
    board_overrides = board_overrides or {}
    camera_points = []
    for point in points:
        frame = frame_overrides.get(point["frame"], scene.frames[point["frame"]])
        board = board_overrides.get(point["board"], scene.boards[point["board"]])
        camera_points.append((frame @ board @ point["point"])[:3])
    pixels, valid = project_ds(camera, np.asarray(camera_points, dtype=np.float64))
    if not np.all(valid):
        raise RuntimeError("Clean Fisher construction encountered an invalid DS projection")
    return pixels.reshape(-1)


def canonicalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    pivot = int(np.argmax(np.abs(direction)))
    return -direction if direction[pivot] < 0.0 else direction


def compute_weak_modes(
    scene: Scene,
    points: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[list[WeakMode], np.ndarray, dict[str, Any]]:
    frame_ids = sorted({int(point["frame"]) for point in points})
    board_ids = sorted({int(point["board"]) for point in points})
    reference_board = min(board_ids)
    variable_board_ids = [board for board in board_ids if board != reference_board]
    nuisance_blocks = [("frame", identifier) for identifier in frame_ids]
    nuisance_blocks += [("board", identifier) for identifier in variable_board_ids]
    residual_count = 2 * len(points)
    jacobian_camera = np.zeros((residual_count, 6), dtype=np.float64)
    jacobian_nuisance = np.zeros((residual_count, 6 * len(nuisance_blocks)), dtype=np.float64)

    coordinates = camera_to_coordinates(scene.camera, width, height)
    camera_steps = np.asarray([1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6])
    for column, step in enumerate(camera_steps):
        plus = coordinates.copy()
        minus = coordinates.copy()
        plus[column] += step
        minus[column] -= step
        jacobian_camera[:, column] = (
            observation_pixels(scene, camera_from_coordinates(plus, width, height), points) -
            observation_pixels(scene, camera_from_coordinates(minus, width, height), points)
        ) / (2.0 * step)

    for block_index, (kind, identifier) in enumerate(nuisance_blocks):
        for axis in range(6):
            step = 1e-5 if axis < 3 else 1e-6
            plus_transform = perturb_transform_left(
                scene.frames[identifier] if kind == "frame" else scene.boards[identifier], axis, step
            )
            minus_transform = perturb_transform_left(
                scene.frames[identifier] if kind == "frame" else scene.boards[identifier], axis, -step
            )
            plus_frames = {identifier: plus_transform} if kind == "frame" else None
            minus_frames = {identifier: minus_transform} if kind == "frame" else None
            plus_boards = {identifier: plus_transform} if kind == "board" else None
            minus_boards = {identifier: minus_transform} if kind == "board" else None
            jacobian_nuisance[:, 6 * block_index + axis] = (
                observation_pixels(scene, scene.camera, points, plus_frames, plus_boards) -
                observation_pixels(scene, scene.camera, points, minus_frames, minus_boards)
            ) / (2.0 * step)

    hcc = jacobian_camera.T @ jacobian_camera
    hcn = jacobian_camera.T @ jacobian_nuisance
    hnn = jacobian_nuisance.T @ jacobian_nuisance
    nuisance_eigenvalues, nuisance_eigenvectors = np.linalg.eigh(0.5 * (hnn + hnn.T))
    nuisance_tolerance = max(1e-12, float(np.max(nuisance_eigenvalues)) * 1e-10)
    nuisance_inverse_values = np.where(
        nuisance_eigenvalues > nuisance_tolerance,
        1.0 / nuisance_eigenvalues,
        0.0,
    )
    hnn_pinv = (nuisance_eigenvectors * nuisance_inverse_values) @ nuisance_eigenvectors.T
    fisher = hcc - hcn @ hnn_pinv @ hcn.T
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    modes = [
        WeakMode(
            index, float(eigenvalues[index]),
            canonicalize_direction(eigenvectors[:, index]), f"W{index + 1}",
        )
        for index in range(6)
    ]
    positive = eigenvalues[eigenvalues > max(1e-12, float(eigenvalues[-1]) * 1e-12)]
    audit = {
        "coordinate_order": ["xi", "alpha", "log_fu", "log_fv", "cu_over_width", "cv_over_height"],
        "outer_point_count": len(points),
        "frame_count": len(frame_ids),
        "board_count": len(board_ids),
        "reference_board_id": reference_board,
        "nuisance_dimension": int(jacobian_nuisance.shape[1]),
        "nuisance_rank": int(np.count_nonzero(nuisance_eigenvalues > nuisance_tolerance)),
        "fisher_eigenvalues": [float(value) for value in eigenvalues],
        "fisher_condition_number": (
            float(positive[-1] / positive[0]) if positive.size else math.inf
        ),
    }
    return modes, fisher, audit


def valid_camera(camera: sweep.Camera) -> bool:
    return (
        0.0 < camera.alpha < 1.0 and
        camera.fu > 0.0 and camera.fv > 0.0 and
        all(math.isfinite(value) for value in
            (camera.xi, camera.alpha, camera.fu, camera.fv, camera.cu, camera.cv))
    )


def calibrate_perturbation(
    reference: sweep.Camera,
    direction: np.ndarray,
    sign: int,
    target: float,
    width: int,
    height: int,
    mask: sweep.EvaluationMask,
    minimum_valid_ratio: float,
) -> tuple[float, sweep.Camera, dict[str, float | int]]:
    base = camera_to_coordinates(reference, width, height)

    def evaluate(amplitude: float) -> tuple[sweep.Camera, dict[str, float | int]]:
        camera = camera_from_coordinates(base + sign * amplitude * direction, width, height)
        if not valid_camera(camera):
            return camera, {"valid_grid_ratio": 0.0, "peripheral_ray_p95_deg": math.nan}
        return camera, sweep.ray_metrics(mask, camera)

    lower = 0.0
    upper = 1e-7
    upper_camera, upper_metrics = evaluate(upper)
    while (
        float(upper_metrics["valid_grid_ratio"]) >= minimum_valid_ratio and
        math.isfinite(float(upper_metrics["peripheral_ray_p95_deg"])) and
        float(upper_metrics["peripheral_ray_p95_deg"]) < target and
        upper < 32.0
    ):
        lower = upper
        upper *= 2.0
        upper_camera, upper_metrics = evaluate(upper)
    if (
        float(upper_metrics["valid_grid_ratio"]) < minimum_valid_ratio or
        not math.isfinite(float(upper_metrics["peripheral_ray_p95_deg"])) or
        float(upper_metrics["peripheral_ray_p95_deg"]) < target
    ):
        raise RuntimeError(
            f"Weak direction sign={sign:+d} cannot reach peripheral P95={target} deg "
            f"while retaining valid_grid_ratio>={minimum_valid_ratio}"
        )
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        _, metrics = evaluate(midpoint)
        if (
            float(metrics["valid_grid_ratio"]) >= minimum_valid_ratio and
            float(metrics["peripheral_ray_p95_deg"]) < target
        ):
            lower = midpoint
        else:
            upper = midpoint
    camera, metrics = evaluate(upper)
    if float(metrics["valid_grid_ratio"]) < minimum_valid_ratio:
        raise RuntimeError("Ray-target bisection crossed the fixed-mask validity boundary")
    return upper, camera, metrics


def target_label(target: float) -> str:
    return f"rayp95_{int(round(target * 1000.0)):05d}mdeg"


def run_command(command: list[str], cwd: Path, output: Path, resume: bool, dry_run: bool) -> None:
    if resume and sweep.completed_run(output):
        print(f"= resume {output}", flush=True)
        return
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def paired_run(
    args: argparse.Namespace,
    repo: Path,
    output: Path,
    scene_path: Path,
) -> tuple[Path, Path]:
    outer = output / "outer_only"
    internal = output / "outer_internal"
    # P1 at scale zero is an identity operation. It deliberately keeps the
    # native Stage5 perturbation boundary and frozen-observation audit active.
    run_command(
        stage5_command(args, "outer_only", outer, scene_path),
        repo, outer, args.resume, args.dry_run,
    )
    run_command(
        stage5_command(args, "outer_internal", internal, scene_path),
        repo, internal, args.resume, args.dry_run,
    )
    return outer, internal


def stage5_command(
    args: argparse.Namespace,
    mode: str,
    output: Path,
    reference_scene: Path,
) -> list[str]:
    if args.train_precomputed_dir is None:
        command = sweep.run_stage5_command(
            args, "P1", 0.0, mode, output, reference_scene
        )
        if args.persistent_fix_board_layout:
            command.extend([
                "--stage5-trial-backend-selection-persistent-fix-board-layout",
                "1",
            ])
        return command
    if args.test_precomputed_dir is None:
        raise ValueError(
            "--test-precomputed-dir is required with --train-precomputed-dir"
        )
    command = [
        str(args.backend.resolve()),
        "--config", str(args.config.resolve()),
        "--models", args.model,
        "--kalibr-camchain", str(args.reference_camchain.resolve()),
        "--output", str(output.resolve()),
        "--stage5-precomputed-observations-dir",
        str(args.train_precomputed_dir.resolve()),
        "--stage5-precomputed-holdout-observations-dir",
        str(args.test_precomputed_dir.resolve()),
        "--stage5-precomputed-target-mode", "multi_board",
        "--runtime-mode", "research",
        "--cache-dir", str((output / ".stage5_backend_cache").resolve()),
        "--include-internal-points", "1",
        "--stage5-large-intrinsic-perturbation", "P1",
        "--stage5-large-intrinsic-perturbation-scale", "0",
        "--stage5-large-intrinsic-perturbation-reference-scene",
        str(reference_scene.resolve()),
        "--stage5-disable-selected-case-visualizations",
        "--stage5-enable-polar-angle-diagnostics",
        "--stage5-trial-backend-selection-force-include-frame-board-list",
        str(args.fixed_backend_input_list.resolve()),
        "--stage5-trial-backend-selection-candidate-order", "random_shuffle",
        "--stage5-trial-backend-selection-candidate-shuffle-seed",
        str(args.fixed_backend_shuffle_seed),
        "--stage5-trial-backend-selection-mi-tol", "1e12",
        "--stage5-trial-backend-selection-seed-frame-board-list",
        str(args.fixed_backend_seed_list.resolve()),
    ]
    if args.exact_fixed_backend_input:
        command.extend([
            "--stage5-trial-backend-selection-force-include-list-is-exact-input",
            "1",
        ])
    if args.release_intrinsics_immediately:
        command.extend([
            "--stage5-trial-backend-selection-optimize-intrinsics",
            "1",
            "--stage5-trial-backend-selection-delayed-intrinsics-release",
            "0",
            "--stage5-trial-backend-selection-persistent-intrinsics-anchor-prior",
            "0",
        ])
    if args.persistent_fix_board_layout:
        command.extend([
            "--stage5-trial-backend-selection-persistent-fix-board-layout",
            "1",
        ])
    if mode == "outer_only":
        command.append(
            "--stage5-large-intrinsic-perturbation-outer-only-after-application"
        )
    return command


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.output_root = args.output_root.resolve()
    args.clean_reference_run = args.clean_reference_run.resolve()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)

    source_scene_path = args.clean_reference_run / "large_intrinsic_perturbation_reference_scene.txt"
    points_path = args.clean_reference_run / "precomputed_input/training/points.csv"
    scene = parse_scene(source_scene_path)
    width, height = sweep.image_size_from_mat(args.train_mat.resolve())
    optimization_keys = read_frame_board_keys(
        (args.fixed_backend_seed_list, args.fixed_backend_input_list)
    )
    fisher_input = (
        args.fisher_backend_input_list
        if args.fisher_backend_input_list is not None
        else args.fixed_backend_input_list
    )
    fisher_keys = read_frame_board_keys(
        (args.fixed_backend_seed_list, fisher_input)
    )
    points = read_points(points_path, fisher_keys, {"outer"})
    full_points = read_points(points_path, fisher_keys)
    modes, fisher, fisher_audit = compute_weak_modes(scene, points, width, height)
    _, full_fisher, full_fisher_audit = compute_weak_modes(
        scene, full_points, width, height
    )
    fisher_audit.update({
        "optimization_frame_board_count": len(optimization_keys),
        "fisher_frame_board_count": len(fisher_keys),
        "fisher_backend_input_list": str(fisher_input.resolve()),
        "outer_internal_point_count": len(full_points),
        "outer_internal_fisher_eigenvalues":
            full_fisher_audit["fisher_eigenvalues"],
        "outer_internal_fisher_condition_number":
            full_fisher_audit["fisher_condition_number"],
        "minimum_eigenvalue_gain_outer_internal_over_outer_only":
            full_fisher_audit["fisher_eigenvalues"][0] /
            fisher_audit["fisher_eigenvalues"][0],
    })
    common_mask = sweep.build_evaluation_mask(scene.camera, width, height, args.grid_size)

    mode_indices = parse_int_list(args.weak_mode_indices)
    signs = parse_int_list(args.signs)
    if any(index < 0 or index >= len(modes) for index in mode_indices):
        raise ValueError("--weak-mode-indices must lie in [0, 5]")
    if any(sign not in (-1, 1) for sign in signs):
        raise ValueError("--signs supports only -1 and 1")
    targets = parse_float_list(args.target_peripheral_p95_deg)

    selected_modes = [modes[index] for index in mode_indices]
    if args.weak_subspace_angles_deg.strip():
        angles = sorted({
            float(value.strip())
            for value in args.weak_subspace_angles_deg.split(",")
            if value.strip()
        })
        if not angles or any(not math.isfinite(angle) for angle in angles):
            raise ValueError("--weak-subspace-angles-deg requires finite angles")
        selected_modes = []
        for ordinal, angle in enumerate(angles):
            radians = math.radians(angle)
            direction = (
                math.cos(radians) * modes[0].direction +
                math.sin(radians) * modes[1].direction
            )
            direction /= np.linalg.norm(direction)
            selected_modes.append(WeakMode(
                1000 + ordinal,
                float(direction @ fisher @ direction),
                direction,
                f"S12_{angle:06.2f}deg".replace(".", "p"),
                angle,
            ))

    np.savetxt(output / "outer_only_schur_fisher.csv", fisher, delimiter=",", fmt="%.17g")
    np.savetxt(
        output / "outer_internal_schur_fisher.csv", full_fisher,
        delimiter=",", fmt="%.17g",
    )
    weak_rows = []
    for mode in modes:
        weak_rows.append({
            "mode_index": mode.index,
            "eigenvalue": mode.eigenvalue,
            **{name: float(value) for name, value in zip(fisher_audit["coordinate_order"], mode.direction)},
        })
    sweep.write_csv(output / "outer_only_weak_modes.csv", weak_rows)
    (output / "outer_only_weak_mode_audit.json").write_text(
        json.dumps(fisher_audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    perturbations: list[CalibratedPerturbation] = []
    mode_by_label = {mode.label: mode for mode in selected_modes}
    for mode in selected_modes:
        for sign in signs:
            for target in targets:
                amplitude, camera, metrics = calibrate_perturbation(
                    scene.camera, mode.direction, sign, target,
                    width, height, common_mask, args.minimum_valid_grid_ratio,
                )
                perturbation = CalibratedPerturbation(
                    mode_index=mode.index,
                    mode_label=mode.label,
                    subspace_angle_deg=mode.subspace_angle_deg,
                    sign=sign,
                    target_peripheral_ray_p95_deg=target,
                    amplitude=amplitude,
                    camera=camera,
                    initial_full_ray_p95_deg=float(metrics["full_ray_p95_deg"]),
                    initial_peripheral_ray_p95_deg=float(metrics["peripheral_ray_p95_deg"]),
                    valid_grid_ratio=float(metrics["valid_grid_ratio"]),
                )
                perturbations.append(perturbation)
                label = f"{mode.label}_{'plus' if sign > 0 else 'minus'}/{target_label(target)}"
                write_scene(output / label / "perturbed_reference_scene.txt", scene, camera)

    perturbation_rows = []
    for item in perturbations:
        row = asdict(item)
        row.pop("camera")
        row.update({
            "initial_xi": item.camera.xi,
            "initial_alpha": item.camera.alpha,
            "initial_fu": item.camera.fu,
            "initial_fv": item.camera.fv,
            "initial_cu": item.camera.cu,
            "initial_cv": item.camera.cv,
            "camera_fingerprint": sweep.camera_fingerprint(item.camera),
        })
        perturbation_rows.append(row)
    sweep.write_csv(output / "calibrated_perturbations.csv", perturbation_rows)
    if args.analysis_only or args.dry_run:
        return 0

    clean_scene = output / "reference/clean_reference_scene.txt"
    write_scene(clean_scene, scene, scene.camera)
    clean_outer, clean_internal = paired_run(args, repo, output / "reference/clean", clean_scene)
    clean_outer_camera = sweep.camera_from_training_summary(
        clean_outer / "backend_training_summary.txt", "ds-none"
    )
    clean_internal_camera = sweep.camera_from_training_summary(
        clean_internal / "backend_training_summary.txt", "ds-none"
    )
    common_reference = clean_internal_camera
    common_mask = sweep.build_evaluation_mask(common_reference, width, height, args.grid_size)

    errors: list[str] = []
    run_outputs: dict[tuple[str, int, float], tuple[Path, Path]] = {}

    def run_one(item: CalibratedPerturbation) -> tuple[tuple[str, int, float], tuple[Path, Path]]:
        root = output / f"{item.mode_label}_{'plus' if item.sign > 0 else 'minus'}" / target_label(item.target_peripheral_ray_p95_deg)
        return (
            (item.mode_label, item.sign, item.target_peripheral_ray_p95_deg),
            paired_run(args, repo, root, root / "perturbed_reference_scene.txt"),
        )

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {executor.submit(run_one, item): item for item in perturbations}
        for future in as_completed(futures):
            item = futures[future]
            try:
                key, paths = future.result()
                run_outputs[key] = paths
            except Exception as error:  # Preserve all other preregistered conditions.
                errors.append(
                    f"{item.mode_label} sign={item.sign:+d} "
                    f"target={item.target_peripheral_ray_p95_deg}: {error}"
                )

    rows: list[dict[str, Any]] = []
    for item in perturbations:
        paths = run_outputs.get((item.mode_label, item.sign, item.target_peripheral_ray_p95_deg))
        if paths is None:
            continue
        direction = f"{item.mode_label}_{'plus' if item.sign > 0 else 'minus'}"
        for mode, path, branch_clean in (
            ("outer_only", paths[0], clean_outer_camera),
            ("outer_internal", paths[1], clean_internal_camera),
        ):
            row = sweep.branch_row(
                path, direction, item.target_peripheral_ray_p95_deg, mode,
                common_mask, common_reference, branch_clean, "ds-none",
            )
            row.update({
                "weak_mode_index": item.mode_index,
                "weak_mode_label": item.mode_label,
                "weak_subspace_angle_deg": item.subspace_angle_deg,
                "weak_mode_sign": item.sign,
                "weak_mode_eigenvalue": mode_by_label[item.mode_label].eigenvalue,
                "perturbation_amplitude": item.amplitude,
                "target_initial_peripheral_ray_p95_deg": item.target_peripheral_ray_p95_deg,
                "valid_ds_model": int(valid_camera(sweep.Camera(
                    float(row["final_xi"]), float(row["final_alpha"]),
                    float(row["final_fu"]), float(row["final_fv"]),
                    float(row["final_cu"]), float(row["final_cv"]),
                    family="ds-none",
                ))),
                "xi_in_conservative_initial_perturbation_guard": int(
                    -1.0 < float(row["final_xi"]) < 1.0
                ),
            })
            rows.append(row)

    sweep.write_csv(output / "weak_mode_perturbation_results.csv", rows)
    paired = sweep.paired_rows(rows)
    rows_by_condition = {
        (str(row["direction"]), float(row["scale"]), str(row["mode"])): row
        for row in rows
    }
    for paired_row in paired:
        key = (str(paired_row["direction"]), float(paired_row["scale"]))
        outer = rows_by_condition[key + ("outer_only",)]
        internal = rows_by_condition[key + ("outer_internal",)]
        paired_row.update({
            "delta_branch_relative_full_ray_p95_deg":
                float(outer["final_branch_full_ray_p95_deg"]) -
                float(internal["final_branch_full_ray_p95_deg"]),
            "delta_branch_relative_peripheral_ray_p95_deg":
                float(outer["final_branch_peripheral_ray_p95_deg"]) -
                float(internal["final_branch_peripheral_ray_p95_deg"]),
            "outer_valid_ds_model": outer["valid_ds_model"],
            "internal_valid_ds_model": internal["valid_ds_model"],
        })
    sweep.write_csv(output / "weak_mode_paired_improvements.csv", paired)
    valid_input_pairs = [
        row for row in paired
        if row["paired_initial_state_valid"] == 1 and
        row["backend_seed_set_identical"] == 1 and
        row["backend_attempted_schedule_identical"] == 1
    ]
    same_committed_output_pairs = [
        row for row in valid_input_pairs
        if row["backend_committed_schedule_identical"] == 1 and
        row["backend_committed_set_identical"] == 1
    ]
    audit = {
        "dataset_id": args.dataset_id,
        "protocol": "paired_ground_truth_semi_synthetic_recovery",
        "exact_fixed_backend_input": int(args.exact_fixed_backend_input),
        "intrinsics_released_immediately": int(args.release_intrinsics_immediately),
        "persistent_board_layout_fixed": int(args.persistent_fix_board_layout),
        "condition_count": len(perturbations),
        "completed_condition_count": len(run_outputs),
        "paired_result_count": len(paired),
        "strictly_valid_input_pair_count": len(valid_input_pairs),
        "same_committed_output_pair_count": len(same_committed_output_pairs),
        "branch_asymmetric_commit_count":
            len(valid_input_pairs) - len(same_committed_output_pairs),
        "errors": errors,
    }
    (output / "weak_mode_experiment_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if errors or len(valid_input_pairs) != len(perturbations):
        print(json.dumps(audit, indent=2), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())

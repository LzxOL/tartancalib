#!/usr/bin/env python3
"""Run paired DS or KB intrinsic-perturbation recovery experiments.

The perturbation is applied by Stage5 after internal observations have been
recovered and before selection/persistent incremental BA.  For each condition
the Outer+Internal branch loads the Outer-only branch's saved scene snapshot,
so camera, frame poses, board layout, and perturbation are identical at the
selection boundary.  Only the presence of fixed internal observations differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


DEFAULT_SCIPY_PYTHON = Path(
    "/Users/linzhaoxian/.cache/codex-runtimes/"
    "codex-primary-runtime/dependencies/python/bin/python3"
)
PROFILES = {
    "P1": (0.70, -0.20, 0.10),
    "P1F50": (0.50, -0.20, 0.10),
    "P2": (0.70, 0.20, -0.10),
    "P2F50": (0.50, 0.20, -0.10),
    "P3": (1.30, -0.20, 0.10),
    "P4": (1.30, 0.20, -0.10),
    # Local P1 neighborhood. The actual coefficients are defined by the
    # native Stage5 perturbation profile so the C++ and Python paths cannot
    # silently disagree about the logarithmic focal interpolation.
    "N0": (0.70, -0.20, 0.10),
    "N1": (0.70, -0.20, 0.10),
    "N2": (0.70, -0.20, 0.10),
    "N3": (0.70, -0.18, 0.11),
    "N4": (0.70, -0.22, 0.09),
}


@dataclass(frozen=True)
class Camera:
    xi: float
    alpha: float
    fu: float
    fv: float
    cu: float
    cv: float
    family: str = "ds-none"
    distortion: tuple[float, ...] = ()


@dataclass(frozen=True)
class EvaluationMask:
    pixels: np.ndarray
    rho: np.ndarray
    reference_rays: np.ndarray
    reference_valid: np.ndarray
    width: int
    height: int
    fixed_radius_px: float


def camera_fingerprint(camera: Camera) -> str:
    values = [camera.family, *(f"{value:.15g}" for value in (
        camera.xi, camera.alpha, camera.fu, camera.fv, camera.cu, camera.cv
    )), *(f"{value:.15g}" for value in camera.distortion)]
    return "sha256:" + hashlib.sha256("|".join(values).encode("ascii")).hexdigest()


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-mat", type=Path, required=True)
    parser.add_argument("--test-mat", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--camera", choices=("left", "right"), required=True)
    parser.add_argument("--reference-camchain", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo / "aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml",
    )
    parser.add_argument("--backend", type=Path, default=repo / "build/run_stage5_backend")
    parser.add_argument("--model", choices=("ds-none", "kb"), default="ds-none")
    parser.add_argument("--profiles", default="P1,P2,P3,P4")
    parser.add_argument("--scales", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument(
        "--strict-perturbation-scale", action="store_true",
        help="Preserve scales above one exactly; enables the strict KB focal+principal P1 protocol.",
    )
    parser.add_argument("--grid-size", type=int, default=121)
    parser.add_argument(
        "--fixed-backend-input-list", type=Path,
        help="Force every listed frame-board batch through the persistent Backend in a fixed shuffled order.",
    )
    parser.add_argument(
        "--fixed-backend-seed-list", type=Path,
        help="Use exactly these frame-board observations as the common fixed-intrinsics persistent seed.",
    )
    parser.add_argument("--fixed-backend-shuffle-seed", type=int, default=1337)
    parser.add_argument(
        "--reject-unlisted-fixed-backend-input",
        action="store_true",
        help=(
            "Accept only forced batches from --fixed-backend-input-list; "
            "unlisted candidates remain visible to the audit but fail the "
            "information threshold."
        ),
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Number of independent P×scale pairs to run concurrently. Each pair remains serial.",
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_SCIPY_PYTHON)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_scales(raw: str) -> list[float]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(not math.isfinite(value) or value < 0.0 or value > 2.0 for value in values):
        raise ValueError("--scales must contain finite values in [0, 2]")
    return values


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def as_float(values: dict[str, str], key: str) -> float:
    try:
        return float(values[key])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"Missing or invalid {key!r}") from error


def parse_csv_floats(text: str) -> tuple[float, ...]:
    if not text.strip():
        return ()
    return tuple(float(value) for value in text.split(","))


def camera_from_training_summary(path: Path, family: str) -> Camera:
    values = read_kv(path)
    precise_intrinsics = parse_csv_floats(values.get("camera_intrinsics_csv", ""))
    if family == "kb":
        distortion = parse_csv_floats(values.get("camera_distortion_csv", ""))
        if len(precise_intrinsics) == 4 and len(distortion) == 4:
            return Camera(0.0, 0.0, *precise_intrinsics, family="pinhole-equi", distortion=distortion)
        if len(distortion) != 4:
            raise RuntimeError(f"{path} does not contain four KB coefficients")
        return Camera(
            0.0, 0.0,
            *(as_float(values, "camera_" + key) for key in ("fu", "fv", "cu", "cv")),
            family="pinhole-equi",
            distortion=distortion,
        )
    if len(precise_intrinsics) == 6:
        return Camera(*precise_intrinsics, family="ds-none")
    return Camera(
        *(as_float(values, "camera_" + key)
          for key in ("xi", "alpha", "fu", "fv", "cu", "cv")),
        family="ds-none",
    )


def camera_from_intrinsics(text: str, distortion_text: str, family: str) -> Camera:
    values = [float(value) for value in text.split(",")]
    if family == "kb":
        distortion = parse_csv_floats(distortion_text)
        if len(values) != 4 or len(distortion) != 4:
            raise RuntimeError("Expected four KB intrinsics and four distortion coefficients")
        return Camera(0.0, 0.0, *values, family="pinhole-equi", distortion=distortion)
    if len(values) != 6:
        raise RuntimeError(f"Expected six DS intrinsics, got {text!r}")
    return Camera(*values, family="ds-none")


def image_size_from_mat(path: Path) -> tuple[int, int]:
    values = np.asarray(loadmat(path, variable_names=["imgsize"])["imgsize"]).reshape(-1)
    if values.size != 2:
        raise RuntimeError(f"{path} has malformed imgsize")
    height, width = (int(values[0]), int(values[1]))
    if height <= 0 or width <= 0:
        raise RuntimeError(f"{path} has invalid imgsize {values.tolist()}")
    return width, height


def unproject_ds(camera: Camera, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx = (pixels[:, 0] - camera.cu) / camera.fu
    my = (pixels[:, 1] - camera.cv) / camera.fv
    r2 = mx * mx + my * my
    inner = 1.0 - (2.0 * camera.alpha - 1.0) * r2
    valid = np.isfinite(inner) & (inner > 1e-12)
    mz = np.full_like(mx, np.nan)
    denom = camera.alpha * np.sqrt(np.maximum(inner, 0.0)) + (1.0 - camera.alpha)
    good = valid & (np.abs(denom) > 1e-12)
    mz[good] = (1.0 - camera.alpha * camera.alpha * r2[good]) / denom[good]
    ray_denom = mz * mz + r2
    good &= np.isfinite(ray_denom) & (np.abs(ray_denom) > 1e-12)
    root = np.sqrt(np.maximum(mz * mz + (1.0 - camera.xi * camera.xi) * r2, 0.0))
    k = np.full_like(mx, np.nan)
    k[good] = (mz[good] * camera.xi + root[good]) / ray_denom[good]
    rays = np.stack([k * mx, k * my, k * mz - camera.xi], axis=1)
    norms = np.linalg.norm(rays, axis=1)
    good &= np.isfinite(norms) & (norms > 1e-12)
    rays[good] /= norms[good, None]
    return rays, good


def unproject_kb(camera: Camera, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(camera.distortion) != 4:
        return np.full((pixels.shape[0], 3), np.nan), np.zeros(pixels.shape[0], dtype=bool)
    xd = (pixels[:, 0] - camera.cu) / camera.fu
    yd = (pixels[:, 1] - camera.cv) / camera.fv
    rd = np.hypot(xd, yd)
    k1, k2, k3, k4 = camera.distortion

    def distorted(theta: np.ndarray) -> np.ndarray:
        theta2 = theta * theta
        return theta * (
            1.0 + k1 * theta2 + k2 * theta2**2
            + k3 * theta2**3 + k4 * theta2**4
        )

    max_theta = 0.5 * np.pi - 1e-9
    max_rd = float(distorted(np.asarray(max_theta)))
    valid = np.isfinite(rd) & math.isfinite(max_rd) & (rd <= max_rd + 1e-9)
    low = np.zeros_like(rd)
    high = np.full_like(rd, max_theta)
    for _ in range(80):
        mid = 0.5 * (low + high)
        move_low = distorted(mid) < rd
        low = np.where(move_low, mid, low)
        high = np.where(move_low, high, mid)
    theta = 0.5 * (low + high)
    rays = np.zeros((pixels.shape[0], 3), dtype=np.float64)
    center = rd < 1e-12
    noncenter = ~center
    radial = np.zeros_like(rd)
    radial[noncenter] = np.sin(theta[noncenter]) / rd[noncenter]
    rays[:, 0] = xd * radial
    rays[:, 1] = yd * radial
    rays[:, 2] = np.cos(theta)
    rays[center] = np.asarray([0.0, 0.0, 1.0])
    norms = np.linalg.norm(rays, axis=1)
    valid &= np.isfinite(norms) & (norms > 1e-12)
    rays[valid] /= norms[valid, None]
    rays[~valid] = np.nan
    return rays, valid


def unproject_camera(camera: Camera, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return unproject_kb(camera, pixels) if camera.family == "pinhole-equi" else unproject_ds(camera, pixels)


def build_evaluation_mask(reference: Camera, width: int, height: int, grid_size: int) -> EvaluationMask:
    if grid_size < 11:
        raise ValueError("--grid-size must be at least 11")
    axis_x = np.linspace(0.0, float(width - 1), grid_size)
    axis_y = np.linspace(0.0, float(height - 1), grid_size)
    pixels = np.asarray([(x, y) for y in axis_y for x in axis_x], dtype=np.float64)
    center = np.asarray([reference.cu, reference.cv], dtype=np.float64)
    fixed_radius = min(reference.cu, reference.cv, width - 1 - reference.cu, height - 1 - reference.cv)
    if not math.isfinite(fixed_radius) or fixed_radius <= 0.0:
        raise RuntimeError("Common-reference principal point cannot define an evaluation disc")
    rho = np.linalg.norm(pixels - center[None, :], axis=1) / fixed_radius
    reference_rays, reference_valid = unproject_camera(reference, pixels)
    # The fixed disc avoids rectangular corner/black-border regions.  Validity
    # is defined only by the common reference and never by a trial model.
    reference_valid &= rho <= 1.0
    if int(np.count_nonzero(reference_valid)) == 0:
        raise RuntimeError("Common reference produced an empty fixed ray-evaluation mask")
    return EvaluationMask(pixels, rho, reference_rays, reference_valid, width, height, fixed_radius)


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else math.nan


def ray_metrics(
    mask: EvaluationMask,
    candidate: Camera,
    comparison_reference: Camera | None = None,
) -> dict[str, float | int]:
    if comparison_reference is None:
        comparison_rays = mask.reference_rays
        comparison_valid = mask.reference_valid
    else:
        comparison_rays, comparison_valid = unproject_camera(comparison_reference, mask.pixels)
        comparison_valid &= mask.reference_valid
    candidate_rays, candidate_valid = unproject_camera(candidate, mask.pixels)
    # valid_grid_ratio is always measured on the fixed common-reference mask;
    # comparison-reference validity is separately respected for ray angles.
    valid_on_fixed_mask = mask.reference_valid & candidate_valid
    valid = comparison_valid & candidate_valid
    dot = np.sum(comparison_rays * candidate_rays, axis=1)
    angles = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    def region(lower: float, upper: float) -> np.ndarray:
        return valid & (mask.rho >= lower) & (mask.rho < upper)

    full = angles[valid]
    central = angles[region(0.0, 0.4)]
    middle = angles[region(0.4, 0.7)]
    peripheral = angles[valid & (mask.rho >= 0.7) & (mask.rho <= 1.0)]
    mask_count = int(np.count_nonzero(mask.reference_valid))
    valid_count = int(np.count_nonzero(valid_on_fixed_mask))
    return {
        "valid_grid_ratio": valid_count / mask_count,
        "valid_grid_count": valid_count,
        "invalid_grid_count": mask_count - valid_count,
        "ray_mean_deg": float(np.mean(full)) if full.size else math.nan,
        "full_ray_median_deg": percentile_or_nan(full, 50),
        "full_ray_p95_deg": percentile_or_nan(full, 95),
        "full_ray_max_deg": float(np.max(full)) if full.size else math.nan,
        "central_ray_median_deg": percentile_or_nan(central, 50),
        "central_ray_p95_deg": percentile_or_nan(central, 95),
        "middle_ray_median_deg": percentile_or_nan(middle, 50),
        "middle_ray_p95_deg": percentile_or_nan(middle, 95),
        "peripheral_ray_median_deg": percentile_or_nan(peripheral, 50),
        "peripheral_ray_p95_deg": percentile_or_nan(peripheral, 95),
    }


def scale_label(scale: float) -> str:
    hundredths = scale * 100.0
    if math.isclose(hundredths, round(hundredths), abs_tol=1e-10):
        return f"scale_{int(round(hundredths)):03d}"
    return f"scale_{int(round(scale * 1000.0)):04d}"


def run_stage5_command(
    args: argparse.Namespace,
    profile: str,
    scale: float,
    mode: str,
    output: Path,
    reference_scene: Path | None,
) -> list[str]:
    runner = Path(__file__).with_name("run_stage5_from_mat.py")
    command = [
        str(args.python), str(runner),
        "--config", str(args.config.resolve()),
        "--models", args.model,
        "--target-mode", "multi_board",
        "--kalibr-camchain", str(args.reference_camchain.resolve()),
        "--output", str(output.resolve()),
        "--cache-dir", str((output / ".stage5_backend_cache").resolve()),
        "--backend", str(args.backend.resolve()),
        "--train-mat", str(args.train_mat.resolve()),
        "--test-mat", str(args.test_mat.resolve()),
        # Both branches consume the same frozen full observation set and build
        # the same pre-perturbation intermediate state. Outer-only drops the
        # internal residuals only after perturbation inside Stage5.
        "--include-internal-points", "1",
        "--stage5-large-intrinsic-perturbation", profile,
        "--stage5-large-intrinsic-perturbation-scale", f"{scale:.12g}",
        "--stage5-disable-selected-case-visualizations",
        "--stage5-enable-polar-angle-diagnostics",
    ]
    if args.strict_perturbation_scale:
        command.append("--stage5-large-intrinsic-perturbation-strict-scale")
    if mode == "outer_only":
        command.append(
            "--stage5-large-intrinsic-perturbation-outer-only-after-application"
        )
    if reference_scene is not None:
        command.extend(["--stage5-large-intrinsic-perturbation-reference-scene", str(reference_scene.resolve())])
    if args.fixed_backend_input_list is not None:
        command.extend([
            "--stage5-trial-backend-selection-force-include-frame-board-list",
            str(args.fixed_backend_input_list.resolve()),
            "--stage5-trial-backend-selection-candidate-order", "random_shuffle",
            "--stage5-trial-backend-selection-candidate-shuffle-seed",
            str(args.fixed_backend_shuffle_seed),
        ])
        if args.reject_unlisted_fixed_backend_input:
            command.extend([
                "--stage5-trial-backend-selection-mi-tol", "1e12",
            ])
    if args.fixed_backend_seed_list is not None:
        command.extend([
            "--stage5-trial-backend-selection-seed-frame-board-list",
            str(args.fixed_backend_seed_list.resolve()),
        ])
    return command


def completed_run(path: Path) -> bool:
    training = path / "backend_training_summary.txt"
    perturbation = path / "large_intrinsic_perturbation_summary.txt"
    if not training.is_file() or not perturbation.is_file():
        return False
    values = read_kv(perturbation)
    return (
        values.get("internal_observations_regenerated_after_perturbation") == "0"
        and values.get("selection_seed_matches_perturbed_camera") == "1"
        and values.get("selection_candidate_matches_perturbed_camera") == "1"
        and bool(values.get("frozen_observation_fingerprint"))
    )


def run_stage5(command: list[str], cwd: Path, output: Path, resume: bool, dry_run: bool) -> None:
    if resume and completed_run(output):
        print(f"= resume {output}", flush=True)
        return
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def pair_run(
    args: argparse.Namespace,
    repo: Path,
    profile: str,
    scale: float,
    root: Path,
) -> tuple[Path, Path]:
    outer = root / "outer_only"
    internal = root / "outer_internal"
    run_stage5(run_stage5_command(args, profile, scale, "outer_only", outer, None), repo, outer, args.resume, args.dry_run)
    snapshot = outer / "large_intrinsic_perturbation_reference_scene.txt"
    if not args.dry_run and not snapshot.is_file():
        raise RuntimeError(f"Outer-only branch did not write reference scene: {snapshot}")
    run_stage5(run_stage5_command(args, profile, scale, "outer_internal", internal, snapshot), repo, internal, args.resume, args.dry_run)
    return outer, internal


def holdout_metrics(path: Path) -> dict[str, str]:
    values = read_kv(path / "backend_holdout_summary.txt")
    return {
        "heldout_overall_rmse": values.get("overall_rmse", ""),
        "heldout_outer_rmse": values.get("outer_only_rmse", ""),
        "heldout_internal_rmse": values.get("internal_only_rmse", ""),
    }


def backend_schedule_fingerprints(path: Path) -> dict[str, str | int]:
    decisions = read_decisions(path)
    baseline = sorted({
        (int(row["frame_index"]), int(row["board_id"]))
        for row in decisions if row.get("baseline_seed") == "1"
    })
    attempted_groups: dict[int, set[tuple[int, int]]] = {}
    committed_groups: dict[int, set[tuple[int, int]]] = {}
    for row in decisions:
        raw_order = row.get("persistent_incremental_attempt_order", "")
        if not raw_order or int(raw_order) < 0:
            continue
        key = (int(row["frame_index"]), int(row["board_id"]))
        order = int(raw_order)
        if row.get("persistent_incremental_attempted") == "1":
            attempted_groups.setdefault(order, set()).add(key)
        if row.get("persistent_incremental_batch_accepted") == "1":
            committed_groups.setdefault(order, set()).add(key)

    def fingerprint(value: object) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    attempted_schedule = [
        [order, sorted(keys)] for order, keys in sorted(attempted_groups.items())
    ]
    committed_schedule = [
        [order, sorted(keys)] for order, keys in sorted(committed_groups.items())
    ]
    attempted_set = sorted(set(baseline).union(*attempted_groups.values())) if attempted_groups else baseline
    committed_set = sorted(set(baseline).union(*committed_groups.values())) if committed_groups else baseline
    return {
        "backend_seed_set_fingerprint": fingerprint(baseline),
        "backend_attempted_schedule_fingerprint": fingerprint(attempted_schedule),
        "backend_committed_schedule_fingerprint": fingerprint(committed_schedule),
        "backend_attempted_frame_board_set_fingerprint": fingerprint(attempted_set),
        "backend_committed_frame_board_set_fingerprint": fingerprint(committed_set),
        "backend_seed_frame_board_count": len(baseline),
        "backend_attempted_batch_count": len(attempted_groups),
        "backend_committed_batch_count": len(committed_groups),
        "backend_attempted_frame_board_count": len(attempted_set),
        "backend_committed_frame_board_count": len(committed_set),
    }


def branch_row(
    path: Path,
    profile: str,
    scale: float,
    mode: str,
    common_mask: EvaluationMask,
    common_reference: Camera,
    branch_clean: Camera,
    family: str,
) -> dict[str, Any]:
    perturb = read_kv(path / "large_intrinsic_perturbation_summary.txt")
    training = read_kv(path / "backend_training_summary.txt")
    runtime = read_kv(path / "runtime_summary.txt")
    initial = camera_from_intrinsics(
        perturb["perturbed_camera_intrinsics"],
        perturb.get("perturbed_camera_distortion", ""),
        family,
    )
    final = camera_from_training_summary(path / "backend_training_summary.txt", family)
    initial_common = ray_metrics(common_mask, initial)
    final_common = ray_metrics(common_mask, final)
    clean_common = ray_metrics(common_mask, branch_clean)
    clean_branch = ray_metrics(common_mask, branch_clean, branch_clean)
    initial_branch = ray_metrics(common_mask, initial, branch_clean)
    final_branch = ray_metrics(common_mask, final, branch_clean)
    row: dict[str, Any] = {
        "direction": profile,
        "scale": scale,
        "method": "Outer+Internal" if mode == "outer_internal" else "Outer-only",
        "mode": mode,
        "run_dir": str(path),
        "reference_scene_fingerprint": perturb.get("reference_scene_fingerprint", ""),
        "perturbed_scene_fingerprint": perturb.get("perturbed_scene_fingerprint", ""),
        "frozen_observation_fingerprint": perturb.get("frozen_observation_fingerprint", ""),
        "requested_scale": perturb.get("requested_scale", ""),
        "effective_scale": perturb.get("effective_scale", ""),
        "initial_xi": initial.xi,
        "initial_alpha": initial.alpha,
        "initial_fu": initial.fu,
        "initial_fv": initial.fv,
        "final_xi": final.xi,
        "final_alpha": final.alpha,
        "final_fu": final.fu,
        "final_fv": final.fv,
        "final_cu": final.cu,
        "final_cv": final.cv,
        "initial_camera_fingerprint": camera_fingerprint(initial),
        "final_camera_fingerprint": camera_fingerprint(final),
        "common_reference_camera_fingerprint": camera_fingerprint(common_reference),
        "max_abs_parameter_difference": max(
            abs(value_a - value_b) for value_a, value_b in zip(
                (final.xi, final.alpha, final.fu, final.fv, final.cu, final.cv),
                (common_reference.xi, common_reference.alpha, common_reference.fu,
                 common_reference.fv, common_reference.cu, common_reference.cv),
            )
        ),
        "solver_status": training.get("success", "0"),
        "runtime_sec": runtime.get("total_runtime_seconds", ""),
        "common_reference_xi": common_reference.xi,
        "common_reference_alpha": common_reference.alpha,
        "common_reference_fu": common_reference.fu,
        "common_reference_fv": common_reference.fv,
        "relative_fu_error_common": abs(final.fu - common_reference.fu) / abs(common_reference.fu),
        "relative_fv_error_common": abs(final.fv - common_reference.fv) / abs(common_reference.fv),
        "absolute_xi_error_common": abs(final.xi - common_reference.xi),
        "absolute_alpha_error_common": abs(final.alpha - common_reference.alpha),
        "relative_fu_error_branch_clean": abs(final.fu - branch_clean.fu) / abs(branch_clean.fu),
        "relative_fv_error_branch_clean": abs(final.fv - branch_clean.fv) / abs(branch_clean.fv),
        "absolute_xi_error_branch_clean": abs(final.xi - branch_clean.xi),
        "absolute_alpha_error_branch_clean": abs(final.alpha - branch_clean.alpha),
    }
    for index in range(4):
        label = f"k{index + 1}"
        initial_value = initial.distortion[index] if index < len(initial.distortion) else math.nan
        final_value = final.distortion[index] if index < len(final.distortion) else math.nan
        common_value = common_reference.distortion[index] if index < len(common_reference.distortion) else math.nan
        row[f"initial_{label}"] = initial_value
        row[f"final_{label}"] = final_value
        row[f"common_reference_{label}"] = common_value
        row[f"absolute_{label}_error_common"] = abs(final_value - common_value)
    for prefix, values in (("initial_common_", initial_common), ("final_common_", final_common), ("clean_common_", clean_common), ("initial_branch_", initial_branch), ("final_branch_", final_branch)):
        row.update({prefix + key: value for key, value in values.items()})
    row.update(holdout_metrics(path))
    row.update(backend_schedule_fingerprints(path))
    for name in ("full_ray_p95_deg", "peripheral_ray_p95_deg"):
        denominator = float(initial_branch[name]) - float(clean_branch[name])
        numerator = float(final_branch[name]) - float(clean_branch[name])
        row[name.replace("_deg", "_recovery")] = 1.0 - numerator / denominator if abs(denominator) > 1e-12 else math.nan
    return row


def read_decisions(path: Path) -> list[dict[str, str]]:
    source = path / "trial_backend_frame_board_selection_decisions.csv"
    if not source.is_file():
        return []
    with source.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_incremental_rows(path: Path, common_mask: EvaluationMask, profile: str, scale: float, mode: str, family: str) -> list[dict[str, Any]]:
    # The current checkpoint evaluator exports KB projection intrinsics but not
    # k1--k4. Ray trajectories would therefore be under-specified; final KB
    # metrics remain complete in perturbation_results.csv.
    if family == "kb":
        return []
    checkpoints = path / "persistent_camera_checkpoint_evaluations.csv"
    if not checkpoints.is_file():
        return []
    with checkpoints.open(newline="", encoding="utf-8") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    decisions = read_decisions(path)
    decision_by_order: dict[int, dict[str, str]] = {}
    for decision in decisions:
        raw_order = decision.get("persistent_incremental_attempt_order", "")
        if not raw_order or int(raw_order) < 0:
            continue
        order = int(raw_order)
        # Every row in an accepted frame batch repeats the same optimizer
        # diagnostics. Keep one representative row for trajectory metadata.
        if decision.get("persistent_incremental_batch_accepted") == "1":
            decision_by_order.setdefault(order, decision)
    seed_frames = {row["frame_index"] for row in decisions if row.get("baseline_seed") == "1"}
    seed_boards = {(row["frame_index"], row["board_id"]) for row in decisions if row.get("baseline_seed") == "1"}
    accepted_by_order: dict[int, list[dict[str, str]]] = {}
    for decision in decisions:
        if decision.get("persistent_incremental_batch_accepted") != "1":
            continue
        order = int(decision["persistent_incremental_attempt_order"])
        accepted_by_order.setdefault(order, []).append(decision)
    frame_set, board_set = set(seed_frames), set(seed_boards)
    result: list[dict[str, Any]] = []
    for checkpoint in sorted(checkpoint_rows, key=lambda row: int(row["attempt_order"])):
        order = int(checkpoint["attempt_order"])
        if order >= 0:
            for accepted_order in sorted(key for key in accepted_by_order if key <= order):
                for decision in accepted_by_order.pop(accepted_order):
                    frame_set.add(decision["frame_index"])
                    board_set.add((decision["frame_index"], decision["board_id"]))
        distortion = tuple(
            float(checkpoint.get(f"k{index}", "nan")) for index in range(1, 5)
        ) if family == "kb" else ()
        camera = Camera(
            *(float(checkpoint[key]) for key in ("xi", "alpha", "fu", "fv", "cu", "cv")),
            family="pinhole-equi" if family == "kb" else "ds-none",
            distortion=distortion,
        )
        metrics = ray_metrics(common_mask, camera)
        decision = decision_by_order.get(order, {})
        row: dict[str, Any] = {
            "direction": profile,
            "scale": scale,
            "method": "Outer+Internal" if mode == "outer_internal" else "Outer-only",
            "increment_index": len(result),
            "attempt_order": order,
            "selected_frame_count": len(frame_set),
            "selected_board_count": len(board_set),
            "xi": camera.xi,
            "alpha": camera.alpha,
            "fu": camera.fu,
            "fv": camera.fv,
            "cu": camera.cu,
            "cv": camera.cv,
            "information_gain": checkpoint.get("information_gain", ""),
            "holdout_rmse": checkpoint.get("test_rmse", ""),
            "solver_status": (
                "accepted_converged"
                if decision.get("persistent_incremental_converged_by_relative_objective") == "1"
                or decision.get("persistent_incremental_converged_by_camera_step") == "1"
                else "accepted_incremental_checkpoint"
            ),
            "iterations": decision.get("persistent_incremental_iterations", ""),
            "objective_start": decision.get("persistent_incremental_objective_start", ""),
            "objective_final": decision.get("persistent_incremental_objective_final", ""),
            "outer_residual_count": decision.get("outer_point_count", ""),
            "internal_residual_count": decision.get("internal_point_count", ""),
            "total_residual_count": decision.get("point_count", ""),
            "incremental_runtime_sec": decision.get("persistent_incremental_elapsed_time_seconds", ""),
        }
        row.update(metrics)
        result.append(row)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        formatted_rows = [
            {
                key: (f"{value:.12f}" if isinstance(value, (float, np.floating)) else value)
                for key, value in row.items()
            }
            for row in rows
        ]
        writer.writerows(formatted_rows)


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["direction"], row["scale"], row["mode"]): row for row in rows}
    result: list[dict[str, Any]] = []
    for direction, scale in sorted({(str(row["direction"]), float(row["scale"])) for row in rows}):
        if scale == 0.0:
            continue
        outer = indexed.get((direction, scale, "outer_only"))
        internal = indexed.get((direction, scale, "outer_internal"))
        if outer is None or internal is None:
            continue
        paired_reference_scene_identical = (
            outer["reference_scene_fingerprint"] == internal["reference_scene_fingerprint"]
        )
        paired_perturbed_scene_identical = (
            outer["perturbed_scene_fingerprint"] == internal["perturbed_scene_fingerprint"]
        )
        paired_frozen_observations_identical = (
            outer["frozen_observation_fingerprint"] == internal["frozen_observation_fingerprint"]
        )
        paired_initial_camera_identical = all(
            math.isclose(float(outer[name]), float(internal[name]), rel_tol=1e-12, abs_tol=1e-12)
            for name in ("initial_xi", "initial_alpha", "initial_fu", "initial_fv")
        )
        backend_seed_set_identical = (
            outer["backend_seed_set_fingerprint"] == internal["backend_seed_set_fingerprint"]
        )
        backend_attempted_schedule_identical = (
            outer["backend_attempted_schedule_fingerprint"] == internal["backend_attempted_schedule_fingerprint"]
        )
        backend_committed_schedule_identical = (
            outer["backend_committed_schedule_fingerprint"] == internal["backend_committed_schedule_fingerprint"]
        )
        backend_committed_set_identical = (
            outer["backend_committed_frame_board_set_fingerprint"] == internal["backend_committed_frame_board_set_fingerprint"]
        )
        result.append({
            "direction": direction,
            "scale": scale,
            "paired_reference_scene_identical": int(paired_reference_scene_identical),
            "paired_perturbed_scene_identical": int(paired_perturbed_scene_identical),
            "paired_frozen_observations_identical": int(paired_frozen_observations_identical),
            "paired_initial_camera_identical": int(paired_initial_camera_identical),
            "backend_seed_set_identical": int(backend_seed_set_identical),
            "backend_attempted_schedule_identical": int(backend_attempted_schedule_identical),
            "backend_committed_schedule_identical": int(backend_committed_schedule_identical),
            "backend_committed_set_identical": int(backend_committed_set_identical),
            "paired_initial_state_valid": int(
                paired_reference_scene_identical
                and paired_perturbed_scene_identical
                and paired_frozen_observations_identical
                and paired_initial_camera_identical
            ),
            "delta_full_ray_p95_deg": float(outer["final_common_full_ray_p95_deg"]) - float(internal["final_common_full_ray_p95_deg"]),
            "delta_peripheral_ray_p95_deg": float(outer["final_common_peripheral_ray_p95_deg"]) - float(internal["final_common_peripheral_ray_p95_deg"]),
            "outer_valid_grid_ratio": outer["final_common_valid_grid_ratio"],
            "internal_valid_grid_ratio": internal["final_common_valid_grid_ratio"],
        })
    return result


def summary_rows(rows: list[dict[str, Any]], profiles: list[str], scales: list[float]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    direction_summaries: list[dict[str, Any]] = []
    for direction in profiles:
        for mode in ("outer_only", "outer_internal"):
            matching = sorted((row for row in rows if row["direction"] == direction and row["mode"] == mode), key=lambda row: float(row["scale"]))
            if len(matching) != len(scales):
                continue
            x = np.asarray([float(row["scale"]) for row in matching])
            full = np.asarray([float(row["final_common_full_ray_p95_deg"]) for row in matching])
            peri = np.asarray([float(row["final_common_peripheral_ray_p95_deg"]) for row in matching])
            recovery = np.asarray([float(row["peripheral_ray_p95_recovery"]) for row in matching[1:]])
            finite_recovery = recovery[np.isfinite(recovery)]
            direction_summaries.append({
                "scope": direction,
                "method": "Outer+Internal" if mode == "outer_internal" else "Outer-only",
                "full_ray_auc_deg": float(np.trapezoid(full, x)),
                "peripheral_ray_auc_deg": float(np.trapezoid(peri, x)),
                "median_peripheral_recovery": (
                    float(np.median(finite_recovery))
                    if finite_recovery.size
                    else math.nan
                ),
                "invalid_runs": int(sum(float(row["final_common_valid_grid_ratio"]) < 1.0 for row in matching)),
            })
    result.extend(direction_summaries)
    for method in ("Outer-only", "Outer+Internal"):
        values = [row for row in direction_summaries if row["method"] == method]
        if not values:
            continue
        result.append({
            "scope": "mean_over_P1_P4",
            "method": method,
            "full_ray_auc_deg": float(np.mean([float(row["full_ray_auc_deg"]) for row in values])),
            "peripheral_ray_auc_deg": float(np.mean([float(row["peripheral_ray_auc_deg"]) for row in values])),
            "median_peripheral_recovery": float(np.median([float(row["median_peripheral_recovery"]) for row in values])),
            "invalid_runs": int(sum(int(row["invalid_runs"]) for row in values)),
        })
    return result


def main() -> int:
    args = parse_args()
    profiles = [value.strip().upper() for value in args.profiles.split(",") if value.strip()]
    unknown = [value for value in profiles if value not in PROFILES]
    if unknown:
        raise SystemExit(f"Unknown profiles: {', '.join(unknown)}")
    if args.model == "kb" and any(profile != "P1" for profile in profiles):
        raise SystemExit("KB currently defines only P1 as a focal-only perturbation")
    scales = parse_scales(args.scales)
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    if 0.0 not in scales:
        raise SystemExit("--scales must include 0 for branch-relative clean references")
    repo = Path(__file__).resolve().parents[2]
    output = args.output_root.resolve()
    if output.exists() and not args.resume and any(output.iterdir()):
        raise SystemExit(f"{output} already exists and is non-empty; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    (output / "incremental_logs").mkdir(exist_ok=True)
    if args.dry_run:
        pair_run(args, repo, "P1", 0.0, output / "reference" / "clean")
        for profile in profiles:
            for scale in scales:
                if scale > 0.0:
                    pair_run(args, repo, profile, scale, output / profile / scale_label(scale))
        return 0

    clean_outer, clean_internal = pair_run(args, repo, "P1", 0.0, output / "reference" / "clean")
    # The paired clean Outer+Internal run is the common reference. It is a
    # full-data, no-perturbation selection/incremental-BA run, while loading
    # the Outer-only pre-selection snapshot makes the clean ablation paired.
    common_path = clean_internal
    common_reference = camera_from_training_summary(
        common_path / "backend_training_summary.txt", args.model
    )
    width, height = image_size_from_mat(args.train_mat)
    mask = build_evaluation_mask(common_reference, width, height, args.grid_size)
    common_payload = {
        "dataset_id": args.dataset_id,
        "camera": args.camera,
        "model": args.model,
        "perturbation_definition": (
            "focal_scale_only_kb_coefficients_frozen_at_perturbation_boundary"
            if args.model == "kb"
            else "ds_focal_xi_alpha_joint"
        ),
        "camera_intrinsics": asdict(common_reference),
        "source_run": str(common_path),
        "evaluation_mask": {
            "width": width,
            "height": height,
            "grid_size": args.grid_size,
            "fixed_radius_px": mask.fixed_radius_px,
            "mask_grid_count": int(np.count_nonzero(mask.reference_valid)),
            "mask_rule": "common_reference_valid AND rho<=1 using the common-reference principal point and inscribed image disc",
            "black_border_note": "No pixel-validity mask is present in the MAT interchange; the inscribed disc excludes rectangular image corners and the common-reference validity domain is fixed for every candidate.",
        },
    }
    (output / "reference" / "common_reference.json").write_text(json.dumps(common_payload, indent=2) + "\n", encoding="utf-8")
    clean_by_mode = {
        "outer_only": camera_from_training_summary(
            clean_outer / "backend_training_summary.txt", args.model
        ),
        "outer_internal": camera_from_training_summary(
            clean_internal / "backend_training_summary.txt", args.model
        ),
    }
    completed: list[tuple[str, float, str, Path]] = [
        ("P1", 0.0, "outer_only", clean_outer),
        ("P1", 0.0, "outer_internal", clean_internal),
    ]
    conditions = [
        (profile, scale)
        for profile in profiles
        for scale in scales
        if scale > 0.0
    ]
    if args.jobs == 1:
        pair_paths = [
            (profile, scale, *pair_run(args, repo, profile, scale, output / profile / scale_label(scale)))
            for profile, scale in conditions
        ]
    else:
        print(f"running {len(conditions)} paired conditions with {args.jobs} workers", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                (profile, scale, executor.submit(pair_run, args, repo, profile, scale, output / profile / scale_label(scale)))
                for profile, scale in conditions
            ]
            pair_paths = [
                (profile, scale, *future.result())
                for profile, scale, future in futures
            ]
    for profile, scale, outer, internal in pair_paths:
        completed.extend([(profile, scale, "outer_only", outer), (profile, scale, "outer_internal", internal)])

    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for profile, scale, mode, path in completed:
        branch_clean = clean_by_mode[mode]
        rows.append(branch_row(
            path, profile, scale, mode, mask, common_reference, branch_clean,
            args.model,
        ))
        trajectories.extend(make_incremental_rows(
            path, mask, profile, scale, mode, args.model
        ))
    # P1 clean is the shared s=0 point for every profile. Copy its measured
    # clean rows into P2--P4 so each direction has a complete AUC curve.
    for profile in profiles:
        if profile == "P1":
            continue
        for mode in ("outer_only", "outer_internal"):
            source = next(row for row in rows if row["direction"] == "P1" and row["scale"] == 0.0 and row["mode"] == mode)
            copied = dict(source)
            copied["direction"] = profile
            copied["clean_source_direction"] = "P1"
            rows.append(copied)
    rows.sort(key=lambda row: (str(row["direction"]), float(row["scale"]), str(row["mode"])))
    write_csv(output / "perturbation_results.csv", rows)
    write_csv(output / "incremental_trajectory.csv", trajectories)
    paired = paired_rows(rows)
    write_csv(output / "paired_improvements.csv", paired)
    write_csv(output / "summary_metrics.csv", summary_rows(rows, profiles, scales))
    fingerprint_lines = []
    for pair in paired:
        fixed_backend_pair_valid = (
            pair["backend_seed_set_identical"] == 1
            and pair["backend_attempted_schedule_identical"] == 1
            and pair["backend_committed_schedule_identical"] == 1
            and pair["backend_committed_set_identical"] == 1
        )
        fingerprint_lines.append(
            f"{pair['direction']} scale={pair['scale']:.12g} "
            f"paired_initial_state_valid: {pair['paired_initial_state_valid']} "
            f"fixed_backend_pair_valid: {int(fixed_backend_pair_valid)}"
        )
        if pair["paired_initial_state_valid"] != 1:
            raise RuntimeError(
                f"Paired initial-state check failed for {pair['direction']} "
                f"scale={pair['scale']}"
            )
        if args.fixed_backend_input_list is not None and not fixed_backend_pair_valid:
            raise RuntimeError(
                f"Fixed Backend schedule check failed for {pair['direction']} "
                f"scale={pair['scale']}"
            )
    (output / "paired_state_check.txt").write_text("\n".join(fingerprint_lines) + "\n", encoding="utf-8")
    manifest = {
        "dataset_id": args.dataset_id,
        "model": args.model,
        "profiles": profiles,
        "scales": scales,
        "protocol": "perturb_after_internal_recovery_before_selection_incremental_ba",
        "internal_observations_regenerated_after_perturbation": False,
        "internal_observations_refiltered_after_perturbation": False,
        "fixed_backend_input_enabled": args.fixed_backend_input_list is not None,
        "fixed_backend_input_list": str(args.fixed_backend_input_list.resolve()) if args.fixed_backend_input_list is not None else "",
        "fixed_backend_shuffle_seed": args.fixed_backend_shuffle_seed,
        "common_reference": str(common_path),
        "common_reference_definition": "clean_outer_internal_paired_full_data_no_perturbation",
        "clean_outer_only": str(clean_outer),
        "clean_outer_internal": str(clean_internal),
        "run_count": len(completed),
        "parallel_pair_workers": args.jobs,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output / 'perturbation_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

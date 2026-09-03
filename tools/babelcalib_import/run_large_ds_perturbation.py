#!/usr/bin/env python3
"""Run paired large-DS-intrinsic perturbation experiments.

Each profile is run once with outer-only observations and once with the same
MAT input including frozen internal observations. The Stage5 flag applies the
perturbation after internal recovery and before selection/incremental BA.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SCIPY_PYTHON = Path(
    "/Users/linzhaoxian/.cache/codex-runtimes/"
    "codex-primary-runtime/dependencies/python/bin/python3"
)


PROFILES = {
    "P1": (0.70, -0.20, 0.10),
    "P2": (0.70, 0.20, -0.10),
    "P3": (1.30, -0.20, 0.10),
    "P4": (1.30, 0.20, -0.10),
}


@dataclass
class Camera:
    xi: float
    alpha: float
    fu: float
    fv: float
    cu: float
    cv: float


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-mat", type=Path, required=True)
    parser.add_argument("--test-mat", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--camera", choices=("left", "right"), required=True)
    parser.add_argument("--reference-camchain", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=repo / "aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml")
    parser.add_argument("--backend", type=Path, default=repo / "build/run_stage5_backend")
    parser.add_argument("--profiles", default="P1,P2,P3,P4")
    parser.add_argument("--python", type=Path, default=DEFAULT_SCIPY_PYTHON)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def camera_from_summary(values: dict[str, str], prefix: str = "camera_") -> Camera:
    return Camera(*(float(values[prefix + key]) for key in ("xi", "alpha", "fu", "fv", "cu", "cv")))


def camera_from_intrinsics(text: str) -> Camera:
    values = [float(value) for value in text.split(",")]
    if len(values) != 6:
        raise ValueError(f"Expected six DS parameters, got {text!r}")
    return Camera(*values)


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
    sqrt1 = np.sqrt(np.maximum(mz * mz + (1.0 - camera.xi * camera.xi) * r2, 0.0))
    ray_denom = mz * mz + r2
    good &= np.isfinite(ray_denom) & (np.abs(ray_denom) > 1e-12)
    k = np.full_like(mx, np.nan)
    k[good] = (mz[good] * camera.xi + sqrt1[good]) / ray_denom[good]
    rays = np.stack([k * mx, k * my, k * mz - camera.xi], axis=1)
    norms = np.linalg.norm(rays, axis=1)
    good &= np.isfinite(norms) & (norms > 1e-12)
    rays[good] /= norms[good, None]
    return rays, good


def ray_metrics(reference: Camera, candidate: Camera, width: int = 4512, height: int = 4512) -> dict[str, float | int]:
    axis = np.linspace(0.0, 1.0, 101)
    pixels = np.array([(x * (width - 1), y * (height - 1)) for y in axis for x in axis], dtype=np.float64)
    ref_rays, ref_valid = unproject_ds(reference, pixels)
    cand_rays, cand_valid = unproject_ds(candidate, pixels)
    both = ref_valid & cand_valid
    center = np.array([0.5 * (width - 1), 0.5 * (height - 1)])
    max_radius = float(np.linalg.norm(center))
    rho = np.linalg.norm(pixels - center[None, :], axis=1) / max_radius
    dot = np.sum(ref_rays * cand_rays, axis=1)
    angles = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    valid_angles = angles[both]
    peripheral = angles[both & (rho >= 0.7)]
    return {
        "valid_projection_grid": int(np.count_nonzero(both) == pixels.shape[0]),
        "valid_grid_count": int(np.count_nonzero(both)),
        "invalid_grid_count": int(pixels.shape[0] - np.count_nonzero(both)),
        "ray_median_deg": float(np.percentile(valid_angles, 50)) if valid_angles.size else math.nan,
        "ray_p95_deg": float(np.percentile(valid_angles, 95)) if valid_angles.size else math.nan,
        "ray_max_deg": float(np.max(valid_angles)) if valid_angles.size else math.nan,
        "peripheral_ray_p95_deg": float(np.percentile(peripheral, 95)) if peripheral.size else math.nan,
    }


def command_for(
    args: argparse.Namespace,
    profile: str,
    mode: str,
    output: Path,
    reference_scene: Path | None = None,
) -> list[str]:
    include_internal = "1" if mode == "outer_internal" else "0"
    runner = Path(__file__).with_name("run_stage5_from_mat.py")
    command = [
        str(args.python), str(runner),
        "--config", str(args.config.resolve()),
        "--models", "ds-none",
        "--target-mode", "multi_board",
        "--kalibr-camchain", str(args.reference_camchain.resolve()),
        "--output", str(output.resolve()),
        "--backend", str(args.backend.resolve()),
        "--train-mat", str(args.train_mat.resolve()),
        "--test-mat", str(args.test_mat.resolve()),
        "--include-internal-points", include_internal,
        "--stage5-large-intrinsic-perturbation", profile,
        "--stage5-disable-selected-case-visualizations",
        "--stage5-enable-polar-angle-diagnostics",
    ]
    if reference_scene is not None:
        command.extend([
            "--stage5-large-intrinsic-perturbation-reference-scene",
            str(reference_scene.resolve()),
        ])
    return command


def run_one(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def write_svg_plot(rows: list[dict[str, object]], profiles: list[str], path: Path) -> None:
    """Dependency-free publication plot; QuickLook converts it to PNG on macOS."""
    width, height = 1600, 1600
    left, right, top, bottom = 150, 70, 95, 190
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [float(row["ray_p95_deg"]) for row in rows]
    initial = [float(row["initial_ray_p95_deg"]) for row in rows]
    ymax = max(1.0, math.ceil(max(values + initial) / 5.0) * 5.0)
    colors = {"outer_only": "#1f77b4", "outer_internal": "#d62728"}
    labels = {"outer_only": "Outer-only", "outer_internal": "Outer+Internal"}

    def x_at(index: int) -> float:
        return left + (plot_w * index / max(1, len(profiles) - 1))

    def y_at(value: float) -> float:
        return top + plot_h * (1.0 - value / ymax)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#222} .small{font-size:24px} .label{font-size:30px} .title{font-size:36px;font-weight:600}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Recovery from large DS intrinsic perturbations</text>',
    ]
    for tick in range(0, int(ymax) + 1, max(1, int(ymax // 5))):
        y = y_at(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#d9dee5" stroke-width="1"/>')
        parts.append(f'<text x="{left-14}" y="{y+6:.2f}" text-anchor="end" class="small">{tick}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333" stroke-width="2"/>')
    parts.append(f'<text x="28" y="{top+plot_h/2}" transform="rotate(-90 28 {top+plot_h/2})" text-anchor="middle" class="label">Ray deviation P95 (deg)</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" class="label">DS intrinsic perturbation</text>')

    for index, profile in enumerate(profiles):
        x = x_at(index)
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+34}" text-anchor="middle" class="label">{profile}</text>')
        for mode in ("outer_only", "outer_internal"):
            row = next(row for row in rows if row["profile"] == profile and row["mode"] == mode)
            color = colors[mode]
            x0 = x - 8 if mode == "outer_only" else x + 8
            final_y = y_at(float(row["ray_p95_deg"]))
            init_y = y_at(float(row["initial_ray_p95_deg"]))
            parts.append(f'<line x1="{x0:.2f}" y1="{init_y:.2f}" x2="{x0:.2f}" y2="{final_y:.2f}" stroke="{color}" stroke-width="2" stroke-dasharray="7 6" opacity="0.45"/>')
            parts.append(f'<circle cx="{x0:.2f}" cy="{final_y:.2f}" r="7" fill="{color}"/>')
            parts.append(f'<path d="M{x0-6:.2f},{init_y-6:.2f} L{x0+6:.2f},{init_y+6:.2f} M{x0+6:.2f},{init_y-6:.2f} L{x0-6:.2f},{init_y+6:.2f}" stroke="{color}" stroke-width="2" opacity="0.65"/>')

    legend_x, legend_y = left + plot_w - 290, top + 25
    for offset, mode in enumerate(("outer_only", "outer_internal")):
        y = legend_y + offset * 32
        color = colors[mode]
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="6" fill="{color}"/><text x="{legend_x+16}" y="{y+6}" class="small">{labels[mode]} final</text>')
        parts.append(f'<path d="M{legend_x-6},{y+16} L{legend_x+6},{y+28} M{legend_x+6},{y+16} L{legend_x-6},{y+28}" stroke="{color}" stroke-width="2" opacity="0.65"/><text x="{legend_x+16}" y="{y+27}" class="small">{labels[mode]} initial</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    profiles = [value.strip().upper() for value in args.profiles.split(",") if value.strip()]
    unknown = [value for value in profiles if value not in PROFILES]
    if unknown:
        raise SystemExit(f"Unknown profiles: {', '.join(unknown)}")

    rows: list[dict[str, object]] = []
    for profile in profiles:
        reference_scene: Path | None = None
        for mode in ("outer_only", "outer_internal"):
            run_dir = output_root / profile / mode
            run_one(
                command_for(args, profile, mode, run_dir, reference_scene),
                repo,
                args.dry_run,
            )
            if args.dry_run:
                continue
            if mode == "outer_only":
                reference_scene = run_dir / "large_intrinsic_perturbation_reference_scene.txt"
                if not reference_scene.is_file():
                    raise RuntimeError(f"Outer-only run did not write {reference_scene}")
            perturb = read_kv(run_dir / "large_intrinsic_perturbation_summary.txt")
            training = read_kv(run_dir / "backend_training_summary.txt")
            holdout = read_kv(run_dir / "backend_holdout_summary.txt")
            reference = camera_from_intrinsics(perturb["reference_camera_intrinsics"])
            initial = camera_from_intrinsics(perturb["perturbed_camera_intrinsics"])
            final = camera_from_summary(training)
            initial_ray = ray_metrics(reference, initial)
            final_ray = ray_metrics(reference, final)
            row: dict[str, object] = {
                "profile": profile,
                "mode": mode,
                "run_dir": str(run_dir),
                "reference_scene_fingerprint": perturb.get("reference_scene_fingerprint", ""),
                "actual_focal_scale": perturb.get("actual_focal_scale", ""),
                "actual_xi_delta": perturb.get("actual_xi_delta", ""),
                "actual_alpha_delta": perturb.get("actual_alpha_delta", ""),
                "solver_success": training.get("success", "0"),
                "valid_projection_grid": final_ray["valid_projection_grid"],
                "initial_ray_median_deg": initial_ray["ray_median_deg"],
                "initial_ray_p95_deg": initial_ray["ray_p95_deg"],
                "initial_ray_max_deg": initial_ray["ray_max_deg"],
                "initial_peripheral_ray_p95_deg": initial_ray["peripheral_ray_p95_deg"],
                "ray_median_deg": final_ray["ray_median_deg"],
                "ray_p95_deg": final_ray["ray_p95_deg"],
                "ray_max_deg": final_ray["ray_max_deg"],
                "peripheral_ray_p95_deg": final_ray["peripheral_ray_p95_deg"],
                "relative_fu_error": abs(final.fu - reference.fu) / abs(reference.fu),
                "relative_fv_error": abs(final.fv - reference.fv) / abs(reference.fv),
                "absolute_xi_error": abs(final.xi - reference.xi),
                "absolute_alpha_error": abs(final.alpha - reference.alpha),
                "heldout_overall_rmse": holdout.get("overall_rmse", ""),
                "heldout_outer_rmse": holdout.get("outer_only_rmse", ""),
                "heldout_internal_rmse": holdout.get("internal_only_rmse", ""),
                "final_xi": final.xi,
                "final_alpha": final.alpha,
                "final_fu": final.fu,
                "final_fv": final.fv,
                "final_cu": final.cu,
                "final_cv": final.cv,
            }
            rows.append(row)

    if args.dry_run:
        return 0
    csv_path = output_root / "large_intrinsic_perturbation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fingerprints = {str(row["reference_scene_fingerprint"]) for row in rows}
    (output_root / "paired_state_check.txt").write_text(
        "reference_scene_fingerprint_count: " + str(len(fingerprints)) + "\n"
        + "paired_initial_scene_identical: " + str(int(len(fingerprints) == 1)) + "\n"
        + "note: a value other than 1 means the two ablation runs did not start from the same scene state.\n",
        encoding="utf-8",
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        positions = np.arange(len(profiles))
        fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=220)
        for mode, color, label in (("outer_only", "#1f77b4", "Outer-only"), ("outer_internal", "#d62728", "Outer+Internal")):
            values = [next(float(row["ray_p95_deg"]) for row in rows if row["profile"] == profile and row["mode"] == mode) for profile in profiles]
            initial_values = [next(float(row["initial_ray_p95_deg"]) for row in rows if row["profile"] == profile and row["mode"] == mode) for profile in profiles]
            ax.plot(positions, values, marker="o", linewidth=2.2, color=color, label=label)
            ax.plot(positions, initial_values, marker="x", linestyle="--", linewidth=1.2, alpha=0.55, color=color, label=label + " initial")
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.set_xticks(positions, profiles)
        ax.set_xlabel("DS intrinsic perturbation")
        ax.set_ylabel("Ray deviation P95 (deg)")
        ax.set_title("Recovery from large DS intrinsic perturbations")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(output_root / "large_intrinsic_perturbation_ray_p95.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as error:
        svg_path = output_root / "large_intrinsic_perturbation_ray_p95.svg"
        write_svg_plot(rows, profiles, svg_path)
        png_path = output_root / "large_intrinsic_perturbation_ray_p95.png"
        try:
            subprocess.run(
                ["qlmanage", "-t", "-s", "1600", "-o", str(output_root), str(svg_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            generated = output_root / (svg_path.name + ".png")
            if generated.is_file():
                generated.replace(png_path)
        except Exception as render_error:
            (output_root / "plot_warning.txt").write_text(
                f"matplotlib: {error}\nQuickLook: {render_error}\n",
                encoding="utf-8",
            )

    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

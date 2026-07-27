#!/usr/bin/env python3
"""Run and plot the three-panel DS-W1 paper experiment.

The final figure is drawn natively as one Matplotlib 1x3 canvas:
  (a) Peripheral ray recovery versus selected frame-board budget.
  (b) Radial pixel-displacement profile at a fixed perturbation level.
  (c) Aggregate frozen-W1 Schur information versus budget.

Modes:
  --mode experiments  Run missing source experiments only.
  --mode plot         Validate completed sources and draw the combined figure.
  --mode all          Run missing experiments, then draw the figure.

The default perturbation level is 2 degrees. Existing complete result
directories are reused; pass --force-experiments to recompute them in place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd

import plot_ds_w1_paired_diagnostics as paired
import run_ds_perturbation_sweep as sweep
import run_ds_weak_mode_perturbation as weak


METHODS = ("Outer-only", "Outer+Internal")
COLORS = {"Outer-only": "#D55E00", "Outer+Internal": "#0072B2"}
MARKERS = {"Outer-only": "o", "Outer+Internal": "s"}
TARGET_ERROR_DEG = 0.01


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("experiments", "plot", "all"), default="plot")
    parser.add_argument("--perturbation-level-deg", type=float, default=2.0)
    parser.add_argument("--peripheral-rho-threshold", type=float, default=0.7)
    parser.add_argument("--radial-bins", type=int, default=20)
    parser.add_argument("--prefix-seed-count", type=int, default=50)
    parser.add_argument("--sweep-seed-count", type=int, default=100)
    parser.add_argument("--noise-sigma-px", type=float, default=0.25)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--force-experiments", action="store_true")
    parser.add_argument(
        "--prefix-dir", type=Path,
        default=repo / "result_may/ds_incremental_prefix_observability_recovery_p95_2deg_20260721",
    )
    parser.add_argument(
        "--sweep-dir", type=Path,
        default=repo / "result_may/ds_w1_paper_statistics_20260721_right_levels_2_4",
    )
    parser.add_argument(
        "--information-dir", type=Path,
        default=repo / "result_may/ds_w1_information_vs_budget_20260721_right",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=repo / "result_may/ds_w1_paper_triptych_20260721_right",
    )
    parser.add_argument("--output-stem", default="ds_w1_recovery_information_triptych")
    parser.add_argument(
        "--sweep-levels", default="2,4",
        help="Levels generated when the local-recovery source experiment is missing.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def prefix_complete(args: argparse.Namespace) -> bool:
    try:
        summary = read_json(args.prefix_dir / "experiment_summary.json")
        return (
            (args.prefix_dir / "prefix_summary.csv").is_file()
            and (args.prefix_dir / "prefix_intrinsic_recovery_runs.csv").is_file()
            and close(float(summary["initial_peripheral_ray_p95_deg"]), args.perturbation_level_deg)
            and int(summary["seed_count"]) == args.prefix_seed_count
            and close(float(summary["noise_sigma_px"]), args.noise_sigma_px)
            and int(summary["accepted_group_count"]) == 208
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def sweep_complete(args: argparse.Namespace) -> bool:
    try:
        protocol = read_json(args.sweep_dir / "protocol_manifest.json")
        audit = read_json(args.sweep_dir / "run_audit.json")
        levels = {float(value) for value in protocol["levels_deg"]}
        return (
            args.perturbation_level_deg in levels
            and len(protocol["seeds"]) == args.sweep_seed_count
            and close(float(protocol["noise_sigma_px"]), args.noise_sigma_px)
            and int(audit["actual_run_count"]) == int(audit["expected_run_count"])
            and int(audit["unique_run_key_count"]) == int(audit["expected_run_count"])
            and (args.sweep_dir / "local_recovery_runs.csv").is_file()
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def information_complete(args: argparse.Namespace) -> bool:
    try:
        verification = read_json(args.information_dir / "verification_report.json")
        return (
            bool(verification["passed"])
            and (args.information_dir / "w1_information_by_budget.csv").is_file()
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def run_command(command: list[str], label: str) -> None:
    print(f"[{label}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def run_experiments(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[2]
    tool_dir = Path(__file__).resolve().parent
    fixed_root = repo / "result_may/ds_semi_synthetic_20260720_right_all_fixed_layout"

    if args.force_experiments or not prefix_complete(args):
        if args.prefix_dir.exists() and not args.force_experiments:
            raise RuntimeError(
                f"Existing prefix directory is incomplete or mismatched: {args.prefix_dir}. "
                "Use a new --prefix-dir or pass --force-experiments."
            )
        run_command([
            sys.executable, str(tool_dir / "run_ds_incremental_prefix_observability.py"),
            "--output", str(args.prefix_dir),
            "--budgets", "10,20,40,60,80,100",
            "--seed-count", str(args.prefix_seed_count),
            "--noise-sigma-px", str(args.noise_sigma_px),
            "--initial-peripheral-ray-p95-deg", str(args.perturbation_level_deg),
        ], "prefix recovery")
    else:
        print(f"[prefix recovery] reuse audited results: {args.prefix_dir}")

    if args.force_experiments or not sweep_complete(args):
        if args.sweep_dir.exists() and not args.force_experiments:
            raise RuntimeError(
                f"Existing sweep directory is incomplete or mismatched: {args.sweep_dir}. "
                "Use a new --sweep-dir or pass --force-experiments."
            )
        requested_levels = {
            float(value.strip()) for value in args.sweep_levels.split(",") if value.strip()
        }
        requested_levels.add(args.perturbation_level_deg)
        levels_text = ",".join(f"{value:g}" for value in sorted(requested_levels))
        run_command([
            sys.executable, str(tool_dir / "run_ds_w1_local_recovery_sweep.py"),
            "--reference-scene", str(fixed_root / "reference/clean_reference_scene.txt"),
            "--training-points", str(fixed_root / "precomputed/training/points.csv"),
            "--external-eval-mat", str(
                repo / "image/babelcalib_export/mul-board/"
                "babelcalib_multiboard_export_1444190clear_frontend_seed1337/all.mat"
            ),
            "--levels", levels_text,
            "--seed-count", str(args.sweep_seed_count),
            "--noise-sigma-px", str(args.noise_sigma_px),
            "--jobs", str(args.jobs),
            "--output", str(args.sweep_dir),
        ], "local W1 recovery")
    else:
        print(f"[local W1 recovery] reuse audited results: {args.sweep_dir}")

    if args.force_experiments or not information_complete(args):
        if args.information_dir.exists() and not args.force_experiments:
            raise RuntimeError(
                f"Existing information directory is incomplete: {args.information_dir}. "
                "Use a new --information-dir or pass --force-experiments."
            )
        run_command([
            sys.executable, str(tool_dir / "run_ds_w1_information_budget.py"),
            "--output", str(args.information_dir),
            "--noise-sigma-px", str(args.noise_sigma_px),
            "--dpi", str(args.dpi),
        ], "W1 information")
    else:
        print(f"[W1 information] reuse audited results: {args.information_dir}")


def validate_prefix(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not prefix_complete(args):
        raise RuntimeError("Prefix experiment is missing or does not match the 2-degree protocol")
    metadata = read_json(args.prefix_dir / "experiment_summary.json")
    summary = pd.read_csv(args.prefix_dir / "prefix_summary.csv")
    raw = pd.read_csv(args.prefix_dir / "prefix_intrinsic_recovery_runs.csv")
    required = {
        "budget_percent", "method", "final_peripheral_ray_p95_deg_median",
        "final_peripheral_ray_p95_deg_q25", "final_peripheral_ray_p95_deg_q75",
    }
    if required - set(summary.columns):
        raise ValueError(f"prefix_summary.csv missing {sorted(required - set(summary.columns))}")
    expected = {
        (budget, method)
        for budget in (10, 20, 40, 60, 80, 100) for method in METHODS
    }
    actual = set(zip(summary.budget_percent.astype(int), summary.method, strict=True))
    if actual != expected or summary.duplicated(["budget_percent", "method"]).any():
        raise RuntimeError("Prefix summary does not contain the complete paired budget grid")
    raw_required = {
        "seed", "budget_percent", "method", "solver_status",
        "final_peripheral_ray_p95_deg",
    }
    if raw_required - set(raw.columns):
        raise ValueError(
            "prefix_intrinsic_recovery_runs.csv missing "
            f"{sorted(raw_required - set(raw.columns))}"
        )
    raw_keys = set(zip(
        raw.seed.astype(int), raw.budget_percent.astype(int), raw.method, strict=True,
    ))
    expected_raw_keys = {
        (seed, budget, method)
        for seed in range(1, args.prefix_seed_count + 1)
        for budget in (10, 20, 40, 60, 80, 100)
        for method in METHODS
    }
    if (
        raw_keys != expected_raw_keys
        or raw.duplicated(["seed", "budget_percent", "method"]).any()
    ):
        raise RuntimeError("Raw prefix recovery runs do not form the complete paired grid")

    recomputed_rows: list[dict[str, Any]] = []
    for budget in (10, 20, 40, 60, 80, 100):
        for method in METHODS:
            group = raw[
                (raw.budget_percent.astype(int) == budget) & (raw.method == method)
            ]
            values = pd.to_numeric(
                group.final_peripheral_ray_p95_deg, errors="coerce",
            ).to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) != args.prefix_seed_count:
                raise RuntimeError(
                    f"Raw prefix group {budget}/{method} has {len(values)} finite runs"
                )
            row = {
                "budget_percent": budget,
                "method": method,
                "paired_trial_count": len(values),
                "final_peripheral_ray_p95_deg_median": float(np.median(values)),
                "final_peripheral_ray_p95_deg_q25": float(np.percentile(values, 25)),
                "final_peripheral_ray_p95_deg_q75": float(np.percentile(values, 75)),
            }
            recorded = summary[
                (summary.budget_percent.astype(int) == budget) &
                (summary.method == method)
            ].iloc[0]
            for suffix in ("median", "q25", "q75"):
                key = f"final_peripheral_ray_p95_deg_{suffix}"
                if not close(float(row[key]), float(recorded[key]), tolerance=1e-12):
                    raise RuntimeError(
                        f"Raw recomputation disagrees with prefix summary for "
                        f"{budget}/{method}/{suffix}"
                    )
            recomputed_rows.append(row)
    return pd.DataFrame(recomputed_rows), metadata


def validate_and_build_radial(
    args: argparse.Namespace, output: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not sweep_complete(args):
        raise RuntimeError("Local W1 sweep is missing or does not match the requested protocol")
    protocol = read_json(args.sweep_dir / "protocol_manifest.json")
    runs = pd.read_csv(args.sweep_dir / "local_recovery_runs.csv")
    runs["perturbation_level_deg"] = runs.perturbation_level_deg.astype(float)
    selected = runs[np.isclose(
        runs.perturbation_level_deg, args.perturbation_level_deg, atol=1e-12,
    )].copy()
    expected_seeds = {int(value) for value in protocol["seeds"]}
    expected_keys = {(seed, method) for seed in expected_seeds for method in METHODS}
    actual_keys = set(zip(selected.seed.astype(int), selected.method, strict=True))
    if actual_keys != expected_keys or selected.duplicated(["seed", "method"]).any():
        raise RuntimeError("Local W1 sweep does not contain a complete paired seed grid")
    scene = weak.parse_scene(Path(protocol["reference_scene"]))
    mask = sweep.build_evaluation_mask(
        scene.camera, int(protocol["image_width"]), int(protocol["image_height"]),
        int(protocol["grid_size"]),
    )
    radial = pd.DataFrame(paired.radial_rows(
        selected, mask, args.perturbation_level_deg, args.radial_bins,
    ))
    plot_columns = [
        "perturbation_level_deg", "method", "rho_bin_index", "rho_center",
        "pixel_p95_median_px", "pixel_p95_q25_px", "pixel_p95_q75_px",
        "valid_run_count",
    ]
    radial_plot = radial[plot_columns].copy()
    radial_plot.to_csv(
        output / "panel_b_radial_profile.csv", index=False, float_format="%.15g",
    )
    return radial_plot, protocol


def validate_information(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not information_complete(args):
        raise RuntimeError("W1 information experiment is missing or failed verification")
    frame = pd.read_csv(args.information_dir / "w1_information_by_budget.csv")
    required = {
        "budget_percent", "actual_budget_percent", "method", "point_count",
        "raw_w1_information", "prefix_fingerprint", "w1_fingerprint",
    }
    if required - set(frame.columns):
        raise ValueError(f"W1 information CSV missing {sorted(required - set(frame.columns))}")
    grouped = {
        method: frame[frame.method == method].sort_values("budget_percent")
        for method in METHODS
    }
    if any(len(grouped[method]) != 11 for method in METHODS):
        raise RuntimeError("W1 information requires 11 budgets for both methods")
    for left, right in zip(
        grouped["Outer-only"].to_dict("records"),
        grouped["Outer+Internal"].to_dict("records"), strict=True,
    ):
        if (
            int(left["budget_percent"]) != int(right["budget_percent"])
            or left["prefix_fingerprint"] != right["prefix_fingerprint"]
            or left["w1_fingerprint"] != right["w1_fingerprint"]
        ):
            raise RuntimeError("W1 information branches do not share budget/prefix/W1")
    return frame, read_json(args.information_dir / "experiment_summary.json")


def validate_cross_experiment_prefix(
    args: argparse.Namespace,
    prefix_meta: dict[str, Any],
    information_meta: dict[str, Any],
) -> None:
    prefix_schedule = pd.read_csv(args.prefix_dir / "accepted_frame_board_schedule.csv")
    information_schedule = pd.read_csv(
        args.information_dir / "accepted_frame_board_schedule.csv"
    )
    columns = ["rank", "frame_id", "board_id"]
    for name, frame in (
        ("prefix", prefix_schedule), ("information", information_schedule),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} schedule is missing {sorted(missing)}")
    left = prefix_schedule[columns].astype(int).reset_index(drop=True)
    right = information_schedule[columns].astype(int).reset_index(drop=True)
    if len(left) != 208 or not left.equals(right):
        raise RuntimeError(
            "Panels (a) and (c) do not use the same ordered 208-group prefix"
        )
    prefix_hashes = prefix_meta["source_hashes"]
    information_hashes = information_meta["source_hashes"]
    key_pairs = (
        ("decisions", "selection_decisions"),
        ("scene", "reference_scene"),
        ("points", "training_points"),
    )
    mismatched = [
        f"{left_key}/{right_key}"
        for left_key, right_key in key_pairs
        if prefix_hashes[left_key] != information_hashes[right_key]
    ]
    if mismatched:
        raise RuntimeError(
            "Panels (a) and (c) have different source hashes: " + ", ".join(mismatched)
        )


def configure_style(dpi: int) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": dpi,
    })


def style_axis(axis: plt.Axes, panel: str) -> None:
    axis.grid(axis="y", color="#D4D4D4", linewidth=0.4, alpha=0.38)
    axis.set_axisbelow(True)
    axis.text(
        0.0, 1.06, panel, transform=axis.transAxes,
        va="top", ha="left", fontfamily="DejaVu Serif",
        fontweight="bold", fontsize=8.2,
    )


def plot_method_curve(
    axis: plt.Axes, x: np.ndarray, y: np.ndarray, method: str,
    *, label: bool = True, markevery: int | None = None,
) -> None:
    axis.plot(
        x, y, color=COLORS[method], marker=MARKERS[method],
        markersize=3.4, markeredgecolor="white", markeredgewidth=0.4,
        linewidth=1.6, label=method if label else "_nolegend_",
        markevery=markevery,
    )


def make_figure(args: argparse.Namespace) -> tuple[Path, Path]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix_frame, prefix_meta = validate_prefix(args)
    radial, sweep_meta = validate_and_build_radial(args, output)
    information, information_meta = validate_information(args)
    validate_cross_experiment_prefix(args, prefix_meta, information_meta)
    information.to_csv(output / "panel_a_w1_information.csv", index=False)
    prefix_frame.to_csv(output / "panel_c_prefix_recovery.csv", index=False)
    for legacy_name in ("panel_a_prefix_recovery.csv", "panel_c_w1_information.csv"):
        (output / legacy_name).unlink(missing_ok=True)

    target_budgets: dict[str, int] = {}
    for method in METHODS:
        group = prefix_frame[prefix_frame.method == method].sort_values("budget_percent")
        reached = group[
            group.final_peripheral_ray_p95_deg_median <= TARGET_ERROR_DEG
        ]
        if reached.empty:
            raise RuntimeError(
                f"{method} never reaches the pre-specified {TARGET_ERROR_DEG:g}-degree target"
            )
        target_budgets[method] = int(reached.iloc[0].budget_percent)
        summary_key = f"{method}_budget_at_peripheral_ray_p95_le_001deg"
        recorded = prefix_meta.get(summary_key)
        if recorded is None or int(recorded) != target_budgets[method]:
            raise RuntimeError(
                f"Recomputed target budget for {method} disagrees with experiment summary: "
                f"recomputed={target_budgets[method]}, recorded={recorded}"
            )

    configure_style(args.dpi)
    fig, axes = plt.subplots(1, 3, figsize=(7.18, 2.48))

    # (a) Deterministic aggregate Schur information along one frozen W1 direction.
    axis = axes[0]
    info_groups = {
        method: information[information.method == method].sort_values("budget_percent")
        for method in METHODS
    }
    for method in METHODS:
        group = info_groups[method]
        plot_method_curve(
            axis, group.actual_budget_percent.to_numpy(float),
            group.raw_w1_information.to_numpy(float), method,
        )
    outer = info_groups["Outer-only"]
    internal = info_groups["Outer+Internal"]
    gains = (
        internal.raw_w1_information.to_numpy(float) /
        outer.raw_w1_information.to_numpy(float)
    )
    axis.set_yscale("log")
    axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Selected frame-board budget (%)")
    axis.set_ylabel(r"Weak-direction information, $I_{W_1}$")
    axis.set_xticks([5, 20, 40, 60, 80, 100])
    axis.set_xlim(1, 103)
    axis.set_ylim(1.4e3, 1.45e6)
    axis.text(
        0.98, 0.07,
        f"Median gain: {np.median(gains):.2f}$\\times$\n"
        f"At 100% budget: {gains[-1]:.2f}$\\times$",
        transform=axis.transAxes, va="bottom", ha="right", fontsize=6.8,
        color="#333333",
    )
    style_axis(axis, "(a)")

    # (b) Fixed-grid pixel displacement, summarized over paired noise seeds.
    axis = axes[1]
    for method in METHODS:
        group = radial[radial.method == method].sort_values("rho_center")
        x = group.rho_center.to_numpy(float)
        median = group.pixel_p95_median_px.to_numpy(float)
        q25 = group.pixel_p95_q25_px.to_numpy(float)
        q75 = group.pixel_p95_q75_px.to_numpy(float)
        axis.fill_between(x, q25, q75, color=COLORS[method], alpha=0.14, linewidth=0)
        plot_method_curve(axis, x, median, method, label=False, markevery=2)
    axis.axvline(
        args.peripheral_rho_threshold, color="#666666", linestyle=(0, (2, 2)),
        linewidth=0.8,
    )
    axis.text(
        args.peripheral_rho_threshold + 0.015, 0.95,
        rf"$\rho={args.peripheral_rho_threshold:g}$",
        transform=axis.get_xaxis_transform(), va="top", ha="left",
        fontsize=6.6, color="#555555",
    )
    axis.set_xlabel(r"Normalized radius $\rho$")
    axis.set_ylabel("Pixel-displacement error, P95 (px)")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.set_xticks([0.0, 0.25, 0.5, 0.7, 1.0])
    style_axis(axis, "(b)")

    # (c) Paired intrinsic recovery over the shared selected frame-board prefix.
    axis = axes[2]
    for method in METHODS:
        group = prefix_frame[prefix_frame.method == method].sort_values("budget_percent")
        x = group.budget_percent.to_numpy(float)
        median = group.final_peripheral_ray_p95_deg_median.to_numpy(float)
        q25 = group.final_peripheral_ray_p95_deg_q25.to_numpy(float)
        q75 = group.final_peripheral_ray_p95_deg_q75.to_numpy(float)
        axis.fill_between(x, q25, q75, color=COLORS[method], alpha=0.14, linewidth=0)
        plot_method_curve(axis, x, median, method, label=False)
    axis.set_xlabel("Selected frame-board budget (%)")
    axis.set_ylabel("Peripheral ray error, P95 (deg)")
    axis.set_xticks([10, 20, 40, 60, 80, 100])
    axis.set_xlim(7, 103)
    axis.set_ylim(bottom=0.0)
    axis.axhline(
        TARGET_ERROR_DEG, color="#777777", linestyle=(0, (2, 2)),
        linewidth=0.75, zorder=0,
    )
    axis.text(
        0.03, 0.96,
        rf"Target error: ${TARGET_ERROR_DEG:g}^\circ$",
        transform=axis.transAxes, va="top", ha="left", fontsize=6.4,
        color="#555555",
    )
    style_axis(axis, "(c)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 1.01), handlelength=2.2, columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.21, top=0.80, wspace=0.37)

    pdf = output / f"{args.output_stem}.pdf"
    png = output / f"{args.output_stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    key_values = [
        {
            "panel": "a", "metric": "median_information_gain",
            "value": float(np.median(gains)), "unit": "ratio",
            "aggregation": "median_over_budgets",
        },
        {
            "panel": "a", "metric": "information_gain_at_100pct_budget",
            "value": float(gains[-1]), "unit": "ratio",
            "aggregation": "deterministic",
        },
        {
            "panel": "b", "metric": "initial_peripheral_ray_perturbation",
            "value": args.perturbation_level_deg, "unit": "deg",
            "aggregation": "protocol_constant",
        },
        {
            "panel": "b", "metric": "peripheral_rho_threshold",
            "value": args.peripheral_rho_threshold, "unit": "normalized_radius",
            "aggregation": "protocol_constant",
        },
        {
            "panel": "c", "metric": "pre_specified_target_error",
            "value": TARGET_ERROR_DEG, "unit": "deg",
            "aggregation": "protocol_constant",
        },
        {
            "panel": "c", "metric": "first_budget_reaching_target_outer_only",
            "value": target_budgets["Outer-only"], "unit": "percent",
            "aggregation": "recomputed_from_raw_paired_trials",
        },
        {
            "panel": "c", "metric": "first_budget_reaching_target_outer_internal",
            "value": target_budgets["Outer+Internal"], "unit": "percent",
            "aggregation": "recomputed_from_raw_paired_trials",
        },
    ]
    pd.DataFrame(key_values).to_csv(
        output / "figure_key_values.csv", index=False, float_format="%.15g",
    )

    caption = (
        "DS weak-direction recovery and information analysis. "
        "(a) Deterministic aggregate information along $W_1$. $W_1$ is computed once from "
        "the 100% Outer-only Schur Fisher matrix and frozen for both branches and all "
        "budgets; frame poses and non-reference board poses are Schur-eliminated. "
        f"(b) Radial pixel-displacement P95 over {args.sweep_seed_count} paired trials. "
        f"(c) Peripheral ray P95 versus selected frame-board budget over "
        f"{args.prefix_seed_count} paired trials. Panels (b) and (c) use an "
        f"intrinsic-only recovery experiment initialized with a "
        f"{args.perturbation_level_deg:g}$^\\circ$ peripheral-ray perturbation, while "
        "frame and board poses are fixed. Solid curves and markers in panels (b) and "
        "(c) denote paired-trial medians; shaded regions denote IQR. The dashed line in "
        f"panel (b) marks $\\rho={args.peripheral_rho_threshold:g}$. The pre-specified "
        f"target error in panel (c) is {TARGET_ERROR_DEG:g}$^\\circ$; recomputation from "
        f"the raw paired trials gives first-hit budgets of "
        f"{target_budgets['Outer+Internal']}% for Outer+Internal and "
        f"{target_budgets['Outer-only']}% for Outer-only. "
        "Panel (a) is deterministic and therefore has no uncertainty band. "
        "Together, the panels relate increased aggregate weak-direction information "
        "to reduced radial model error and accurate recovery at smaller observation "
        "budgets."
    )
    (output / f"{args.output_stem}_caption.md").write_text(caption + "\n", encoding="utf-8")
    latex_caption = caption.replace("%", "\\%")
    latex = (
        "\\begin{figure*}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=0.98\\textwidth]{{{args.output_stem}.pdf}}\n"
        f"  \\caption{{{latex_caption}}}\n"
        "  \\label{fig:ds-w1-recovery-information}\n"
        "\\end{figure*}\n"
    )
    (output / f"{args.output_stem}_include.tex").write_text(latex, encoding="utf-8")

    manifest = {
        "schema": "ds_w1_recovery_information_triptych_v1",
        "perturbation_level_deg": args.perturbation_level_deg,
        "noise_sigma_px": args.noise_sigma_px,
        "peripheral_rho_threshold": args.peripheral_rho_threshold,
        "prefix_seed_count": args.prefix_seed_count,
        "sweep_seed_count": args.sweep_seed_count,
        "panel_order": [
            "aggregate_weak_direction_information",
            "radial_pixel_displacement_error",
            "peripheral_ray_recovery_vs_budget",
        ],
        "panel_a_w1_source": information_meta["w1_source"],
        "panel_a_w1_fingerprint": information_meta["w1_fingerprint"],
        "panel_b_protocol_levels_deg": sweep_meta["levels_deg"],
        "panel_c_initial_peripheral_ray_p95_deg": prefix_meta["initial_peripheral_ray_p95_deg"],
        "panel_c_pre_specified_target_error_deg": TARGET_ERROR_DEG,
        "panel_c_first_budget_reaching_target": target_budgets,
        "panel_a_c_shared_ordered_208_group_prefix_verified": True,
        "panel_a_c_shared_source_hashes_verified": True,
        "source_paths": {
            "prefix_summary": str((args.prefix_dir / "prefix_summary.csv").resolve()),
            "prefix_raw_runs": str(
                (args.prefix_dir / "prefix_intrinsic_recovery_runs.csv").resolve()
            ),
            "local_recovery_runs": str((args.sweep_dir / "local_recovery_runs.csv").resolve()),
            "w1_information": str((args.information_dir / "w1_information_by_budget.csv").resolve()),
        },
        "source_hashes": {
            "prefix_summary": sha256_file(args.prefix_dir / "prefix_summary.csv"),
            "prefix_raw_runs": sha256_file(
                args.prefix_dir / "prefix_intrinsic_recovery_runs.csv"
            ),
            "local_recovery_runs": sha256_file(args.sweep_dir / "local_recovery_runs.csv"),
            "w1_information": sha256_file(args.information_dir / "w1_information_by_budget.csv"),
        },
        "output_pdf": str(pdf),
        "output_png": str(png),
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    return pdf, png


def main() -> int:
    args = parse_args()
    args.prefix_dir = args.prefix_dir.resolve()
    args.sweep_dir = args.sweep_dir.resolve()
    args.information_dir = args.information_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if (
        args.perturbation_level_deg <= 0.0 or args.noise_sigma_px < 0.0
        or args.prefix_seed_count <= 0 or args.sweep_seed_count <= 0
        or args.radial_bins <= 0 or args.jobs <= 0 or args.dpi <= 0
        or not 0.0 < args.peripheral_rho_threshold <= 1.0
    ):
        raise ValueError("Invalid perturbation, noise, count, rho, jobs, or dpi argument")
    if args.mode in ("experiments", "all"):
        run_experiments(args)
    if args.mode in ("plot", "all"):
        pdf, png = make_figure(args)
        print(f"PDF: {pdf}")
        print(f"PNG: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

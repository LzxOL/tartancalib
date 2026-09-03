#!/usr/bin/env python3
"""Measure frozen-W1 Schur information over a shared frame-board budget prefix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import run_ds_incremental_prefix_observability as prefix
import run_ds_weak_mode_perturbation as weak


METHODS = ("Outer-only", "Outer+Internal")
COLORS = {"Outer-only": "#D55E00", "Outer+Internal": "#0072B2"}
DEFAULT_BUDGETS = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
COORDINATES = ("xi", "alpha", "log_fu", "log_fv", "cu_over_width", "cv_over_height")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selection-decisions", type=Path,
        default=prefix.DEFAULT_ROOT / "outer_only/trial_backend_frame_board_selection_decisions.csv",
        help="Outer-only selection log defining the frozen frame-board order.",
    )
    parser.add_argument(
        "--reference-scene", type=Path,
        default=prefix.DEFAULT_ROOT / "outer_only/final_persistent_backend_scene.txt",
        help="Fixed camera, frame poses, and board layout used to linearize the Fisher matrices.",
    )
    parser.add_argument(
        "--training-points", type=Path,
        default=prefix.DEFAULT_ROOT / "outer_internal/backend_training_points.csv",
        help="Point source containing both outer and frozen internal observations.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budgets", default=",".join(map(str, DEFAULT_BUDGETS)),
        help="Comma-separated frame-board prefix percentages; must include 100.",
    )
    parser.add_argument(
        "--noise-sigma-px", type=float, default=0.25,
        help="Common isotropic pixel sigma used only to scale the Fisher weight.",
    )
    parser.add_argument("--width", type=int, default=4512)
    parser.add_argument("--height", type=int, default=4512)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate figures and captions from existing audited CSV artifacts.",
    )
    return parser.parse_args()


def array_fingerprint(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.float64)
    return "sha256:" + hashlib.sha256(value.tobytes()).hexdigest()


def configure_style(dpi: int) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "savefig.dpi": dpi,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(
    axis: plt.Axes,
    panel: str,
    actual_budgets: np.ndarray,
    budget_labels: tuple[int, ...],
) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.35)
    axis.set_axisbelow(True)
    tick_indices = np.asarray(
        sorted(set([0, len(actual_budgets) // 2, len(actual_budgets) - 1])),
        dtype=int,
    )
    axis.set_xticks(actual_budgets[tick_indices])
    axis.set_xticklabels([f"{budget_labels[index]}" for index in tick_indices])
    axis.set_xlim(
        max(0.0, float(actual_budgets[0]) - 4.0),
        min(103.0, float(actual_budgets[-1]) + 3.0),
    )
    axis.text(
        -0.14, 1.08, panel, transform=axis.transAxes,
        va="top", ha="left", fontfamily="DejaVu Serif",
        fontweight="bold", fontsize=8,
    )


def method_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row["method"] == method),
        key=lambda row: int(row["budget_percent"]),
    )


def save_figure(fig: plt.Figure, output: Path, stem: str, dpi: int) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        output / f"{stem}.png", dpi=dpi,
        bbox_inches="tight", pad_inches=0.04,
    )
    plt.close(fig)


def plot_information(
    rows: list[dict[str, Any]], output: Path, budgets: tuple[int, ...], dpi: int,
) -> None:
    configure_style(dpi)
    by_method = {method: method_rows(rows, method) for method in METHODS}
    actual_budgets = np.asarray([
        row["actual_budget_percent"] for row in by_method["Outer-only"]
    ], dtype=float)
    gains = np.asarray([
        internal["raw_w1_information"] / outer["raw_w1_information"]
        for outer, internal in zip(
            by_method["Outer-only"], by_method["Outer+Internal"], strict=True,
        )
    ], dtype=float)
    full_outer = by_method["Outer-only"][-1]
    full_internal = by_method["Outer+Internal"][-1]
    point_ratio = full_internal["point_count"] / full_outer["point_count"]
    per_point_ratio = (
        full_internal["normalized_w1_information_per_2d_point_weight"] /
        full_outer["normalized_w1_information_per_2d_point_weight"]
    )
    full_gain = gains[-1]
    median_gain = float(np.median(gains))

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.62))
    markers = {"Outer-only": "o", "Outer+Internal": "s"}
    for method in METHODS:
        selected = by_method[method]
        x = np.asarray([row["actual_budget_percent"] for row in selected], dtype=float)
        axes[0].plot(
            x, [row["raw_w1_information"] for row in selected],
            color=COLORS[method], marker=markers[method], markersize=3.2,
            linewidth=1.5, label=method,
        )
        axes[2].plot(
            x, [row["normalized_w1_information_per_2d_point_weight"] for row in selected],
            color=COLORS[method], marker=markers[method], markersize=3.2,
            linewidth=1.5, label=method,
        )
    axes[1].plot(
        actual_budgets, gains, color=COLORS["Outer+Internal"], marker="s",
        markersize=3.2, linewidth=1.5,
    )
    axes[1].axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)

    axes[0].set_yscale("log")
    axes[0].set_title("Aggregate Weak-direction Information", fontsize=8.2, pad=7)
    axes[0].set_ylabel(r"Raw $I_{W1}=d_{W1}^{\mathsf{T}}S_c d_{W1}$")
    axes[1].set_title("Information Gain over Outer-only", fontsize=8.2, pad=7)
    axes[1].set_ylabel(r"$I_{W1}^{\mathrm{O+I}}/I_{W1}^{\mathrm{Outer}}$ ($\times$)")
    axes[2].set_title("Per-point Directional Information", fontsize=8.2, pad=7)
    axes[2].set_ylabel(r"$I_{W1}/\sum_i w_i$")
    axes[2].set_ylim(0.0, 21.5)
    for axis in axes:
        axis.set_xlabel("Actual selected frame-board budget (%)")
    for axis, panel in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        style_axis(axis, panel, actual_budgets, budgets)

    annotation_box = {
        "boxstyle": "round,pad=0.22", "facecolor": "white",
        "edgecolor": "#C8C8C8", "linewidth": 0.5, "alpha": 0.94,
    }
    axes[0].text(
        0.04, 0.95,
        f"median gain {median_gain:.2f}$\\times$\n100%: {full_gain:.2f}$\\times$",
        transform=axes[0].transAxes, va="top", ha="left", fontsize=7.2,
        bbox=annotation_box,
    )
    axes[1].text(
        0.04, 0.08,
        f"min {np.min(gains):.2f}$\\times$   max {np.max(gains):.2f}$\\times$\n"
        f"100%: {full_gain:.2f}$\\times$",
        transform=axes[1].transAxes, va="bottom", ha="left", fontsize=7.2,
        bbox=annotation_box,
    )
    axes[2].text(
        0.98, 0.97, "Control for observation count",
        transform=axes[2].transAxes, va="top", ha="right", fontsize=7.0,
        color="#555555",
    )
    axes[2].text(
        0.98, 0.56,
        f"100%: {point_ratio:.2f}$\\times$ points $\\times$ "
        f"{per_point_ratio:.2f}$\\times$ per-point\n"
        f"$\\approx$ {full_gain:.2f}$\\times$ total information",
        transform=axes[2].transAxes, va="center", ha="right", fontsize=6.8,
        bbox=annotation_box,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2,
        frameon=False, bbox_to_anchor=(0.5, 1.015), handlelength=2.0,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.77, wspace=0.36)
    save_figure(fig, output, "w1_information_vs_frame_board_budget", dpi)


def plot_supplementary_fisher_diagnostics(
    rows: list[dict[str, Any]], output: Path, budgets: tuple[int, ...], dpi: int,
) -> None:
    configure_style(dpi)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    actual_budgets = np.asarray([
        row["actual_budget_percent"] for row in method_rows(rows, "Outer-only")
    ], dtype=float)
    markers = {"Outer-only": "o", "Outer+Internal": "s"}
    for method in METHODS:
        selected = method_rows(rows, method)
        x = [row["actual_budget_percent"] for row in selected]
        axes[0].plot(
            x,
            [row["smallest_positive_eigenvalue"] for row in selected],
            color=COLORS[method], marker=markers[method], markersize=3.2,
            linewidth=1.5, label=method,
        )
        axes[1].plot(
            x, [row["log10_condition_number"] for row in selected],
            color=COLORS[method], marker=markers[method], markersize=3.2,
            linewidth=1.5, label=method,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Smallest positive eigenvalue")
    axes[1].set_ylabel(r"$\log_{10}$ condition number")
    for axis, panel in zip(axes, ("(a)", "(b)"), strict=True):
        axis.set_xlabel("Actual selected frame-board budget (%)")
        style_axis(axis, panel, actual_budgets, budgets)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.80, wspace=0.28)
    save_figure(fig, output, "supplementary_fisher_spectrum_diagnostics", dpi)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_information_rows(path: Path) -> list[dict[str, Any]]:
    numeric_fields = {
        "budget_percent": int,
        "actual_budget_percent": float,
        "frame_board_count": int,
        "frame_count": int,
        "board_count": int,
        "point_count": int,
        "raw_w1_information": float,
        "normalized_w1_information_per_2d_point_weight": float,
        "smallest_positive_eigenvalue": float,
        "log10_condition_number": float,
    }
    rows: list[dict[str, Any]] = []
    for source in read_csv(path):
        row: dict[str, Any] = dict(source)
        for field, converter in numeric_fields.items():
            if field not in row:
                raise ValueError(f"Missing required plot field {field!r} in {path}")
            row[field] = converter(row[field])
        rows.append(row)
    return rows


def write_figure_caption(rows: list[dict[str, Any]], output: Path) -> None:
    by_method = {method: method_rows(rows, method) for method in METHODS}
    if any(not selected for selected in by_method.values()):
        raise ValueError("Both Fisher branches are required to write the caption")
    group_count = int(by_method["Outer-only"][-1]["frame_board_count"])
    gains = [
        internal["raw_w1_information"] / outer["raw_w1_information"]
        for outer, internal in zip(
            by_method["Outer-only"], by_method["Outer+Internal"], strict=True,
        )
    ]
    full_outer = by_method["Outer-only"][-1]
    full_internal = by_method["Outer+Internal"][-1]
    point_ratio = full_internal["point_count"] / full_outer["point_count"]
    per_point_ratio = (
        full_internal["normalized_w1_information_per_2d_point_weight"] /
        full_outer["normalized_w1_information_per_2d_point_weight"]
    )
    caption = (
        "**Weak-direction information versus selected frame-board budget.** "
        "The intrinsic direction W1 is computed once as the weakest eigenvector of "
        "the 100% Outer-only Schur Fisher matrix and then frozen for every method and "
        f"budget. Both branches share the same {group_count}-group frame-board prefix; "
        "frame poses and all non-reference board poses are Schur-eliminated. "
        "(a) Aggregate directional information on a logarithmic axis. "
        "(b) The corresponding Outer+Internal gain over Outer-only. "
        "(c) Directional information normalized by the sum of effective 2D-point "
        f"weights, controlling for observation count. At 100%, {point_ratio:.2f}x "
        f"as many points and {per_point_ratio:.2f}x per-point information yield "
        f"{gains[-1]:.2f}x aggregate information. The Fisher calculation is "
        "deterministic, so no uncertainty bands are shown."
    )
    (output / "figure_caption.md").write_text(caption + "\n", encoding="utf-8")


def verify_artifacts(
    output: Path, budgets: tuple[int, ...], w1: np.ndarray,
) -> dict[str, Any]:
    rows = read_csv(output / "w1_information_by_budget.csv")
    expected_count = 2 * len(budgets)
    checks: dict[str, bool] = {
        "row_count_correct": len(rows) == expected_count,
        "all_fisher_matrices_are_6x6": True,
        "all_matrix_fingerprints_match": True,
        "all_raw_information_recomputed": True,
        "all_normalization_recomputed": True,
        "all_method_prefixes_identical": True,
        "all_w1_fingerprints_identical": len({row["w1_fingerprint"] for row in rows}) == 1,
        "all_intrinsic_fisher_ranks_are_six": all(
            int(row["intrinsic_fisher_rank"]) == 6 for row in rows
        ),
    }
    by_budget: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        budget = int(row["budget_percent"])
        by_budget.setdefault(budget, {})[row["method"]] = row
        matrix = np.loadtxt(row["fisher_matrix_path"], delimiter=",")
        checks["all_fisher_matrices_are_6x6"] &= matrix.shape == (6, 6)
        checks["all_matrix_fingerprints_match"] &= (
            array_fingerprint(matrix) == row["fisher_matrix_fingerprint"]
        )
        raw = float(w1 @ matrix @ w1)
        checks["all_raw_information_recomputed"] &= math.isclose(
            raw, float(row["raw_w1_information"]), rel_tol=1e-11, abs_tol=1e-8,
        )
        weight = float(row["effective_2d_point_weight_sum"])
        checks["all_normalization_recomputed"] &= math.isclose(
            raw / weight,
            float(row["normalized_w1_information_per_2d_point_weight"]),
            rel_tol=1e-11, abs_tol=1e-12,
        )
    checks["budget_set_correct"] = set(by_budget) == set(budgets)
    for budget in budgets:
        pair = by_budget.get(budget, {})
        checks["all_method_prefixes_identical"] &= (
            set(pair) == set(METHODS) and
            pair["Outer-only"]["prefix_fingerprint"] ==
            pair["Outer+Internal"]["prefix_fingerprint"]
        )
    full_outer = by_budget.get(100, {}).get("Outer-only")
    checks["full_outer_w1_equals_smallest_eigenvalue"] = bool(
        full_outer is not None and math.isclose(
            float(full_outer["raw_w1_information"]),
            float(full_outer["smallest_positive_eigenvalue"]),
            rel_tol=1e-10, abs_tol=1e-7,
        )
    )
    report = {
        "schema": "frozen_outer_w1_information_budget_verification_v1",
        "checks": checks,
        "passed": all(checks.values()),
    }
    (output / "verification_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    if not report["passed"]:
        raise RuntimeError(f"W1 information budget verification failed: {checks}")
    return report


def main() -> int:
    args = parse_args()
    if args.noise_sigma_px <= 0.0:
        raise ValueError("--noise-sigma-px must be positive")
    if args.width <= 0 or args.height <= 0 or args.dpi <= 0:
        raise ValueError("width, height, and dpi must be positive")
    budgets = prefix.parse_budgets(args.budgets)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix_dir = output / "fisher_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        information_path = output / "w1_information_by_budget.csv"
        direction_path = output / "frozen_w1_direction.csv"
        if not information_path.is_file() or not direction_path.is_file():
            raise FileNotFoundError(
                "--plot-only requires w1_information_by_budget.csv and "
                "frozen_w1_direction.csv in --output"
            )
        information_rows = load_information_rows(information_path)
        observed_budgets = tuple(sorted({
            int(row["budget_percent"]) for row in information_rows
        }))
        if observed_budgets != budgets:
            raise ValueError(
                f"CSV budgets {observed_budgets} do not match --budgets {budgets}"
            )
        direction_rows = sorted(
            read_csv(direction_path), key=lambda row: int(row["coordinate_index"]),
        )
        w1 = np.asarray([float(row["value"]) for row in direction_rows], dtype=float)
        verify_artifacts(output, budgets, w1)
        write_figure_caption(information_rows, output)
        plot_information(information_rows, output, budgets, args.dpi)
        plot_supplementary_fisher_diagnostics(
            information_rows, output, budgets, args.dpi,
        )
        print(f"Regenerated Fisher figures from {information_path}")
        return 0

    decisions = args.selection_decisions.resolve()
    scene_path = args.reference_scene.resolve()
    points_path = args.training_points.resolve()
    for path in (decisions, scene_path, points_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    scene = weak.parse_scene(scene_path)
    rows = prefix.load_rows(points_path, scene)
    available = {(row["frame"], row["board"]) for row in rows}
    schedule = [
        key for key in prefix.schedule_from_decisions(decisions)
        if key in available
    ]
    if set(schedule) != available:
        raise RuntimeError(
            "The 100% selection prefix must cover every available frame-board group: "
            f"schedule={len(set(schedule))}, available={len(available)}"
        )
    prefixes = {
        budget: schedule[:math.ceil(len(schedule) * budget / 100.0)]
        for budget in budgets
    }
    prefix.write_csv(output / "accepted_frame_board_schedule.csv", [
        {"rank": index + 1, "frame_id": key[0], "board_id": key[1]}
        for index, key in enumerate(schedule)
    ])
    prefix.write_csv(output / "prefix_manifest.csv", [
        {
            "budget_percent": budget,
            "actual_budget_percent": 100.0 * len(keys) / len(schedule),
            "frame_board_count": len(keys),
            "frame_count": len({key[0] for key in keys}),
            "prefix_fingerprint": prefix.digest(keys),
        }
        for budget, keys in prefixes.items()
    ])

    inverse_variance = 1.0 / (args.noise_sigma_px * args.noise_sigma_px)
    full_outer = prefix.select_rows(rows, set(prefixes[100]), "Outer-only")
    full_outer_fisher, full_audit = prefix.fisher(
        scene, full_outer, args.width, args.height, inverse_variance,
    )
    full_eigenvalues, full_eigenvectors = np.linalg.eigh(
        0.5 * (full_outer_fisher + full_outer_fisher.T)
    )
    w1 = weak.canonicalize_direction(full_eigenvectors[:, 0])
    w1_fingerprint = array_fingerprint(w1)
    prefix.write_csv(output / "frozen_w1_direction.csv", [
        {
            "coordinate_index": index,
            "coordinate": coordinate,
            "value": float(value),
            "w1_fingerprint": w1_fingerprint,
        }
        for index, (coordinate, value) in enumerate(zip(COORDINATES, w1, strict=True))
    ])

    information_rows: list[dict[str, Any]] = []
    for budget, keys in prefixes.items():
        key_set = set(keys)
        prefix_fingerprint = prefix.digest(keys)
        for method in METHODS:
            selected = prefix.select_rows(rows, key_set, method)
            matrix, audit = prefix.fisher(
                scene, selected, args.width, args.height, inverse_variance,
            )
            eigenvalues, rank, smallest, log_condition = prefix.eig_summary(matrix)
            raw_information = float(w1 @ matrix @ w1)
            point_weight_sum = inverse_variance * len(selected)
            scalar_residual_weight_sum = 2.0 * point_weight_sum
            matrix_stem = method.lower().replace("+", "_").replace("-", "_")
            matrix_path = matrix_dir / f"budget_{budget:03d}_{matrix_stem}.csv"
            np.savetxt(matrix_path, matrix, delimiter=",", fmt="%.17g")
            information_rows.append({
                "budget_percent": budget,
                "actual_budget_percent": 100.0 * len(keys) / len(schedule),
                "method": method,
                "frame_board_count": len(keys),
                "frame_count": len({key[0] for key in keys}),
                "board_count": len({key[1] for key in keys}),
                "point_count": len(selected),
                "scalar_residual_count": 2 * len(selected),
                "raw_w1_information": raw_information,
                "effective_2d_point_weight_sum": point_weight_sum,
                "effective_scalar_residual_weight_sum": scalar_residual_weight_sum,
                "normalized_w1_information_per_2d_point_weight": (
                    raw_information / point_weight_sum
                ),
                "normalized_w1_information_per_scalar_residual_weight": (
                    raw_information / scalar_residual_weight_sum
                ),
                "smallest_positive_eigenvalue": smallest,
                "intrinsic_fisher_rank": rank,
                "log10_condition_number": log_condition,
                "frame_pose_nuisance_dimension": 6 * int(audit["frame_count"]),
                "board_pose_nuisance_dimension": 6 * (int(audit["board_count"]) - 1),
                "schur_nuisance_dimension": int(audit["nuisance_dimension"]),
                "schur_nuisance_rank": int(audit["nuisance_rank"]),
                "reference_board_id": int(audit["reference_board_id"]),
                "prefix_fingerprint": prefix_fingerprint,
                "w1_fingerprint": w1_fingerprint,
                "fisher_matrix_fingerprint": array_fingerprint(matrix),
                "fisher_matrix_path": str(matrix_path),
                "fisher_eigenvalues": json.dumps(eigenvalues.tolist()),
            })

    gains: list[dict[str, Any]] = []
    for budget in budgets:
        pair = {
            row["method"]: row
            for row in information_rows if row["budget_percent"] == budget
        }
        outer = pair["Outer-only"]
        internal = pair["Outer+Internal"]
        if outer["prefix_fingerprint"] != internal["prefix_fingerprint"]:
            raise RuntimeError(f"Method prefixes differ at budget {budget}")
        gains.append({
            "budget_percent": budget,
            "actual_budget_percent": outer["actual_budget_percent"],
            "frame_board_count": outer["frame_board_count"],
            "outer_raw_w1_information": outer["raw_w1_information"],
            "outer_internal_raw_w1_information": internal["raw_w1_information"],
            "raw_gain_internal_over_outer": (
                internal["raw_w1_information"] / outer["raw_w1_information"]
            ),
            "outer_normalized_w1_information": (
                outer["normalized_w1_information_per_2d_point_weight"]
            ),
            "outer_internal_normalized_w1_information": (
                internal["normalized_w1_information_per_2d_point_weight"]
            ),
            "normalized_gain_internal_over_outer": (
                internal["normalized_w1_information_per_2d_point_weight"] /
                outer["normalized_w1_information_per_2d_point_weight"]
            ),
            "prefix_fingerprint": outer["prefix_fingerprint"],
            "w1_fingerprint": w1_fingerprint,
        })

    prefix.write_csv(output / "w1_information_by_budget.csv", information_rows)
    prefix.write_csv(output / "w1_information_gain_by_budget.csv", gains)
    full_directional_information = float(w1 @ full_outer_fisher @ w1)
    full_eigenvalue_relative_error = abs(
        full_directional_information - float(full_eigenvalues[0])
    ) / max(abs(float(full_eigenvalues[0])), 1.0)
    if full_eigenvalue_relative_error > 1e-10:
        raise RuntimeError("Frozen W1 does not match the 100% Outer-only weakest eigenvector")
    verification = verify_artifacts(output, budgets, w1)

    summary = {
        "protocol": "frozen_outer_w1_frame_board_budget_v1",
        "budgets_percent": list(budgets),
        "accepted_frame_board_count": len(schedule),
        "available_frame_board_count": len(available),
        "schedule_covers_all_available_groups": set(schedule) == available,
        "outer_and_internal_prefixes_identical": True,
        "w1_source": "100_percent_outer_only_schur_fisher",
        "w1_coordinate_order": list(COORDINATES),
        "w1_direction": w1.tolist(),
        "w1_fingerprint": w1_fingerprint,
        "w1_frozen_for_all_budgets_and_methods": (
            len({row["w1_fingerprint"] for row in information_rows}) == 1
        ),
        "schur_eliminates": "all observed frame poses and all non-reference board poses",
        "reference_board_id": int(full_audit["reference_board_id"]),
        "weight_model": "isotropic unit point weight scaled by 1/sigma_px^2",
        "normalization_primary": "raw_w1_information / sum effective 2D point weights",
        "noise_sigma_px": args.noise_sigma_px,
        "full_outer_weakest_eigenvalue": float(full_eigenvalues[0]),
        "full_outer_w1_directional_information": full_directional_information,
        "full_outer_w1_eigenvalue_relative_error": full_eigenvalue_relative_error,
        "verification_passed": verification["passed"],
        "source_paths": {
            "selection_decisions": str(decisions),
            "reference_scene": str(scene_path),
            "training_points": str(points_path),
        },
        "source_hashes": {
            "selection_decisions": prefix.digest_file(decisions),
            "reference_scene": prefix.digest_file(scene_path),
            "training_points": prefix.digest_file(points_path),
        },
        "raw_gain_internal_over_outer": {
            "min": min(row["raw_gain_internal_over_outer"] for row in gains),
            "median": float(np.median([
                row["raw_gain_internal_over_outer"] for row in gains
            ])),
            "max": max(row["raw_gain_internal_over_outer"] for row in gains),
        },
        "normalized_gain_internal_over_outer": {
            "min": min(row["normalized_gain_internal_over_outer"] for row in gains),
            "median": float(np.median([
                row["normalized_gain_internal_over_outer"] for row in gains
            ])),
            "max": max(row["normalized_gain_internal_over_outer"] for row in gains),
        },
    }
    (output / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    write_figure_caption(information_rows, output)
    plot_information(information_rows, output, budgets, args.dpi)
    plot_supplementary_fisher_diagnostics(information_rows, output, budgets, args.dpi)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

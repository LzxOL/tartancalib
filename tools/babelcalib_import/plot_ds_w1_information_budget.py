#!/usr/bin/env python3
"""Plot frozen-W1 directional information from audited budget CSV artifacts.

The default paper figure contains only Aggregate Weak-direction Information.

Optional panels already implemented in this file:
  --panels gain       Information Gain over Outer-only
  --panels per-point  Per-point Directional Information
  --panels all        Aggregate + gain + per-point as the original 1x3 figure

No Fisher matrix is recomputed. All plotted values come from
``w1_information_by_budget.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("Outer-only", "Outer+Internal")
COLORS = {"Outer-only": "#D55E00", "Outer+Internal": "#0072B2"}
MARKERS = {"Outer-only": "o", "Outer+Internal": "s"}
PANEL_CHOICES = ("aggregate", "gain", "per-point")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing w1_information_by_budget.csv.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory; defaults to --input-dir.",
    )
    parser.add_argument(
        "--panels", default="aggregate",
        help=(
            "Comma-separated panels: aggregate,gain,per-point; use 'all' for all "
            "three. The default produces only the requested aggregate figure."
        ),
    )
    parser.add_argument(
        "--output-stem", default="aggregate_weak_direction_information",
        help="Output filename without extension.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def parse_panels(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return PANEL_CHOICES
    panels = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not panels or len(set(panels)) != len(panels):
        raise ValueError("--panels must contain one or more unique panel names")
    unknown = set(panels) - set(PANEL_CHOICES)
    if unknown:
        raise ValueError(f"Unknown panels: {sorted(unknown)}")
    return panels


def configure_style(dpi: int) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.handlelength": 2.2,
        "legend.handletextpad": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "savefig.dpi": dpi,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_rows(path: Path) -> list[dict[str, Any]]:
    converters = {
        "budget_percent": int,
        "actual_budget_percent": float,
        "frame_board_count": int,
        "point_count": int,
        "raw_w1_information": float,
        "normalized_w1_information_per_2d_point_weight": float,
    }
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        row: dict[str, Any] = dict(source)
        for name, converter in converters.items():
            if name not in row:
                raise ValueError(f"Missing required column {name!r} in {path}")
            row[name] = converter(row[name])
        rows.append(row)
    return rows


def rows_by_method(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {
        method: sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["budget_percent"],
        )
        for method in METHODS
    }
    if any(not result[method] for method in METHODS):
        raise ValueError("CSV must contain both Outer-only and Outer+Internal")
    outer = result["Outer-only"]
    internal = result["Outer+Internal"]
    if len(outer) != len(internal):
        raise ValueError("The two branches contain different numbers of budgets")
    for left, right in zip(outer, internal, strict=True):
        if left["budget_percent"] != right["budget_percent"]:
            raise ValueError("The two branches use different nominal budgets")
        if not np.isclose(
            left["actual_budget_percent"], right["actual_budget_percent"], atol=1e-12,
        ):
            raise ValueError("The two branches use different actual budgets")
        if left["frame_board_count"] != right["frame_board_count"]:
            raise ValueError("The two branches do not share the same group prefix")
        if left.get("prefix_fingerprint") != right.get("prefix_fingerprint"):
            raise ValueError("The two branches have different prefix fingerprints")
        if left.get("w1_fingerprint") != right.get("w1_fingerprint"):
            raise ValueError("The two branches have different frozen W1 fingerprints")
    return result


def style_axis(
    axis: plt.Axes,
    panel_label: str,
    actual_budgets: np.ndarray,
    nominal_budgets: np.ndarray,
) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.35)
    axis.set_axisbelow(True)
    target_nominal_ticks = {5, 20, 40, 60, 80, 100}
    indices = np.asarray([
        index for index, value in enumerate(nominal_budgets)
        if int(value) in target_nominal_ticks
    ], dtype=int)
    axis.set_xticks(actual_budgets[indices])
    axis.set_xticklabels([f"{nominal_budgets[index]:g}" for index in indices])
    axis.set_xlim(
        max(0.0, float(actual_budgets[0]) - 4.0),
        min(103.0, float(actual_budgets[-1]) + 3.0),
    )
    if panel_label:
        axis.text(
            -0.14, 1.08, panel_label, transform=axis.transAxes,
            va="top", ha="left", fontfamily="DejaVu Serif",
            fontweight="bold", fontsize=8,
        )


def plot(args: argparse.Namespace) -> tuple[Path, Path]:
    panels = parse_panels(args.panels)
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_dir / "w1_information_by_budget.csv")
    grouped = rows_by_method(rows)

    outer = grouped["Outer-only"]
    internal = grouped["Outer+Internal"]
    actual = np.asarray([row["actual_budget_percent"] for row in outer], dtype=float)
    nominal = np.asarray([row["budget_percent"] for row in outer], dtype=float)
    gains = np.asarray([
        right["raw_w1_information"] / left["raw_w1_information"]
        for left, right in zip(outer, internal, strict=True)
    ])
    point_ratio = internal[-1]["point_count"] / outer[-1]["point_count"]
    per_point_ratio = (
        internal[-1]["normalized_w1_information_per_2d_point_weight"] /
        outer[-1]["normalized_w1_information_per_2d_point_weight"]
    )

    configure_style(args.dpi)
    width = 3.42 if len(panels) == 1 else 2.35 * len(panels)
    height = 2.48 if len(panels) == 1 else 2.62
    fig, axes_value = plt.subplots(1, len(panels), figsize=(width, height), squeeze=False)
    axes = list(axes_value[0])
    annotation_box = {
        "boxstyle": "round,pad=0.22", "facecolor": "white",
        "edgecolor": "#C8C8C8", "linewidth": 0.5, "alpha": 0.94,
    }

    for index, (panel, axis) in enumerate(zip(panels, axes, strict=True)):
        panel_label = f"({chr(ord('a') + index)})" if len(panels) > 1 else ""
        if panel == "aggregate":
            for method in METHODS:
                selected = grouped[method]
                axis.plot(
                    actual, [row["raw_w1_information"] for row in selected],
                    color=COLORS[method], marker=MARKERS[method], markersize=3.8,
                    markeredgecolor="white", markeredgewidth=0.45,
                    linewidth=1.7, label=method,
                )
            axis.set_yscale("log")
            axis.set_ylabel(r"Weak-direction information, $I_{W_1}$")
            axis.set_ylim(1.4e3, 1.45e6)
            axis.text(
                0.035, 0.955, f"Median gain: {np.median(gains):.2f}$\\times$",
                transform=axis.transAxes, va="top", ha="left", fontsize=7.2,
                color="#333333",
            )
            outer_final = float(outer[-1]["raw_w1_information"])
            internal_final = float(internal[-1]["raw_w1_information"])
            bracket_x = 103.0
            axis.annotate(
                "", xy=(bracket_x, internal_final), xytext=(bracket_x, outer_final),
                arrowprops={
                    "arrowstyle": "<->", "color": "#555555",
                    "linewidth": 0.8, "shrinkA": 0, "shrinkB": 0,
                },
                annotation_clip=False,
            )
            axis.text(
                105.0, np.sqrt(outer_final * internal_final),
                f"{gains[-1]:.2f}$\\times$", va="center", ha="left",
                fontsize=7.0, color="#333333", clip_on=False,
            )
            axis.set_xlim(max(0.0, float(actual[0]) - 4.0), 112.0)
        elif panel == "gain":
            axis.plot(
                actual, gains, color=COLORS["Outer+Internal"], marker="s",
                markersize=3.2, linewidth=1.5,
            )
            axis.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
            axis.set_ylabel(r"$I_{W1}^{\mathrm{O+I}}/I_{W1}^{\mathrm{Outer}}$ ($\times$)")
            axis.text(
                0.04, 0.08,
                f"min {np.min(gains):.2f}$\\times$   max {np.max(gains):.2f}$\\times$\n"
                f"100%: {gains[-1]:.2f}$\\times$",
                transform=axis.transAxes, va="bottom", ha="left", fontsize=7.2,
                bbox=annotation_box,
            )
        else:
            for method in METHODS:
                selected = grouped[method]
                axis.plot(
                    actual,
                    [row["normalized_w1_information_per_2d_point_weight"]
                     for row in selected],
                    color=COLORS[method], marker=MARKERS[method], markersize=3.2,
                    linewidth=1.5, label=method,
                )
            axis.set_ylabel(r"$I_{W1}/\sum_i w_i$")
            axis.set_ylim(0.0, 21.5)
            axis.text(
                0.98, 0.97, "Control for observation count",
                transform=axis.transAxes, va="top", ha="right", fontsize=7.0,
                color="#555555",
            )
            axis.text(
                0.98, 0.56,
                f"100%: {point_ratio:.2f}$\\times$ points $\\times$ "
                f"{per_point_ratio:.2f}$\\times$ per-point\n"
                f"$\\approx$ {gains[-1]:.2f}$\\times$ total information",
                transform=axis.transAxes, va="center", ha="right", fontsize=6.8,
                bbox=annotation_box,
            )
        axis.set_xlabel("Selected frame-board budget (%)")
        style_axis(axis, panel_label, actual, nominal)
        if panel == "aggregate":
            axis.set_xlim(max(0.0, float(actual[0]) - 4.0), 112.0)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for axis in axes[1:]:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=2,
            frameon=False, bbox_to_anchor=(0.5, 1.015), handlelength=2.0,
        )
        top = 0.80 if len(panels) == 1 else 0.77
    else:
        top = 0.86
    fig.subplots_adjust(
        left=0.17 if len(panels) == 1 else 0.075,
        right=0.94 if len(panels) == 1 else 0.985,
        bottom=0.20, top=top,
        wspace=0.36,
    )

    pdf_path = output_dir / f"{args.output_stem}.pdf"
    png_path = output_dir / f"{args.output_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    if panels == ("aggregate",):
        caption = (
            "Aggregate weak-direction information as a function of the actual "
            "selected frame-board budget. "
            "$I_{W_1}=\\mathbf{d}_{W_1}^{\\mathsf{T}}S_c\\mathbf{d}_{W_1}$, "
            "where $W_1$ is computed once from the 100% Outer-only Schur Fisher "
            "matrix and frozen for both branches and all budgets. Both branches use "
            "the same 208-group prefix; frame poses and non-reference board poses are "
            "Schur-eliminated. Outer+Internal provides a median 4.16$\\times$ gain "
            "and a 4.11$\\times$ gain at the full budget."
        )
        (output_dir / f"{args.output_stem}_caption.md").write_text(
            caption + "\n", encoding="utf-8",
        )
        latex = (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=0.48\\textwidth]{{{args.output_stem}.pdf}}\n"
            f"  \\caption{{{caption}}}\n"
            "  \\label{fig:w1-aggregate-information}\n"
            "\\end{figure}\n"
        )
        (output_dir / f"{args.output_stem}_include.tex").write_text(
            latex, encoding="utf-8",
        )
    return pdf_path, png_path


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    pdf_path, png_path = plot(args)
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

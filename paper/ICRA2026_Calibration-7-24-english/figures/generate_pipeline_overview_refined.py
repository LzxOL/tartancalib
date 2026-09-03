#!/usr/bin/env python3
"""Render a compact, single-column overview of the calibration pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "pic"

# A restrained palette: neutral modules are shared, while orange and blue only
# identify the Outer4 and internal-observation streams, respectively.
INK = "#1F2933"
MUTED = "#60707E"
NEUTRAL = "#718096"
NEUTRAL_FILL = "#F7F9FB"
OUTER = "#C65A25"
OUTER_FILL = "#FDF2EA"
INIT = "#A77918"
INIT_FILL = "#FCF7E7"
INTERNAL = "#1976A8"
INTERNAL_FILL = "#EDF6FB"
FINAL = "#37895E"
FINAL_FILL = "#EFF7F1"


def rounded_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    detail: str | None,
    edge: str,
    fill: str,
    title_size: float = 7.0,
    detail_size: float = 5.3,
    title_weight: str = "semibold",
) -> None:
    """Draw one compact process module with an understated accent bar."""
    radius = 0.045
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            facecolor=fill,
            edgecolor="#D8E0E6",
            linewidth=0.65,
            zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + height - 0.052),
            width,
            0.052,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            facecolor=edge,
            edgecolor=edge,
            linewidth=0,
            zorder=3,
        )
    )
    title_y = y + (height * 0.60 if detail else height * 0.50)
    ax.text(
        x + width / 2,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight=title_weight,
        color=INK,
        zorder=4,
    )
    if detail:
        ax.text(
            x + width / 2,
            y + height * 0.29,
            detail,
            ha="center",
            va="center",
            fontsize=detail_size,
            color=MUTED,
            linespacing=1.24,
            zorder=4,
        )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NEUTRAL,
    linewidth: float = 1.05,
    connectionstyle: str = "arc3",
    zorder: int = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9.5,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=connectionstyle,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
    )


def flow_label(
    ax,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    ha: str = "center",
    va: str = "center",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=4.85,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.13},
        zorder=5,
    )


def draw_fisheye_input(ax, x: float, y: float, size: float) -> None:
    """Draw an abstract wide-angle field with tags across radial regions."""
    center = (x + size * 0.50, y + size * 0.50)
    radius = size * 0.44
    ax.add_patch(Circle(center, radius, fill=False, edgecolor="#6D8FA4", linewidth=0.85, zorder=3))
    ax.add_patch(Circle(center, radius * 0.52, fill=False, edgecolor="#C5D8E3", linewidth=0.55, zorder=2))
    ax.add_patch(Circle(center, radius * 0.045, facecolor="#6D8FA4", edgecolor="none", zorder=3))

    # The tags intentionally appear at the center, mid-field, and periphery.
    tags = [
        (0.51, 0.49, 0.11, "#C65A25"),
        (0.25, 0.34, 0.11, "#1976A8"),
        (0.72, 0.28, 0.10, "#A77918"),
        (0.22, 0.67, 0.10, "#37895E"),
        (0.74, 0.70, 0.10, "#A55D7B"),
        (0.51, 0.81, 0.09, "#5D83B8"),
    ]
    for tx, ty, side, color in tags:
        left = x + size * (tx - side / 2)
        bottom = y + size * (ty - side / 2)
        length = size * side
        ax.add_patch(
            Rectangle((left, bottom), length, length, facecolor="white", edgecolor=color, linewidth=0.6, zorder=4)
        )
        cell = length * 0.20
        for px, py in ((0.15, 0.15), (0.55, 0.15), (0.35, 0.50), (0.70, 0.65)):
            ax.add_patch(
                Rectangle(
                    (left + length * px, bottom + length * py),
                    cell,
                    cell,
                    facecolor=color,
                    edgecolor="none",
                    zorder=5,
                )
            )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 4.08))
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.028, top=0.982)
    ax.set_xlim(0, 3.45)
    ax.set_ylim(0, 4.08)
    ax.axis("off")

    # Input is intentionally visual so that the reason for a distributed target
    # is visible before readers enter the processing stages.
    draw_fisheye_input(ax, 0.22, 3.38, 0.56)
    ax.text(0.94, 3.74, "Wide-angle frames", ha="left", va="center", fontsize=7.0, fontweight="semibold", color=INK)
    ax.text(0.94, 3.52, "with distributed AprilTag boards", ha="left", va="center", fontsize=5.3, color=MUTED)
    arrow(ax, (2.10, 3.42), (2.10, 3.22), color=NEUTRAL, linewidth=1.0)

    rounded_box(
        ax, 0.22, 2.75, 2.96, 0.47,
        title="Detect Boards and Extract Outer4",
        detail="board IDs and outer-corner observations",
        edge=NEUTRAL,
        fill=NEUTRAL_FILL,
        title_size=7.15,
    )

    # This split gives equal visual status to the two uses of Outer4: bootstrap
    # and retention in the final calibration set.
    rounded_box(
        ax, 0.22, 1.97, 1.34, 0.56,
        title="Outer4 Bootstrap",
        detail="camera model + temporary poses",
        edge=INIT,
        fill=INIT_FILL,
        title_size=6.55,
        detail_size=4.8,
    )
    rounded_box(
        ax, 1.89, 1.97, 1.29, 0.56,
        title="Outer4 Observations",
        detail="kept for final calibration",
        edge=OUTER,
        fill=OUTER_FILL,
        title_size=6.20,
        detail_size=4.8,
    )
    arrow(ax, (0.95, 2.75), (0.89, 2.53), color=INIT, linewidth=1.08)
    arrow(ax, (2.45, 2.75), (2.53, 2.53), color=OUTER, linewidth=1.08)

    rounded_box(
        ax, 0.22, 1.14, 1.34, 0.61,
        title="Recover Internal\nObservations",
        detail="spherical interpolation\n+ local image refinement",
        edge=INTERNAL,
        fill=INTERNAL_FILL,
        title_size=6.2,
        detail_size=4.55,
    )
    arrow(ax, (0.89, 1.97), (0.89, 1.75), color=INIT, linewidth=1.1)
    flow_label(ax, 0.89, 1.83, "intermediate model", INIT, va="bottom")

    # The two colored paths meet at a neutral observation-set module.
    rounded_box(
        ax, 0.22, 0.52, 2.96, 0.43,
        title="Combined Observation Set",
        detail=r"$\mathcal{O}=\mathcal{O}^{\mathrm{out}}\cup\mathcal{O}^{\mathrm{int}}$",
        edge=NEUTRAL,
        fill=NEUTRAL_FILL,
        title_size=6.95,
        detail_size=6.7,
    )
    arrow(ax, (0.89, 1.14), (1.12, 0.95), color=INTERNAL, linewidth=1.1)
    arrow(ax, (2.53, 1.97), (2.28, 0.95), color=OUTER, linewidth=1.1, connectionstyle="arc3,rad=-0.08")
    flow_label(ax, 0.30, 1.01, "internal observations", INTERNAL, ha="left", va="bottom")

    rounded_box(
        ax, 0.22, 0.05, 2.96, 0.34,
        title="Multi-board Bundle Adjustment",
        detail="camera intrinsics  ·  inter-board geometry  ·  frame poses",
        edge=FINAL,
        fill=FINAL_FILL,
        title_size=6.7,
        detail_size=4.7,
    )
    arrow(ax, (1.70, 0.52), (1.70, 0.39), color=FINAL, linewidth=1.08)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = OUTPUT_DIR / f"pipeline_overview_refined.{extension}"
        fig.savefig(output, dpi=420, bbox_inches="tight", pad_inches=0.008)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()

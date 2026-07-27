#!/usr/bin/env python3
"""Generate the single-column method flowchart used in Fig. 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "pic"

INK = "#252B30"
MUTED = "#5E6971"
FLOW = "#444D54"
OUTER = "#D26919"
INPUT_FILL, INPUT_EDGE = "#F1F4F8", "#728191"
OUTER_FILL, OUTER_EDGE = "#FCF0E5", "#C66A25"
INIT_FILL, INIT_EDGE = "#F8F1DF", "#B79336"
INTERNAL_FILL, INTERNAL_EDGE = "#EAF3F8", "#2C7EAC"
MERGE_FILL, MERGE_EDGE = "#F4F5F6", "#65737C"
FINAL_FILL, FINAL_EDGE = "#EAF4EC", "#4A9568"


def block(ax, x, y, width, height, fill, edge, title, subtitle=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            facecolor=fill, edgecolor=edge, linewidth=1.05,
        )
    )
    if subtitle:
        ax.text(x + width / 2, y + height * 0.63, title, ha="center", va="center",
                fontsize=7.45, fontweight="bold", color=INK)
        ax.text(x + width / 2, y + height * 0.31, subtitle, ha="center", va="center",
                fontsize=5.85, color=MUTED)
    else:
        ax.text(x + width / 2, y + height / 2, title, ha="center", va="center",
                fontsize=7.45, fontweight="bold", color=INK)


def arrow(ax, start, end, color=FLOW, linewidth=1.05, mutation=10, connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=mutation,
                        linewidth=linewidth, color=color, shrinkA=0, shrinkB=0,
                        connectionstyle=connectionstyle)
    )


def data_label(ax, x, y, text, color=FLOW, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=5.55, color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35})


def stage_marker(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=6.1, color=color,
            fontweight="bold", bbox={"boxstyle": "circle,pad=0.29", "facecolor": "white", "edgecolor": color, "linewidth": 0.85})


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.5, 5.55))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.985)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 5.55)
    ax.axis("off")

    x, width = 0.43, 2.35
    # Source input.
    block(ax, x, 4.94, width, 0.43, INPUT_FILL, INPUT_EDGE,
          "Input", "Wide-angle frames  +  distributed multi-board target")

    # Main processing modules.
    block(ax, x, 4.11, width, 0.54, OUTER_FILL, OUTER_EDGE,
          "Board Detection and Outer4 Extraction", "Tag association and outer-corner observations")
    block(ax, x, 3.28, width, 0.54, INIT_FILL, INIT_EDGE,
          "Outer4-only Bootstrap", "Intermediate model and frame--board poses")
    block(ax, x, 2.38, width, 0.62, INTERNAL_FILL, INTERNAL_EDGE,
          "Model-guided Internal Observation Recovery",
          "Unprojection  $\\rightarrow$  spherical interpolation  $\\rightarrow$  local refinement")
    block(ax, x, 1.36, width, 0.58, MERGE_FILL, MERGE_EDGE,
          "Observation Set", r"$\mathcal{O}=\mathcal{O}^{\mathrm{out}} \cup \mathcal{O}^{\mathrm{int}}$")
    block(ax, x, 0.53, width, 0.54, FINAL_FILL, FINAL_EDGE,
          "Selection-based Two-pass Bundle Adjustment",
          "Camera intrinsics, board geometry, and frame poses")

    # Central data flow.
    arrow(ax, (1.605, 4.94), (1.605, 4.65))
    arrow(ax, (1.605, 4.11), (1.605, 3.82))
    arrow(ax, (1.605, 3.28), (1.605, 3.00))
    arrow(ax, (1.605, 2.38), (1.605, 1.94))
    arrow(ax, (1.605, 1.36), (1.605, 1.07))
    data_label(ax, 1.605, 3.10, "intermediate camera model", color=INIT_EDGE, va="bottom")
    data_label(ax, 1.605, 2.02, "refined internal observations", color=INTERNAL_EDGE, va="bottom")

    # The same stable outer observations are retained for the final solve.
    bypass_x = 3.13
    ax.plot([x + width, bypass_x], [4.38, 4.38], color=OUTER, linewidth=1.0)
    ax.plot([bypass_x, bypass_x], [4.38, 1.65], color=OUTER, linewidth=1.0)
    arrow(ax, (bypass_x, 1.65), (x + width, 1.65), color=OUTER, linewidth=1.0, mutation=9)
    ax.text(bypass_x + 0.08, 2.98, "Outer4 observations", ha="center", va="center",
            rotation=90, fontsize=5.55, color=OUTER,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25})

    # Stage markers keep the vertical reading order visible without adding a legend.
    stage_marker(ax, 0.20, 4.38, "1", OUTER_EDGE)
    stage_marker(ax, 0.20, 3.55, "2", INIT_EDGE)
    stage_marker(ax, 0.20, 2.69, "3", INTERNAL_EDGE)
    stage_marker(ax, 0.20, 0.80, "4", FINAL_EDGE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUTPUT_DIR / f"pipeline_flowchart.{extension}"
        fig.savefig(path, dpi=360, bbox_inches="tight", pad_inches=0.012)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

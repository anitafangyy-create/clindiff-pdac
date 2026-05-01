#!/usr/bin/env python3
"""
Generate a Nature-style overview figure for the experimental framework.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "experimental_framework_overview.png"


def add_box(ax, x, y, w, h, text, fc, fontsize=10, ec="#344054", lw=1.4):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        wrap=True,
    )


def add_arrow(ax, p1, p2, color="#667085", lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            shrinkA=6,
            shrinkB=6,
        )
    )


def main():
    os.makedirs(OUTPUT.parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Overview of the experimental framework",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    ax.text(0.03, 0.91, "1 | Cohort and features", fontsize=13, fontweight="bold")
    add_box(
        ax,
        0.04,
        0.73,
        0.20,
        0.14,
        "Internal PDAC cohort\n2,347 records\n6 structured features",
        "#dbeafe",
    )
    add_box(
        ax,
        0.28,
        0.73,
        0.20,
        0.14,
        "Liver-function trio\nTB / DB / GGT\nplus age, survival,\nfasting glucose",
        "#e0f2fe",
    )
    add_arrow(ax, (0.24, 0.80), (0.28, 0.80))

    ax.text(0.53, 0.91, "2 | Missingness-aware design", fontsize=13, fontweight="bold")
    add_box(
        ax,
        0.54,
        0.73,
        0.18,
        0.14,
        "Four-state mask\nObserved / True Missing /\nNot Applicable / OTW",
        "#ede9fe",
    )
    add_box(
        ax,
        0.76,
        0.73,
        0.18,
        0.14,
        "Mechanism profiling\nMCAR / MAR / MNAR /\nStructural",
        "#fae8ff",
    )
    add_arrow(ax, (0.48, 0.80), (0.54, 0.80))
    add_arrow(ax, (0.72, 0.80), (0.76, 0.80))

    ax.text(0.03, 0.62, "3 | Experimental comparisons", fontsize=13, fontweight="bold")
    add_box(
        ax,
        0.05,
        0.41,
        0.16,
        0.16,
        "Artificial missingness\n20% / 40% / 60%\nrepeated masking",
        "#fef3c7",
    )
    add_box(
        ax,
        0.26,
        0.41,
        0.16,
        0.16,
        "Baselines\nMean / Median /\nKNN / MICE /\nMissForest",
        "#fff7d6",
    )
    add_box(
        ax,
        0.47,
        0.41,
        0.16,
        0.16,
        "ClinDiff-Lite\nadaptive per-feature\nstrategy selection",
        "#dcfce7",
    )
    add_box(
        ax,
        0.68,
        0.41,
        0.16,
        0.16,
        "Optional advanced branch\nknowledge-guided diffusion\n+ clinical constraints",
        "#d1fae5",
    )
    add_arrow(ax, (0.21, 0.49), (0.26, 0.49))
    add_arrow(ax, (0.42, 0.49), (0.47, 0.49))
    add_arrow(ax, (0.63, 0.49), (0.68, 0.49))

    ax.text(0.03, 0.31, "4 | Evaluation and translation", fontsize=13, fontweight="bold")
    add_box(
        ax,
        0.08,
        0.11,
        0.20,
        0.14,
        "Primary metrics\noverall RMSE\nper-feature RMSE",
        "#fee2e2",
    )
    add_box(
        ax,
        0.40,
        0.11,
        0.20,
        0.14,
        "Robustness checks\nMCAR comparison\nfeature-level analysis",
        "#fce7f3",
    )
    add_box(
        ax,
        0.72,
        0.11,
        0.20,
        0.14,
        "Clinical relevance\nSEER comparison\nsurvival-model utility\nliterature positioning",
        "#e5e7eb",
    )
    add_arrow(ax, (0.55, 0.41), (0.18, 0.25))
    add_arrow(ax, (0.55, 0.41), (0.50, 0.25))
    add_arrow(ax, (0.55, 0.41), (0.82, 0.25))

    fig.text(
        0.5,
        0.03,
        "OTW = out of time window. The overview highlights how cohort definition, missingness logic,\n"
        "comparative imputation, and clinically oriented evaluation are connected in the manuscript.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#475467",
    )

    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight")
    print(str(OUTPUT))


if __name__ == "__main__":
    main()

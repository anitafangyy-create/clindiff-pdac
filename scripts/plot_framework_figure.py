#!/usr/bin/env python3
"""
Generate an updated ClinDiff-PDAC framework architecture figure that matches
the current codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "framework_figure_current.png"


def add_box(ax, x, y, w, h, text, fc, ec="#1f2937", fontsize=10, radius=0.03, lw=1.6):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
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


def add_arrow(ax, start, end, color="#475569", lw=1.8, style="-|>"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(arrow)


def panel_a(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.02, "a | Current codebase pipeline", fontsize=14, fontweight="bold")

    add_box(ax, 0.03, 0.74, 0.20, 0.16, "Raw PDAC EMR table\nvalues + dates + rules", "#e0f2fe", fontsize=9)
    add_box(ax, 0.29, 0.74, 0.22, 0.16, "RuleEngine\nfour-state mask\nObserved / True Missing /\nNot Applicable / OTW", "#dbeafe", fontsize=8.6)
    add_box(ax, 0.57, 0.74, 0.20, 0.16, "MissingnessAnalyzer\nMCAR / MAR / MNAR /\nStructural profiles", "#ede9fe", fontsize=9)
    add_box(ax, 0.81, 0.74, 0.16, 0.16, "Routing\nby data type + goal", "#fae8ff", fontsize=9)

    add_arrow(ax, (0.23, 0.82), (0.29, 0.82))
    add_arrow(ax, (0.51, 0.82), (0.57, 0.82))
    add_arrow(ax, (0.77, 0.82), (0.81, 0.82))

    add_box(ax, 0.06, 0.43, 0.25, 0.18, "Enhanced baselines\nMissForest / KNN / MICE", "#fef3c7")
    add_box(ax, 0.38, 0.40, 0.25, 0.24, "ClinDiffPDAC deep imputer\nnormalize -> KnowledgeGuidedDiffusion\nConditionalDenoiser +\nKnowledgeConstraintNetwork", "#dcfce7")
    add_box(ax, 0.71, 0.43, 0.23, 0.18, "LLMConstraintLayer\nConstraint + ClinicalContext ->\nvalue + confidence + evidence", "#fee2e2")

    add_arrow(ax, (0.89, 0.74), (0.18, 0.61))
    add_arrow(ax, (0.89, 0.74), (0.50, 0.64))
    add_arrow(ax, (0.89, 0.74), (0.82, 0.61))

    add_box(ax, 0.26, 0.10, 0.48, 0.18, "Evaluation + validation\nImputationEvaluator (RMSE / MAE / per-feature metrics)\nClinicalValidator (ranges, logical checks, temporal checks)", "#f3f4f6")
    add_arrow(ax, (0.18, 0.43), (0.37, 0.28))
    add_arrow(ax, (0.50, 0.40), (0.50, 0.28))
    add_arrow(ax, (0.82, 0.43), (0.63, 0.28))


def panel_b(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.02, "b | Knowledge-guided diffusion path", fontsize=14, fontweight="bold")

    add_box(ax, 0.03, 0.72, 0.22, 0.18, "Normalized feature matrix\n+ observed mask", "#e0f2fe")
    add_box(ax, 0.31, 0.72, 0.18, 0.18, "Random / encoded\nKG context", "#ede9fe")
    add_box(ax, 0.55, 0.72, 0.18, 0.18, "Forward diffusion\nq(x_t | x_0)", "#fef3c7")
    add_box(ax, 0.77, 0.72, 0.20, 0.18, "ConditionalDenoiser\nnoise prediction", "#dcfce7")

    add_arrow(ax, (0.25, 0.81), (0.55, 0.81))
    add_arrow(ax, (0.49, 0.81), (0.77, 0.81))

    add_box(ax, 0.16, 0.36, 0.24, 0.18, "predict x_0 from x_t\nand predicted noise", "#f3f4f6")
    add_box(ax, 0.46, 0.34, 0.24, 0.22, "KnowledgeConstraintNetwork\nsoft correction weights\n+ correction directions", "#fee2e2")
    add_box(ax, 0.76, 0.36, 0.18, 0.18, "Observed values\nreinserted", "#dbeafe")

    add_arrow(ax, (0.87, 0.72), (0.28, 0.54))
    add_arrow(ax, (0.40, 0.45), (0.46, 0.45))
    add_arrow(ax, (0.70, 0.45), (0.76, 0.45))

    add_box(ax, 0.28, 0.07, 0.44, 0.16, "Iterative reverse diffusion / fast denoising loop\nreturns imputed matrix in original feature space", "#ecfccb")
    add_arrow(ax, (0.85, 0.36), (0.50, 0.23))


def panel_c(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.02, "c | Structured clinical constraint branch", fontsize=14, fontweight="bold")

    add_box(ax, 0.05, 0.66, 0.23, 0.18, "Patient row / partial record", "#e0f2fe")
    add_box(ax, 0.36, 0.66, 0.23, 0.18, "Constraint objects\ncategorical / range /\nregex / custom", "#fef3c7")
    add_box(ax, 0.67, 0.66, 0.23, 0.18, "ClinicalContext\ndiagnosis, stage, labs,\ntreatments, timing", "#ede9fe")

    add_box(ax, 0.30, 0.35, 0.36, 0.18, "LLMConstraintLayer\nprompt build -> placeholder / LLM call ->\nparse + validate", "#dcfce7")
    add_arrow(ax, (0.17, 0.66), (0.42, 0.53))
    add_arrow(ax, (0.47, 0.66), (0.48, 0.53))
    add_arrow(ax, (0.78, 0.66), (0.54, 0.53))

    add_box(ax, 0.20, 0.08, 0.56, 0.16, "Structured output\nimputed value + confidence score + evidence +\nconstraint satisfaction + alternative values", "#fee2e2")
    add_arrow(ax, (0.48, 0.35), (0.48, 0.24))


def main():
    os.makedirs(OUTPUT.parent, exist_ok=True)

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    gs = fig.add_gridspec(3, 1, hspace=0.30)

    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[1, 0]))
    panel_c(fig.add_subplot(gs[2, 0]))

    fig.suptitle(
        "ClinDiff-PDAC framework architecture aligned to the current codebase",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "OTW = out of time window. The current repository exposes rule-based masking, mechanism analysis,\n"
        "baseline imputers, a knowledge-guided diffusion imputer, and a structured LLM constraint layer.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#374151",
    )

    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    print(str(OUTPUT))


if __name__ == "__main__":
    main()

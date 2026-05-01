#!/usr/bin/env python3
"""
Regenerate supplementary figures so their in-image wording matches the manuscript.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT_DIR, "figures")


def build_contextualization_figure():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=180)

    age_labels = ["Internal EMR", "SEER"]
    age_values = [63.3, 68.1]
    age_colors = ["#355070", "#6d597a"]
    axes[0].bar(age_labels, age_values, color=age_colors, width=0.55)
    axes[0].set_title("External Contextualization: Age Structure", fontsize=13, weight="bold")
    axes[0].set_ylabel("Mean age (years)")
    axes[0].set_ylim(0, 80)
    for idx, value in enumerate(age_values):
        axes[0].text(idx, value + 1, f"{value:.1f}", ha="center", fontsize=10)

    panels = ["No TB", "No DB", "No GGT", "Registry stage", "Population survival"]
    internal = [0, 0, 0, 0, 0]
    seer = [0, 0, 0, 1, 1]
    x = np.arange(len(panels))
    width = 0.34
    axes[1].bar(x - width / 2, internal, width=width, color="#355070", label="Internal EMR")
    axes[1].bar(x + width / 2, seer, width=width, color="#6d597a", label="SEER")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(panels, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.2)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Absent", "Present"])
    axes[1].set_title("Shared-feature Landscape", fontsize=13, weight="bold")
    axes[1].legend(frameon=False, loc="upper left")

    fig.suptitle(
        "Supplementary Fig. S1 | SEER registry contextualization only; not external imputation validation",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def build_downstream_figure():
    methods = [
        "Complete-case",
        "Mean",
        "Median",
        "KNN",
        "MICE",
        "ClinDiff-Gated",
    ]
    cindex = [0.612, 0.598, 0.595, 0.605, 0.603, 0.631]
    err = [0.031, 0.038, 0.039, 0.033, 0.035, 0.028]
    colors = ["#a0aec0", "#355070", "#6d597a", "#b56576", "#8d99ae", "#2a9d8f"]

    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=180)
    x = np.arange(len(methods))
    ax.bar(x, cindex, yerr=err, color=colors, width=0.68, capsize=4, edgecolor="none")
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel("Harrell's C-index")
    ax.set_ylim(0.5, 0.70)
    ax.set_title("Exploratory downstream survival modelling", fontsize=14, weight="bold")
    for idx, value in enumerate(cindex):
        ax.text(idx, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)

    fig.text(
        0.02,
        0.02,
        "Updated internal cohort: n=2,347 records; complete-case reference: n=1,772.\n"
        "Exploratory only; not part of the prespecified headline endpoint.",
        fontsize=9,
        color="#4a5568",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    return fig


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    fig1 = build_contextualization_figure()
    fig1.savefig(os.path.join(FIG_DIR, "fig3_seer_validation.png"), bbox_inches="tight")
    plt.close(fig1)

    fig2 = build_downstream_figure()
    fig2.savefig(os.path.join(FIG_DIR, "fig4_downstream.png"), bbox_inches="tight")
    plt.close(fig2)

    print(os.path.join(FIG_DIR, "fig3_seer_validation.png"))
    print(os.path.join(FIG_DIR, "fig4_downstream.png"))


if __name__ == "__main__":
    main()

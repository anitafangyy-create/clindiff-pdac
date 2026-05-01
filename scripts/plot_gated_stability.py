#!/usr/bin/env python3
"""
Create seed-stability and paired-delta distribution figures for the gated experiment.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")


def load_inputs():
    runs_path = os.path.join(OUTPUT_DIR, "gated_experiment_repeated_runs.csv")
    summary_path = os.path.join(OUTPUT_DIR, "gated_experiment_repeated_summary.csv")
    if not os.path.exists(runs_path) or not os.path.exists(summary_path):
        raise FileNotFoundError("Run minimal_experiment.py before plotting stability figures.")
    return pd.read_csv(runs_path), pd.read_csv(summary_path)


def build_seed_stability(runs_df: pd.DataFrame) -> plt.Figure:
    for style_name in ["seaborn-whitegrid", "ggplot", "default"]:
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=180, sharex=True)
    palette = {
        "20%": "#1b9e77",
        "40%": "#d95f02",
        "60%": "#7570b3",
    }

    for ax, rate in zip(axes, ["20%", "40%", "60%"]):
        subset = runs_df[runs_df["MissingRate"] == rate].copy().sort_values("Repeat")
        subset["Overall Delta"] = subset["Mean"] - subset["ClinDiff-Gated"]
        subset["Liver Delta"] = subset["Mean Liver Avg"] - subset["ClinDiff-Gated Liver Avg"]

        ax.plot(
            subset["Repeat"],
            subset["Overall Delta"],
            color=palette[rate],
            linewidth=1.8,
            marker="o",
            markersize=3,
            alpha=0.85,
            label="Overall paired delta",
        )
        ax.plot(
            subset["Repeat"],
            subset["Liver Delta"],
            color="#111111",
            linewidth=1.2,
            alpha=0.45,
            label="Liver-trio paired delta",
        )
        ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
        ax.set_ylabel(f"{rate}\nDelta RMSE")
        ax.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("Repeated masking seed index")
    fig.suptitle(
        "ClinDiff-Gated Seed Stability\nPaired delta RMSE across 100 repeated masks",
        fontsize=16,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def build_delta_distribution(runs_df: pd.DataFrame, summary_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    palette = ["#1b9e77", "#d95f02", "#7570b3"]
    rate_order = ["20%", "40%", "60%"]

    overall = []
    liver = []
    for rate in rate_order:
        subset = runs_df[runs_df["MissingRate"] == rate]
        overall.append((subset["Mean"] - subset["ClinDiff-Gated"]).values)
        liver.append((subset["Mean Liver Avg"] - subset["ClinDiff-Gated Liver Avg"]).values)

    violin1 = axes[0].violinplot(overall, showmeans=True, showextrema=True)
    violin2 = axes[1].violinplot(liver, showmeans=True, showextrema=True)
    for violin, color in zip(violin1["bodies"], palette):
        violin.set_facecolor(color)
        violin.set_alpha(0.65)
    for violin, color in zip(violin2["bodies"], palette):
        violin.set_facecolor(color)
        violin.set_alpha(0.65)

    for ax, title in zip(
        axes,
        ["Overall paired delta: Mean - ClinDiff-Gated", "Liver-trio paired delta: Mean - ClinDiff-Gated"],
    ):
        ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
        ax.set_xticks(np.arange(1, 4))
        ax.set_xticklabels(rate_order)
        ax.set_xlabel("Missing rate")
        ax.set_ylabel("Paired delta RMSE")
        ax.set_title(title, fontsize=12, weight="bold")

    annotations = summary_df.set_index("MissingRate")
    for idx, rate in enumerate(rate_order, start=1):
        axes[0].text(
            idx,
            max(overall[idx - 1]) * 0.92,
            f"mean={annotations.loc[rate, 'Delta MeanMinusClinDiff-Gated Mean']:.2f}\n"
            f"CI [{annotations.loc[rate, 'Delta MeanMinusClinDiff-Gated Bootstrap CI Low']:.2f}, "
            f"{annotations.loc[rate, 'Delta MeanMinusClinDiff-Gated Bootstrap CI High']:.2f}]",
            ha="center",
            va="top",
            fontsize=9,
        )
        axes[1].text(
            idx,
            max(liver[idx - 1]) * 0.92,
            f"mean={annotations.loc[rate, 'Delta MeanLiverAvgMinusClinDiff-GatedLiverAvg Mean']:.2f}\n"
            f"CI [{annotations.loc[rate, 'Delta MeanLiverAvgMinusClinDiff-GatedLiverAvg Bootstrap CI Low']:.2f}, "
            f"{annotations.loc[rate, 'Delta MeanLiverAvgMinusClinDiff-GatedLiverAvg Bootstrap CI High']:.2f}]",
            ha="center",
            va="top",
            fontsize=9,
        )

    fig.suptitle(
        "Distribution of paired delta RMSE across repeated masks",
        fontsize=15,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def main():
    runs_df, summary_df = load_inputs()
    seed_fig = build_seed_stability(runs_df)
    seed_path = os.path.join(OUTPUT_DIR, "gated_seed_stability.png")
    seed_fig.savefig(seed_path, bbox_inches="tight")
    plt.close(seed_fig)

    delta_fig = build_delta_distribution(runs_df, summary_df)
    delta_path = os.path.join(OUTPUT_DIR, "gated_paired_delta_distribution.png")
    delta_fig.savefig(delta_path, bbox_inches="tight")
    plt.close(delta_fig)
    print(seed_path)
    print(delta_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to create stability figures: {exc}", file=sys.stderr)
        sys.exit(1)

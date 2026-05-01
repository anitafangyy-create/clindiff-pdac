#!/usr/bin/env python3
"""
Create publication-friendly plots for the minimal experiment outputs.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")


def load_results():
    summary_path = os.path.join(OUTPUT_DIR, "minimal_experiment_summary.csv")
    feature_path = os.path.join(OUTPUT_DIR, "minimal_experiment_per_feature.csv")

    if not os.path.exists(summary_path) or not os.path.exists(feature_path):
        raise FileNotFoundError(
            "Missing minimal experiment CSV outputs. Run minimal_experiment.py first."
        )

    return pd.read_csv(summary_path), pd.read_csv(feature_path)


def build_figure(summary_df, feature_df):
    for style_name in ["seaborn-whitegrid", "ggplot", "default"]:
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue
    fig = plt.figure(figsize=(14, 8), dpi=160)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1, 1])

    ax_overall = fig.add_subplot(gs[:, 0])
    ax_20 = fig.add_subplot(gs[0, 1])
    ax_40 = fig.add_subplot(gs[1, 1])

    colors = {
        "Mean": "#355070",
        "Median": "#6d597a",
        "ClinDiff-Gated": "#2a9d8f",
    }

    rates = summary_df["MissingRate"].tolist()
    x = np.arange(len(rates))

    for method in ["Mean", "Median", "ClinDiff-Gated"]:
        ax_overall.plot(
            x,
            summary_df[method].values,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=colors[method],
            label=method,
        )

    ax_overall.set_title("Secondary Non-outcome RMSE", fontsize=13, weight="bold", pad=12)
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(rates)
    ax_overall.set_xlabel("Artificial Missing Rate")
    ax_overall.set_ylabel("RMSE")
    ax_overall.legend(frameon=True)

    for idx, rate in enumerate(rates):
        value = summary_df.loc[idx, "ClinDiff-Gated"]
        ax_overall.annotate(
            f"{value:.2f}",
            (idx, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=colors["ClinDiff-Gated"],
        )

    def plot_feature_gain(ax, rate):
        current = feature_df[feature_df["MissingRate"] == rate].copy()
        current = current[current["Feature"] != "survival_days"].copy()
        current["GainVsMean"] = current["Mean"] - current["ClinDiff-Gated"]
        current = current.sort_values("GainVsMean", ascending=True)

        bar_colors = [
            "#2a9d8f" if gain > 0 else "#bdbdbd"
            for gain in current["GainVsMean"].values
        ]

        ax.barh(current["Feature"], current["GainVsMean"], color=bar_colors)
        ax.axvline(0, color="#444444", linewidth=1)
        ax.set_title(f"Per-Feature RMSE Gain vs Mean ({rate})", fontsize=13, weight="bold")
        ax.set_xlabel("Positive values mean lower RMSE for ClinDiff-Gated")

        for y, gain in enumerate(current["GainVsMean"].values):
            x_pos = gain + (1.2 if gain >= 0 else -1.2)
            ha = "left" if gain >= 0 else "right"
            ax.text(x_pos, y, f"{gain:.2f}", va="center", ha=ha, fontsize=8)

    plot_feature_gain(ax_20, "20%")
    plot_feature_gain(ax_40, "40%")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def main():
    summary_df, feature_df = load_results()
    fig = build_figure(summary_df, feature_df)

    output_path = os.path.join(OUTPUT_DIR, "minimal_experiment_figure.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to create plot: {exc}", file=sys.stderr)
        sys.exit(1)

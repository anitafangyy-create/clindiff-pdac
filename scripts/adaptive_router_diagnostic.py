#!/usr/bin/env python3
"""
Generate a lightweight diagnostic report for the adaptive liver-trio router.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from minimal_experiment import clindiff_gated, introduce_missing, load_data  # noqa: E402


def main() -> None:
    df, _ = load_data()
    records = []
    reason_records = []
    for rate in [0.2, 0.4, 0.6]:
        df_miss, _ = introduce_missing(df, rate=rate, seed=42)
        _, _, routing_df = clindiff_gated(df_miss, seed=42)

        reason_counts = routing_df["gate_reason"].value_counts(normalize=True)
        records.append({
            "MissingRate": f"{rate:.0%}",
            "Grouped Usage Rate": float(routing_df["used_grouped_refinement"].mean()),
            "Mean Signal Strength Score": float(routing_df["signal_strength_score"].mean()),
            "Median Transformed Distance": float(routing_df["transformed_distance"].median()),
            "Mean Usable Predictor Count": float(routing_df["usable_predictor_count"].mean()),
            "Mechanism Label": routing_df["mechanism_label"].mode().iloc[0],
            "Mechanism Score": float(routing_df["mechanism_score"].mean()),
            "Top Gate Reason": routing_df["gate_reason"].mode().iloc[0],
        })
        for reason, share in reason_counts.items():
            reason_records.append({
                "MissingRate": f"{rate:.0%}",
                "GateReason": reason,
                "Share": float(share),
            })

    output_dir = os.path.join(ROOT, "outputs")
    summary_path = os.path.join(output_dir, "adaptive_router_diagnostic.csv")
    reason_path = os.path.join(output_dir, "adaptive_router_reason_breakdown.csv")
    pd.DataFrame(records).to_csv(summary_path, index=False)
    pd.DataFrame(reason_records).to_csv(reason_path, index=False)
    print(summary_path)
    print(reason_path)


if __name__ == "__main__":
    main()

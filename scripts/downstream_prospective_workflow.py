#!/usr/bin/env python3
"""
Build a leakage-free downstream workflow specification for future prospective analyses.
"""

from __future__ import annotations

import json
import os

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "raw", "pancreatic_cancer_data_normalized_clean.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "downstream_leakage_free")

OUTCOME_COLUMNS = {
    "survival_days",
    "survival_months",
    "os_months",
    "dfs_months",
    "status",
    "death_event",
}


def summarize_feature_roles(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in df.columns:
        role = "eligible_predictor"
        rationale = "Pre-treatment structured predictor eligible for leakage-free imputation and downstream modeling."
        if col in OUTCOME_COLUMNS:
            role = "excluded_outcome"
            rationale = "Outcome-derived variable; must be excluded from all upstream imputation predictors."
        records.append({
            "feature": col,
            "role": role,
            "missing_rate": float(df[col].isna().mean()),
            "dtype": str(df[col].dtype),
            "rationale": rationale,
        })
    return pd.DataFrame(records)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    role_df = summarize_feature_roles(df)
    csv_path = os.path.join(OUT_DIR, "prospective_feature_roles.csv")
    role_df.to_csv(csv_path, index=False)

    payload = {
        "data_path": DATA_PATH,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "eligible_predictors": role_df[role_df["role"] == "eligible_predictor"]["feature"].tolist(),
        "excluded_outcomes": role_df[role_df["role"] == "excluded_outcome"]["feature"].tolist(),
        "workflow_steps": [
            "Freeze a train/test split before any imputation tuning.",
            "Remove all outcome-derived columns from the imputation predictor set.",
            "Fit imputation strategy selection on training data only.",
            "Replay the fitted strategy on held-out data without retuning.",
            "Compare complete-case, mean, ClinDiff-Lite, and ClinDiff-Gated under the same split.",
            "Evaluate prespecified downstream metrics only after imputation is frozen.",
        ],
    }
    json_path = os.path.join(OUT_DIR, "prospective_workflow.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    md_lines = [
        "# Leakage-free prospective downstream workflow",
        "",
        f"- Source data: `{DATA_PATH}`",
        f"- Rows: {len(df)}",
        f"- Columns: {df.shape[1]}",
        "",
        "## Prespecified workflow",
        "",
        "1. Freeze a train/test split or temporal split before any imputation tuning.",
        "2. Remove all outcome-derived variables from upstream imputation predictors.",
        "3. Fit route selection and grouped refinement thresholds only on the training split.",
        "4. Apply the frozen imputation workflow to held-out patients without retuning.",
        "5. Compare complete-case, mean, ClinDiff-Lite, and ClinDiff-Gated under identical splits.",
        "6. Evaluate downstream endpoints only after the imputation model is frozen.",
        "",
        "## Current leakage audit",
        "",
        f"- Eligible predictors in the current internal dataset: {int((role_df['role'] == 'eligible_predictor').sum())}",
        f"- Outcome-derived columns excluded upstream: {', '.join(payload['excluded_outcomes']) or 'None detected beyond survival fields'}",
        "",
        "## Interpretation",
        "",
        "This file defines a prospective-ready downstream protocol rather than a new headline result.",
        "Its purpose is to prevent survival-derived variables from leaking back into biomarker imputation.",
    ]
    md_path = os.path.join(OUT_DIR, "prospective_workflow.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")

    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()

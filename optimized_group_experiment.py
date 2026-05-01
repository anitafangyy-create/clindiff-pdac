#!/usr/bin/env python3
"""
Grouped optimization experiment for ClinDiff-Lite.

Focus:
- preserve the lightweight setup from minimal_experiment.py
- add physiology-aware liver-trio imputation in log space
- quantify gains on the headline biomarker group under repeated masking
"""

import os
import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from minimal_experiment import (  # noqa: E402
    clindiff_lite,
    introduce_missing,
    load_data,
    mean_impute,
    per_feature_rmse,
    rmse,
)

LIVER_GROUP = [
    "tb_before_treatment",
    "ggt_before_treatment",
    "db_before_treatment",
]

NON_NEGATIVE = {
    "age",
    "survival_days",
    "fasting_glucose_mmol_L",
    "tb_before_treatment",
    "ggt_before_treatment",
    "db_before_treatment",
}


def ensure_output_dir() -> str:
    output_dir = os.path.join(ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _ridge_predict(
    train_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    target: str,
    predictors: List[str],
    reg: float = 0.5,
) -> pd.Series:
    if not predictors:
        return pd.Series(np.nan, index=train_df.index)

    observed_rows = ~raw_df[target].isna()
    X_train = train_df.loc[observed_rows, predictors].values.astype(float)
    y_train = raw_df.loc[observed_rows, target].values.astype(float)

    X_train = np.log1p(np.clip(X_train, 0, None))
    y_train = np.log1p(np.clip(y_train, 0, None))

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std < 1e-8] = 1.0

    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-8:
        y_std = 1.0

    X_scaled = (X_train - x_mean) / x_std
    y_scaled = (y_train - y_mean) / y_std

    beta = np.linalg.solve(
        X_scaled.T @ X_scaled + reg * np.eye(X_scaled.shape[1]),
        X_scaled.T @ y_scaled,
    )

    X_all = train_df[predictors].values.astype(float)
    X_all = np.log1p(np.clip(X_all, 0, None))
    predictions = ((X_all - x_mean) / x_std) @ beta
    predictions = predictions * y_std + y_mean
    predictions = np.expm1(predictions)

    target_observed = raw_df[target].dropna()
    lower = max(0.0, float(target_observed.quantile(0.01)))
    upper = float(target_observed.quantile(0.99))
    predictions = np.clip(predictions, lower, upper)
    return pd.Series(predictions, index=train_df.index)


def clindiff_lite_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean for weakly related non-headline variables + grouped liver-trio refinement.
    """
    out = df.copy().astype(float)
    medians = df.median()
    base = df.fillna(medians).copy()

    # Step 1: TB from DB + GGT + weak clinical anchors
    tb_predictors = [
        col for col in ["db_before_treatment", "ggt_before_treatment", "age", "fasting_glucose_mmol_L"]
        if col in df.columns and col != "tb_before_treatment"
    ]
    tb_pred = _ridge_predict(base, df, "tb_before_treatment", tb_predictors)
    tb_missing = out["tb_before_treatment"].isna()
    out.loc[tb_missing, "tb_before_treatment"] = tb_pred.loc[tb_missing]
    base["tb_before_treatment"] = out["tb_before_treatment"].fillna(medians["tb_before_treatment"])

    # Step 2: DB from TB + GGT, blended with observed DB/TB ratio prior
    db_predictors = [
        col for col in ["tb_before_treatment", "ggt_before_treatment", "age"]
        if col in df.columns and col != "db_before_treatment"
    ]
    db_pred = _ridge_predict(base, df, "db_before_treatment", db_predictors)
    ratio_rows = df[["tb_before_treatment", "db_before_treatment"]].dropna()
    ratio_rows = ratio_rows[ratio_rows["tb_before_treatment"] > 0]
    ratio_prior = float((ratio_rows["db_before_treatment"] / ratio_rows["tb_before_treatment"]).median()) if len(ratio_rows) else 0.3
    db_ratio_pred = ratio_prior * base["tb_before_treatment"]
    db_blend = 0.7 * db_pred + 0.3 * db_ratio_pred
    db_missing = out["db_before_treatment"].isna()
    out.loc[db_missing, "db_before_treatment"] = db_blend.loc[db_missing]
    base["db_before_treatment"] = out["db_before_treatment"].fillna(medians["db_before_treatment"])

    # Step 3: GGT from TB + DB + glucose
    ggt_predictors = [
        col for col in ["tb_before_treatment", "db_before_treatment", "fasting_glucose_mmol_L", "age"]
        if col in df.columns and col != "ggt_before_treatment"
    ]
    ggt_pred = _ridge_predict(base, df, "ggt_before_treatment", ggt_predictors)
    ggt_missing = out["ggt_before_treatment"].isna()
    out.loc[ggt_missing, "ggt_before_treatment"] = ggt_pred.loc[ggt_missing]

    # Non-headline variables keep a simple robust baseline.
    for col in out.columns:
        if col not in LIVER_GROUP:
            out[col] = out[col].fillna(df[col].mean())

    # Physiologic consistency constraints.
    for col in NON_NEGATIVE.intersection(out.columns):
        out[col] = np.clip(out[col], 0, None)
    if {"tb_before_treatment", "db_before_treatment"}.issubset(out.columns):
        out["db_before_treatment"] = np.minimum(out["db_before_treatment"], out["tb_before_treatment"])

    return out


def average_feature_rmse(metrics: Dict[str, float], features: List[str]) -> float:
    values = [metrics[f] for f in features if f in metrics and not np.isnan(metrics[f])]
    return float(np.mean(values)) if values else float("nan")


def run_grouped_optimization(rates=None, repeats: int = 20, base_seed: int = 4040) -> None:
    if rates is None:
        rates = [0.2, 0.4, 0.6]

    df, cols = load_data()
    records = []
    feature_records = []

    for rate in rates:
        for repeat in range(repeats):
            seed = base_seed + repeat + int(rate * 1000)
            df_miss, mask = introduce_missing(df, rate, seed=seed)

            df_mean = mean_impute(df_miss)
            df_clindiff, _ = clindiff_lite(df_miss, seed=seed)
            df_grouped = clindiff_lite_grouped(df_miss)

            mean_feature = per_feature_rmse(df, df_mean, mask)
            clindiff_feature = per_feature_rmse(df, df_clindiff, mask)
            grouped_feature = per_feature_rmse(df, df_grouped, mask)

            records.append({
                "MissingRate": f"{rate:.0%}",
                "Repeat": repeat + 1,
                "Seed": seed,
                "Mean Overall": rmse(df, df_mean, mask),
                "ClinDiff-Lite Overall": rmse(df, df_clindiff, mask),
                "Grouped Overall": rmse(df, df_grouped, mask),
                "Mean Liver Avg": average_feature_rmse(mean_feature, LIVER_GROUP),
                "ClinDiff-Lite Liver Avg": average_feature_rmse(clindiff_feature, LIVER_GROUP),
                "Grouped Liver Avg": average_feature_rmse(grouped_feature, LIVER_GROUP),
            })

            for feature in cols:
                feature_records.append({
                    "MissingRate": f"{rate:.0%}",
                    "Repeat": repeat + 1,
                    "Feature": feature,
                    "Mean": mean_feature[feature],
                    "ClinDiff-Lite": clindiff_feature[feature],
                    "Grouped": grouped_feature[feature],
                })

    run_df = pd.DataFrame(records)
    feature_df = pd.DataFrame(feature_records)

    summary_rows = []
    for rate in sorted(run_df["MissingRate"].unique(), key=lambda x: float(x.strip("%"))):
        subset = run_df[run_df["MissingRate"] == rate]
        summary_rows.append({
            "MissingRate": rate,
            "Repeats": int(len(subset)),
            "Mean Overall": float(subset["Mean Overall"].mean()),
            "ClinDiff-Lite Overall": float(subset["ClinDiff-Lite Overall"].mean()),
            "Grouped Overall": float(subset["Grouped Overall"].mean()),
            "Mean Liver Avg": float(subset["Mean Liver Avg"].mean()),
            "ClinDiff-Lite Liver Avg": float(subset["ClinDiff-Lite Liver Avg"].mean()),
            "Grouped Liver Avg": float(subset["Grouped Liver Avg"].mean()),
            "Delta GroupedMinusClinDiff Overall": float((subset["ClinDiff-Lite Overall"] - subset["Grouped Overall"]).mean()),
            "Delta GroupedMinusClinDiff Liver Avg": float((subset["ClinDiff-Lite Liver Avg"] - subset["Grouped Liver Avg"]).mean()),
        })
    summary_df = pd.DataFrame(summary_rows)

    feature_summary = (
        feature_df.groupby(["MissingRate", "Feature"], as_index=False)[["Mean", "ClinDiff-Lite", "Grouped"]]
        .mean()
    )

    output_dir = ensure_output_dir()
    run_df.to_csv(os.path.join(output_dir, "optimized_group_repeated_runs.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "optimized_group_summary.csv"), index=False)
    feature_summary.to_csv(os.path.join(output_dir, "optimized_group_feature_summary.csv"), index=False)

    print(summary_df.to_string(index=False))
    print("\nPer-feature summary:")
    print(feature_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    run_grouped_optimization()

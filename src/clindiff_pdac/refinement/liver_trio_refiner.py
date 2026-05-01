"""
Lightweight gated refinement for the liver biomarker group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LIVER_TRIO = [
    "tb_before_treatment",
    "db_before_treatment",
    "ggt_before_treatment",
]

NON_NEGATIVE_COLUMNS = {
    "age",
    "survival_days",
    "fasting_glucose_mmol_L",
    "tb_before_treatment",
    "db_before_treatment",
    "ggt_before_treatment",
}


@dataclass
class GatingDecision:
    global_missingness: float
    observed_count: int
    predictor_coverage: float
    usable_predictor_count: int
    signal_strength_score: float
    transformed_distance: float
    mechanism_label: str
    mechanism_score: float
    use_grouped_refinement: bool
    reason: str


def _ridge_log_predict(
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


def apply_clinical_constraints(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NON_NEGATIVE_COLUMNS.intersection(out.columns):
        out[col] = np.clip(out[col], 0, None)
    if {"tb_before_treatment", "db_before_treatment"}.issubset(out.columns):
        out["db_before_treatment"] = np.minimum(
            out["db_before_treatment"],
            out["tb_before_treatment"],
        )
    return out


def _estimate_mechanism_context(raw_df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate whether missingness in the liver trio behaves more like MCAR or MAR.

    The score is based on the correlation between liver-trio missingness indicators
    and non-outcome observed predictors. Large correlations indicate that the
    missingness pattern is associated with available covariates and should trigger
    a more conservative route.
    """
    candidate_predictors = [
        col for col in raw_df.columns
        if col not in LIVER_TRIO and col not in {"survival_days", "survival_months"}
    ]
    candidate_predictors = [
        col for col in candidate_predictors
        if pd.api.types.is_numeric_dtype(raw_df[col])
    ]
    if not candidate_predictors:
        return {"label": "unknown", "score": 0.0}

    filled = raw_df[candidate_predictors].apply(pd.to_numeric, errors="coerce")
    filled = filled.fillna(filled.median())
    correlations = []
    for target in LIVER_TRIO:
        if target not in raw_df.columns:
            continue
        indicator = raw_df[target].isna().astype(float)
        if indicator.nunique() < 2:
            continue
        for predictor in candidate_predictors:
            series = filled[predictor]
            if series.nunique() < 2:
                continue
            corr = np.corrcoef(indicator, series)[0, 1]
            if np.isfinite(corr):
                correlations.append(abs(float(corr)))

    if not correlations:
        return {"label": "unknown", "score": 0.0}

    mechanism_score = float(np.median(correlations))
    if mechanism_score >= 0.12:
        label = "mar_like"
    elif mechanism_score <= 0.05:
        label = "mcar_like"
    else:
        label = "mixed_or_uncertain"
    return {"label": label, "score": mechanism_score}


def _row_distance_to_reference(
    row: pd.Series,
    reference_df: pd.DataFrame,
    usable_predictors: List[str],
) -> float:
    if not usable_predictors:
        return float("inf")
    ref = reference_df[usable_predictors].astype(float)
    ref_mean = ref.mean()
    ref_std = ref.std().replace(0, 1.0).fillna(1.0)
    row_values = row[usable_predictors].astype(float)
    z = ((row_values - ref_mean) / ref_std).abs()
    return float(np.nanmean(z.values))


def _decide_row_gate(
    row: pd.Series,
    global_missingness: float,
    reference_df: pd.DataFrame,
    mechanism_label: str,
    mechanism_score: float,
    min_observed: int = 2,
    max_global_missingness: float = 0.6,
    distance_threshold: float = 2.25,
) -> GatingDecision:
    observed_count = int(row[LIVER_TRIO].notna().sum())
    predictor_coverage = observed_count / len(LIVER_TRIO)
    usable_predictors = [
        col for col in ["age", "fasting_glucose_mmol_L", "tb_before_treatment", "db_before_treatment", "ggt_before_treatment"]
        if col in row.index and pd.notna(row[col])
    ]
    usable_predictor_count = len(usable_predictors)
    transformed_distance = _row_distance_to_reference(row, reference_df, usable_predictors)
    mechanism_penalty = 0.15 if mechanism_label == "mar_like" else (0.05 if mechanism_label == "mixed_or_uncertain" else 0.0)
    signal_strength_score = (
        0.45 * predictor_coverage
        + 0.35 * min(usable_predictor_count / 4.0, 1.0)
        + 0.20 * max(0.0, 1.0 - min(transformed_distance / distance_threshold, 1.0))
        - mechanism_penalty
    )

    if global_missingness > max_global_missingness:
        return GatingDecision(
            global_missingness=global_missingness,
            observed_count=observed_count,
            predictor_coverage=predictor_coverage,
            usable_predictor_count=usable_predictor_count,
            signal_strength_score=signal_strength_score,
            transformed_distance=transformed_distance,
            mechanism_label=mechanism_label,
            mechanism_score=mechanism_score,
            use_grouped_refinement=False,
            reason="high_global_missingness",
        )

    if observed_count < min_observed or usable_predictor_count < 2:
        return GatingDecision(
            global_missingness=global_missingness,
            observed_count=observed_count,
            predictor_coverage=predictor_coverage,
            usable_predictor_count=usable_predictor_count,
            signal_strength_score=signal_strength_score,
            transformed_distance=transformed_distance,
            mechanism_label=mechanism_label,
            mechanism_score=mechanism_score,
            use_grouped_refinement=False,
            reason="insufficient_row_signal",
        )

    if transformed_distance > distance_threshold:
        return GatingDecision(
            global_missingness=global_missingness,
            observed_count=observed_count,
            predictor_coverage=predictor_coverage,
            usable_predictor_count=usable_predictor_count,
            signal_strength_score=signal_strength_score,
            transformed_distance=transformed_distance,
            mechanism_label=mechanism_label,
            mechanism_score=mechanism_score,
            use_grouped_refinement=False,
            reason="out_of_distribution_signal",
        )

    if signal_strength_score < 0.55:
        return GatingDecision(
            global_missingness=global_missingness,
            observed_count=observed_count,
            predictor_coverage=predictor_coverage,
            usable_predictor_count=usable_predictor_count,
            signal_strength_score=signal_strength_score,
            transformed_distance=transformed_distance,
            mechanism_label=mechanism_label,
            mechanism_score=mechanism_score,
            use_grouped_refinement=False,
            reason="low_signal_strength",
        )

    return GatingDecision(
        global_missingness=global_missingness,
        observed_count=observed_count,
        predictor_coverage=predictor_coverage,
        usable_predictor_count=usable_predictor_count,
        signal_strength_score=signal_strength_score,
        transformed_distance=transformed_distance,
        mechanism_label=mechanism_label,
        mechanism_score=mechanism_score,
        use_grouped_refinement=True,
        reason="adaptive_grouped_refinement",
    )


def gated_liver_trio_refinement(
    original_df: pd.DataFrame,
    base_imputed_df: pd.DataFrame,
    global_missingness: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply grouped liver-trio refinement only when the row has enough recoverable signal.

    Returns
    -------
    refined_df, routing_df
    """
    out = base_imputed_df.copy().astype(float)
    raw = original_df.copy().astype(float)
    medians = raw.median()
    base = out.fillna(medians).copy()
    mechanism_context = _estimate_mechanism_context(raw)
    reference_df = base.copy()

    tb_predictors = [c for c in ["db_before_treatment", "ggt_before_treatment", "age", "fasting_glucose_mmol_L"] if c in out.columns]
    db_predictors = [c for c in ["tb_before_treatment", "ggt_before_treatment", "age"] if c in out.columns]
    ggt_predictors = [c for c in ["tb_before_treatment", "db_before_treatment", "fasting_glucose_mmol_L", "age"] if c in out.columns]

    tb_pred = _ridge_log_predict(base, raw, "tb_before_treatment", [c for c in tb_predictors if c != "tb_before_treatment"])
    ratio_rows = raw[["tb_before_treatment", "db_before_treatment"]].dropna()
    ratio_rows = ratio_rows[ratio_rows["tb_before_treatment"] > 0]
    ratio_prior = float((ratio_rows["db_before_treatment"] / ratio_rows["tb_before_treatment"]).median()) if len(ratio_rows) else 0.3

    temp_after_tb = out.copy().fillna(medians)
    tb_missing_global = raw["tb_before_treatment"].isna()
    temp_after_tb.loc[tb_missing_global, "tb_before_treatment"] = tb_pred.loc[tb_missing_global]
    db_model = _ridge_log_predict(
        temp_after_tb,
        raw,
        "db_before_treatment",
        [c for c in db_predictors if c != "db_before_treatment"],
    )

    temp_after_db = temp_after_tb.copy()
    db_missing_global = raw["db_before_treatment"].isna()
    db_ratio_pred_global = ratio_prior * temp_after_tb["tb_before_treatment"]
    temp_after_db.loc[db_missing_global, "db_before_treatment"] = (
        0.7 * db_model.loc[db_missing_global] + 0.3 * db_ratio_pred_global.loc[db_missing_global]
    )
    ggt_model = _ridge_log_predict(
        temp_after_db,
        raw,
        "ggt_before_treatment",
        [c for c in ggt_predictors if c != "ggt_before_treatment"],
    )

    routing_rows: List[Dict[str, object]] = []

    for idx, row in raw.iterrows():
        gate = _decide_row_gate(
            row,
            global_missingness=global_missingness,
            reference_df=reference_df,
            mechanism_label=mechanism_context["label"],
            mechanism_score=mechanism_context["score"],
        )
        routing_rows.append({
            "row_index": int(idx) if isinstance(idx, (int, np.integer)) else idx,
            "global_missingness": gate.global_missingness,
            "liver_trio_observed_count": gate.observed_count,
            "predictor_coverage": gate.predictor_coverage,
            "usable_predictor_count": gate.usable_predictor_count,
            "signal_strength_score": gate.signal_strength_score,
            "transformed_distance": gate.transformed_distance,
            "mechanism_label": gate.mechanism_label,
            "mechanism_score": gate.mechanism_score,
            "used_grouped_refinement": gate.use_grouped_refinement,
            "gate_reason": gate.reason,
        })

        if not gate.use_grouped_refinement:
            continue

        row_missing = row[LIVER_TRIO].isna()
        if row_missing["tb_before_treatment"]:
            out.at[idx, "tb_before_treatment"] = tb_pred.at[idx]

        tb_value = out.at[idx, "tb_before_treatment"]
        if row_missing["db_before_treatment"]:
            db_ratio_pred = ratio_prior * tb_value if pd.notna(tb_value) else np.nan
            db_candidate = db_model.at[idx]
            if pd.notna(db_ratio_pred):
                db_candidate = 0.7 * db_candidate + 0.3 * db_ratio_pred
            out.at[idx, "db_before_treatment"] = db_candidate

        if row_missing["ggt_before_treatment"]:
            out.at[idx, "ggt_before_treatment"] = ggt_model.at[idx]

    out = apply_clinical_constraints(out)
    routing_df = pd.DataFrame(routing_rows)
    return out, routing_df

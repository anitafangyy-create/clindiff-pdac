#!/usr/bin/env python3
"""
Minimal Experiment: ClinDiff-PDAC vs Mean/Median
Fast version - no sklearn, no heavy baselines
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
import json
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from clindiff_pdac.refinement import gated_liver_trio_refinement

OUTCOME_COLUMNS = {"survival_days"}
LIVER_TRIO = {
    "tb_before_treatment",
    "db_before_treatment",
    "ggt_before_treatment",
}
HEADLINE_NON_OUTCOME_COLUMNS = {
    "age",
    "fasting_glucose_mmol_L",
    *LIVER_TRIO,
}


def ensure_output_dir():
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_data():
    path = "data/raw/pancreatic_cancer_data_normalized_clean.csv"
    df = pd.read_csv(path)
    numeric_cols = ['age', 'survival_days', 'fasting_glucose_mmol_L',
                    'tb_before_treatment', 'ggt_before_treatment', 'db_before_treatment']
    cols = [c for c in numeric_cols if c in df.columns]
    df = df[cols].dropna(how='all').copy()
    print(f"Data: {df.shape[0]} samples x {len(cols)} features, "
          f"base missing={df.isna().mean().mean():.1%}")
    return df, cols

def introduce_missing(df, rate=0.3, seed=42):
    """
    Apply artificial masking only to cells that were originally observed.

    The evaluation mask therefore corresponds to additional synthetic missingness
    rather than to a mixture of natural and synthetic missing cells.
    """
    rng = np.random.RandomState(seed)
    df2 = df.copy()
    observed = df2.notna().values
    mask = (rng.rand(*df2.shape) < rate) & observed

    # Preserve at least one originally observed value per row so that every row
    # retains some anchor signal for routed imputation.
    observed_counts = observed.sum(axis=1)
    masked_counts = mask.sum(axis=1)
    for row_idx in np.where((observed_counts > 0) & (masked_counts >= observed_counts))[0]:
        observed_cols = np.flatnonzero(observed[row_idx])
        keep_col = rng.choice(observed_cols)
        mask[row_idx, keep_col] = False

    df2 = df2.mask(mask)
    return df2, mask

def rmse(true, pred, mask, columns=None):
    if columns is None:
        column_idx = np.arange(true.shape[1])
    else:
        column_idx = np.array([true.columns.get_loc(col) for col in columns if col in true.columns])
    if len(column_idx) == 0:
        return float('nan')
    mask_subset = mask[:, column_idx]
    t = true.iloc[:, column_idx].values[mask_subset]
    p = pred.iloc[:, column_idx].values[mask_subset]
    # Only evaluate where ground truth is also observed
    ok = (~np.isnan(t)) & (~np.isnan(p))
    if ok.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.mean((t[ok] - p[ok])**2)))


def per_feature_rmse(true, pred, mask):
    metrics = {}
    for col in true.columns:
        col_mask = mask[:, true.columns.get_loc(col)]
        t = true[col].values[col_mask]
        p = pred[col].values[col_mask]
        ok = (~np.isnan(t)) & (~np.isnan(p))
        if ok.sum() == 0:
            metrics[col] = float("nan")
        else:
            metrics[col] = float(np.sqrt(np.mean((t[ok] - p[ok]) ** 2)))
    return metrics

def mean_impute(df):
    return df.fillna(df.mean())

def median_impute(df):
    return df.fillna(df.median())

def _fit_regression_predictor(train_df, raw_df, target, corr_threshold=0.15, top_k=3, reg=0.5):
    """
    Fit a lightweight feature-to-feature predictor for one target column.

    Returns a Series of predictions for all rows and the predictor columns used.
    """
    correlations = train_df.corr().abs()[target].drop(target).sort_values(ascending=False)
    disallowed = set()
    if target in LIVER_TRIO:
        # Avoid outcome leakage in the headline biomarker-group analysis.
        disallowed |= OUTCOME_COLUMNS
    predictors = [
        col for col, value in correlations.items()
        if value >= corr_threshold and col not in disallowed
    ][:top_k]

    if not predictors:
        return pd.Series(np.nan, index=train_df.index), []

    observed_rows = ~raw_df[target].isna()
    X_train = train_df.loc[observed_rows, predictors].values.astype(float)
    y_train = raw_df.loc[observed_rows, target].values.astype(float)

    target_observed = raw_df[target].dropna()
    positive_skewed = (
        (target_observed >= 0).all()
        and target_observed.quantile(0.95) > 3 * max(target_observed.quantile(0.50), 1e-6)
    )

    if positive_skewed:
        X_train = np.log1p(np.clip(X_train, 0, None))
        y_train = np.log1p(y_train)

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
    if positive_skewed:
        X_all = np.log1p(np.clip(X_all, 0, None))

    predictions = ((X_all - x_mean) / x_std) @ beta
    predictions = predictions * y_std + y_mean

    if positive_skewed:
        predictions = np.expm1(predictions)

    lower, upper = target_observed.quantile([0.01, 0.99])
    predictions = np.clip(predictions, lower, upper)

    return pd.Series(predictions, index=train_df.index), predictors


def clindiff_lite(df, seed=42):
    """
    Lightweight ClinDiff with per-feature strategy selection.

    For each target column we evaluate mean, median, and a correlation-driven
    regression imputer on a small validation split of observed values, then use
    the best-performing strategy to fill the real missing entries.
    """
    rng = np.random.RandomState(seed)

    out = df.copy().astype(float)
    col_mean = df.mean()
    col_median = df.median()
    base_filled = df.fillna(col_median)
    chosen = {}

    for target in df.columns:
        observed_idx = np.flatnonzero(df[target].notna().values)

        if len(observed_idx) < 30:
            chosen[target] = "median"
            out[target] = df[target].fillna(col_median[target])
            continue

        holdout_size = max(10, len(observed_idx) // 5)
        holdout_idx = rng.choice(observed_idx, size=holdout_size, replace=False)

        validation_df = df.copy()
        validation_df.iloc[holdout_idx, validation_df.columns.get_loc(target)] = np.nan
        validation_filled = validation_df.fillna(col_median)

        candidates = {
            "mean": pd.Series(np.full(len(df), col_mean[target]), index=df.index),
            "median": pd.Series(np.full(len(df), col_median[target]), index=df.index),
        }

        reg_predictions, predictors = _fit_regression_predictor(validation_filled, validation_df, target)
        if predictors:
            candidates["regression"] = reg_predictions

        truth = df.iloc[holdout_idx][target].values
        best_name = "median"
        best_score = float("inf")

        for name, pred in candidates.items():
            values = pred.iloc[holdout_idx].values
            score = np.sqrt(np.mean((truth - values) ** 2))
            if score < best_score:
                best_score = score
                best_name = name

        chosen[target] = best_name

        if best_name == "mean":
            out[target] = df[target].fillna(col_mean[target])
        elif best_name == "median":
            out[target] = df[target].fillna(col_median[target])
        else:
            final_predictions, _ = _fit_regression_predictor(base_filled, df, target)
            filled = df[target].copy()
            missing_rows = filled.isna()
            filled.loc[missing_rows] = final_predictions.loc[missing_rows]
            out[target] = filled.fillna(col_median[target])

    return out, chosen


def clindiff_gated(df, seed=42, base_out=None, chosen=None):
    """
    Base ClinDiff-Lite with an adaptive liver-trio refinement branch.
    """
    if base_out is None or chosen is None:
        base_out, chosen = clindiff_lite(df, seed=seed)
    global_missingness = float(df.isna().mean().mean())
    refined_out, routing_df = gated_liver_trio_refinement(
        original_df=df,
        base_imputed_df=base_out,
        global_missingness=global_missingness,
    )
    grouped_usage = float(routing_df["used_grouped_refinement"].mean()) if len(routing_df) else 0.0
    chosen = dict(chosen)
    chosen["liver_trio_gate"] = (
        "adaptive_grouped_refinement" if grouped_usage > 0 else "fallback_only"
    )
    return refined_out, chosen, routing_df


def export_results(summary_df, feature_df, strategy_df):
    output_dir = ensure_output_dir()
    summary_path = os.path.join(output_dir, "minimal_experiment_summary.csv")
    feature_path = os.path.join(output_dir, "minimal_experiment_per_feature.csv")
    strategy_path = os.path.join(output_dir, "minimal_experiment_strategies.csv")

    summary_df.to_csv(summary_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    strategy_df.to_csv(strategy_path, index=False)

    print("\nSaved outputs:")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Per-feature CSV: {feature_path}")
    print(f"  Strategy CSV: {strategy_path}")


def paired_bootstrap_ci(values, n_boot=5000, seed=7):
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(sample.mean())
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def paired_permutation_pvalue(values, n_perm=10000, seed=11):
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    observed = abs(values.mean())
    stats = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(values))
        stats.append(abs((values * signs).mean()))
    stats = np.asarray(stats)
    return float((np.sum(stats >= observed) + 1) / (len(stats) + 1))


def summarize_repeated_runs(records, methods):
    summary_rows = []
    liver_methods = [
        "Mean Liver Avg",
        "ClinDiff-Lite Liver Avg",
        "ClinDiff-Gated Liver Avg",
    ]
    for rate in sorted(records["MissingRate"].unique(), key=lambda x: float(x.strip("%"))):
        subset = records[records["MissingRate"] == rate]
        row = {"MissingRate": rate, "Repeats": int(len(subset))}
        for method in methods:
            values = subset[method].values.astype(float)
            mean_val = float(values.mean())
            ci_low, ci_high = np.quantile(values, [0.025, 0.975])
            row[f"{method} Mean"] = mean_val
            row[f"{method} CI Low"] = float(ci_low)
            row[f"{method} CI High"] = float(ci_high)
        for method in liver_methods:
            values = subset[method].values.astype(float)
            mean_val = float(values.mean())
            ci_low, ci_high = np.quantile(values, [0.025, 0.975])
            row[f"{method} Mean"] = mean_val
            row[f"{method} CI Low"] = float(ci_low)
            row[f"{method} CI High"] = float(ci_high)

        comparisons = [
            ("Mean", "ClinDiff-Lite"),
            ("Mean", "ClinDiff-Gated"),
            ("ClinDiff-Lite", "ClinDiff-Gated"),
            ("Mean Liver Avg", "ClinDiff-Lite Liver Avg"),
            ("Mean Liver Avg", "ClinDiff-Gated Liver Avg"),
            ("ClinDiff-Lite Liver Avg", "ClinDiff-Gated Liver Avg"),
        ]
        for left, right in comparisons:
            if left not in subset.columns or right not in subset.columns:
                continue
            delta = subset[left].values.astype(float) - subset[right].values.astype(float)
            base_name = f"Delta {left.replace(' ', '')}Minus{right.replace(' ', '')}"
            row[f"{base_name} Mean"] = float(delta.mean())
            ci_low, ci_high = paired_bootstrap_ci(delta)
            row[f"{base_name} Bootstrap CI Low"] = ci_low
            row[f"{base_name} Bootstrap CI High"] = ci_high
            row[f"{base_name} Permutation P"] = paired_permutation_pvalue(delta)

        if "Grouped Usage Rate" in subset.columns:
            row["Grouped Usage Rate Mean"] = float(subset["Grouped Usage Rate"].mean())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def export_repeated_results(run_df, summary_df, prefix="minimal_experiment"):
    output_dir = ensure_output_dir()
    runs_path = os.path.join(output_dir, f"{prefix}_repeated_runs.csv")
    summary_path = os.path.join(output_dir, f"{prefix}_repeated_summary.csv")

    run_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved repeated-masking outputs:")
    print(f"  Repeated runs CSV: {runs_path}")
    print(f"  Repeated summary CSV: {summary_path}")


def export_additional_outputs(feature_df, routing_df, prefix):
    output_dir = ensure_output_dir()
    feature_path = os.path.join(output_dir, f"{prefix}_feature_summary.csv")
    routing_path = os.path.join(output_dir, f"{prefix}_routing_summary.csv")
    feature_df.to_csv(feature_path, index=False)
    routing_df.to_csv(routing_path, index=False)
    print(f"  Feature summary CSV: {feature_path}")
    print(f"  Routing summary CSV: {routing_path}")


def export_metadata(metadata, prefix):
    output_dir = ensure_output_dir()
    metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Metadata JSON: {metadata_path}")


def repeated_masking_evaluation(df, rates, repeats=20, base_seed=2026):
    records = []
    feature_records = []
    routing_records = []
    methods = ["Mean", "Median", "ClinDiff-Lite", "ClinDiff-Gated"]
    headline_columns = [c for c in df.columns if c in HEADLINE_NON_OUTCOME_COLUMNS]
    liver_cols = [c for c in df.columns if c in LIVER_TRIO]

    for rate in rates:
        for repeat in range(repeats):
            seed = base_seed + repeat + int(rate * 1000)
            df_miss, mask = introduce_missing(df, rate, seed=seed)
            df_mean = mean_impute(df_miss)
            df_median = median_impute(df_miss)
            df_clindiff, chosen = clindiff_lite(df_miss, seed=seed)
            df_gated, _, routing_df = clindiff_gated(
                df_miss,
                seed=seed,
                base_out=df_clindiff,
                chosen=chosen,
            )

            mean_feature = per_feature_rmse(df, df_mean, mask)
            clindiff_feature = per_feature_rmse(df, df_clindiff, mask)
            gated_feature = per_feature_rmse(df, df_gated, mask)

            mean_liver_avg = float(np.mean([mean_feature[c] for c in liver_cols]))
            clindiff_liver_avg = float(np.mean([clindiff_feature[c] for c in liver_cols]))
            gated_liver_avg = float(np.mean([gated_feature[c] for c in liver_cols]))

            records.append({
                "MissingRate": f"{rate:.0%}",
                "Repeat": repeat + 1,
                "Seed": seed,
                "Mean": rmse(df, df_mean, mask, columns=headline_columns),
                "Median": rmse(df, df_median, mask, columns=headline_columns),
                "ClinDiff-Lite": rmse(df, df_clindiff, mask, columns=headline_columns),
                "ClinDiff-Gated": rmse(df, df_gated, mask, columns=headline_columns),
                "Mean Liver Avg": mean_liver_avg,
                "ClinDiff-Lite Liver Avg": clindiff_liver_avg,
                "ClinDiff-Gated Liver Avg": gated_liver_avg,
                "Grouped Usage Rate": float(routing_df["used_grouped_refinement"].mean()) if len(routing_df) else 0.0,
                "Masked Observed Cells": int(mask.sum()),
                "Masked Headline Cells": int(mask[:, [df.columns.get_loc(c) for c in headline_columns]].sum()),
                "Masked Liver Cells": int(mask[:, [df.columns.get_loc(c) for c in liver_cols]].sum()),
            })

            for feature in df.columns:
                feature_records.append({
                    "MissingRate": f"{rate:.0%}",
                    "Repeat": repeat + 1,
                    "Feature": feature,
                    "Mean": mean_feature[feature],
                    "ClinDiff-Lite": clindiff_feature[feature],
                    "ClinDiff-Gated": gated_feature[feature],
                })

            routing_records.append({
                "MissingRate": f"{rate:.0%}",
                "Repeat": repeat + 1,
                "Grouped Usage Rate": float(routing_df["used_grouped_refinement"].mean()) if len(routing_df) else 0.0,
                "Fallback Usage Rate": 1.0 - (float(routing_df["used_grouped_refinement"].mean()) if len(routing_df) else 0.0),
                "Median Liver Trio Observed Count": float(routing_df["liver_trio_observed_count"].median()) if len(routing_df) else float("nan"),
                "Mean Usable Predictor Count": float(routing_df["usable_predictor_count"].mean()) if len(routing_df) else float("nan"),
                "Mean Signal Strength Score": float(routing_df["signal_strength_score"].mean()) if len(routing_df) else float("nan"),
                "Median Transformed Distance": float(routing_df["transformed_distance"].median()) if len(routing_df) else float("nan"),
                "Mechanism Score": float(routing_df["mechanism_score"].mean()) if len(routing_df) else float("nan"),
                "Mechanism Label": routing_df["mechanism_label"].mode().iloc[0] if len(routing_df) else "unknown",
                "Top Gate Reason": routing_df["gate_reason"].mode().iloc[0] if len(routing_df) else "unknown",
            })

    run_df = pd.DataFrame(records)
    summary_df = summarize_repeated_runs(run_df, methods)
    feature_df = (
        pd.DataFrame(feature_records)
        .groupby(["MissingRate", "Feature"], as_index=False)[["Mean", "ClinDiff-Lite", "ClinDiff-Gated"]]
        .mean()
    )
    routing_df = pd.DataFrame(routing_records)
    return run_df, summary_df, feature_df, routing_df

def main():
    print("=" * 60)
    print("ClinDiff-PDAC: Minimal Validation Experiment")
    print("=" * 60)
    
    df, cols = load_data()
    headline_columns = [c for c in cols if c in HEADLINE_NON_OUTCOME_COLUMNS]
    
    results = []
    feature_results = []
    strategy_results = []
    for rate in [0.2, 0.4, 0.6]:
        df_miss, mask = introduce_missing(df, rate)
        row = {"MissingRate": f"{rate:.0%}"}
        
        t0 = time.time()
        df_mean = mean_impute(df_miss)
        row["Mean"] = rmse(df, df_mean, mask, columns=headline_columns)
        row["t_Mean"]  = time.time() - t0
        
        t0 = time.time()
        df_median = median_impute(df_miss)
        row["Median"] = rmse(df, df_median, mask, columns=headline_columns)
        row["t_Median"] = time.time() - t0
        
        t0 = time.time()
        df_clindiff, chosen = clindiff_lite(df_miss, seed=42)
        row["ClinDiff-Lite"] = rmse(df, df_clindiff, mask, columns=headline_columns)
        row["t_ClinDiff"] = time.time() - t0

        t0 = time.time()
        df_gated, gated_chosen, _ = clindiff_gated(
            df_miss,
            seed=42,
            base_out=df_clindiff,
            chosen=chosen,
        )
        row["ClinDiff-Gated"] = rmse(df, df_gated, mask, columns=headline_columns)
        row["t_ClinDiff_Gated"] = time.time() - t0

        feature_table = pd.DataFrame({
            "Feature": cols,
            "Mean": [per_feature_rmse(df, df_mean, mask)[col] for col in cols],
            "Median": [per_feature_rmse(df, df_median, mask)[col] for col in cols],
            "ClinDiff-Lite": [per_feature_rmse(df, df_clindiff, mask)[col] for col in cols],
            "ClinDiff-Gated": [per_feature_rmse(df, df_gated, mask)[col] for col in cols],
        })
        feature_table.insert(0, "MissingRate", f"{rate:.0%}")
        feature_results.append(feature_table)
        strategy_results.append(pd.DataFrame({
            "MissingRate": [f"{rate:.0%}"] * len(cols),
            "Feature": cols,
            "ClinDiff-Lite Strategy": [chosen[col] for col in cols],
            "ClinDiff-Gated Strategy": [gated_chosen.get(col, chosen.get(col, "unknown")) for col in cols],
        }))
        
        print(f"\nMissing {rate:.0%}:")
        print(f"  Mean         RMSE={row['Mean']:.4f}  ({row['t_Mean']:.2f}s)")
        print(f"  Median       RMSE={row['Median']:.4f}  ({row['t_Median']:.2f}s)")
        print(f"  ClinDiff-Lite RMSE={row['ClinDiff-Lite']:.4f}  ({row['t_ClinDiff']:.2f}s)")
        print(f"  ClinDiff-Gated RMSE={row['ClinDiff-Gated']:.4f}  ({row['t_ClinDiff_Gated']:.2f}s)")
        print(f"  ClinDiff-Lite strategies: {chosen}")
        print(f"  ClinDiff-Gated gate: {gated_chosen['liver_trio_gate']}")
        print("\n  Per-feature RMSE:")
        print(feature_table.drop(columns=["MissingRate"]).round(4).to_string(index=False))
        results.append(row)
    
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    df_res = pd.DataFrame(results)[["MissingRate","Mean","Median","ClinDiff-Lite","ClinDiff-Gated"]]
    print(df_res.to_string(index=False))
    
    # Improvement
    for _, r in df_res.iterrows():
        imp = (r["Mean"] - r["ClinDiff-Lite"]) / r["Mean"] * 100
        gated_imp = (r["Mean"] - r["ClinDiff-Gated"]) / r["Mean"] * 100
        print(f"  {r['MissingRate']}: ClinDiff-Lite vs Mean: {imp:+.1f}%")
        print(f"  {r['MissingRate']}: ClinDiff-Gated vs Mean: {gated_imp:+.1f}%")

    print("\n" + "=" * 60)
    print("PER-FEATURE SUMMARY")
    print("=" * 60)
    feature_summary = pd.concat(feature_results, ignore_index=True)
    strategy_summary = pd.concat(strategy_results, ignore_index=True)
    for rate in feature_summary["MissingRate"].unique():
        print(f"\nMissing {rate}:")
        print(
            feature_summary[feature_summary["MissingRate"] == rate]
            .drop(columns=["MissingRate"])
            .round(4)
            .to_string(index=False)
        )

    export_results(df_res, feature_summary, strategy_summary)

    repeated_runs, repeated_summary, repeated_feature_summary, repeated_routing_summary = repeated_masking_evaluation(
        df, rates=[0.2, 0.4, 0.6], repeats=100
    )
    export_repeated_results(repeated_runs, repeated_summary, prefix="gated_experiment")
    export_additional_outputs(repeated_feature_summary, repeated_routing_summary, prefix="gated_experiment")
    export_metadata({
        "cohort_rows": int(df.shape[0]),
        "source_columns": cols,
        "headline_non_outcome_columns": headline_columns,
        "primary_endpoint": "mean_rmse_liver_trio_original_units",
        "secondary_endpoint": "rmse_non_outcome_columns_original_units",
        "liver_trio_columns": sorted(LIVER_TRIO),
        "outcome_columns_excluded_from_headline_metrics": sorted(OUTCOME_COLUMNS),
        "masking_design": {
            "artificial_missingness": "MCAR",
            "rates": [0.2, 0.4, 0.6],
            "repeats_per_rate": 100,
            "mask_only_originally_observed_cells": True,
            "preserve_at_least_one_observed_value_per_row": True,
        },
        "training_and_refit": {
            "refit_strategy_selection_each_seed": True,
            "validation_holdout_drawn_from_originally_observed_values": True,
            "outcome_columns_excluded_from_biomarker_predictors": sorted(OUTCOME_COLUMNS),
        },
        "statistics": {
            "paired_bootstrap_resamples": 5000,
            "paired_bootstrap_unit": "seed_level_delta",
            "paired_permutation_resamples": 10000,
            "paired_permutation_test": "two_sided_sign_flip",
        },
        "transforms_and_constraints": {
            "regression_transform": "log1p on non-negative predictors and targets",
            "inverse_transform": "expm1 back to original units",
            "winsorization": "clip predictions to observed 1st-99th percentile range",
            "non_negative_truncation": True,
            "clinical_constraint": "DB <= TB",
        },
        "gated_branch": {
            "global_missingness_threshold": 0.60,
            "min_liver_trio_observed": 2,
            "min_usable_predictors": 2,
            "distance_threshold": 2.25,
            "signal_strength_threshold": 0.55,
        },
    }, prefix="gated_experiment")

    print("\n" + "=" * 60)
    print("REPEATED MASKING SUMMARY")
    print("=" * 60)
    for _, row in repeated_summary.iterrows():
        print(f"\nMissing {row['MissingRate']} ({int(row['Repeats'])} repeats):")
        for method in ["Mean", "Median", "ClinDiff-Lite", "ClinDiff-Gated"]:
            print(
                f"  {method:<13}"
                f"mean={row[f'{method} Mean']:.4f}  "
                f"95% CI [{row[f'{method} CI Low']:.4f}, {row[f'{method} CI High']:.4f}]"
            )
        print(
            "  Paired overall "
            f"Mean-Gated={row['Delta MeanMinusClinDiff-Gated Mean']:.4f}  "
            f"bootstrap 95% CI [{row['Delta MeanMinusClinDiff-Gated Bootstrap CI Low']:.4f}, "
            f"{row['Delta MeanMinusClinDiff-Gated Bootstrap CI High']:.4f}]  "
            f"perm-p={row['Delta MeanMinusClinDiff-Gated Permutation P']:.4f}"
        )
        print(
            "  Paired trio    "
            f"Mean-Gated={row['Delta MeanLiverAvgMinusClinDiff-GatedLiverAvg Mean']:.4f}  "
            f"bootstrap 95% CI [{row['Delta MeanLiverAvgMinusClinDiff-GatedLiverAvg Bootstrap CI Low']:.4f}, "
            f"{row['Delta MeanLiverAvgMinusClinDiff-GatedLiverAvg Bootstrap CI High']:.4f}]  "
            f"perm-p={row['Delta MeanLiverAvgMinusClinDiff-GatedLiverAvg Permutation P']:.4f}"
        )
        if "Grouped Usage Rate Mean" in row:
            print(f"  Grouped usage  mean={row['Grouped Usage Rate Mean']:.3f}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()

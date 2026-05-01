#!/usr/bin/env python3
"""
Full Experiment: ClinDiff-PDAC vs All Baselines
===============================================

Comprehensive evaluation on real PDAC data with all 4 modules integrated.
"""

import sys
import os
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from clindiff_pdac.engine.rule_engine import RuleEngine, VariableSpec
from clindiff_pdac.missingness.missingness_analyzer import MissingnessAnalyzer
from clindiff_pdac.baselines.enhanced_baselines import (
    MissForestImputer, KNNImputer, MICEImputer
)
from clindiff_pdac.llm.llm_constraints import LLMConstraintLayer
from clindiff_pdac.imputer import ClinDiffPDAC


def load_data():
    """Load and prepare real PDAC data."""
    path = "data/raw/pancreatic_cancer_data_normalized_clean.csv"
    df = pd.read_csv(path)
    
    # Select key variables for analysis
    numeric_cols = [
        'age', 'survival_days', 'fasting_glucose_mmol_L',
        'tb_before_treatment', 'ggt_before_treatment', 'db_before_treatment'
    ]
    
    # Check which columns exist
    available_cols = [c for c in numeric_cols if c in df.columns]
    df = df[available_cols].copy()
    
    print(f"📊 Loaded data: {df.shape}")
    print(f"   Variables: {', '.join(available_cols)}")
    print(f"   Missing rate: {df.isna().mean().mean():.1%}")
    
    return df, available_cols


def introduce_missing(df, missing_rate=0.3, random_state=42):
    """Artificially introduce MCAR missing for evaluation."""
    np.random.seed(random_state)
    df_missing = df.copy()
    mask = np.random.rand(*df.shape) < missing_rate
    df_missing = df_missing.mask(mask)
    
    # Keep at least some observed values per row
    for i in range(len(df_missing)):
        if df_missing.iloc[i].isna().all():
            col = np.random.choice(df_missing.columns)
            df_missing.iloc[i, df_missing.columns.get_loc(col)] = df.iloc[i, df.columns.get_loc(col)]
    
    return df_missing, mask


def evaluate_imputation(df_true, df_imputed, mask, method_name):
    """Calculate RMSE, MAE, R2 for imputed values only."""
    imputed_positions = mask
    
    if imputed_positions.sum() == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    
    true_values = df_true.values[imputed_positions]
    imputed_values = df_imputed.values[imputed_positions]
    
    # Remove any NaN in predictions
    valid = ~np.isnan(imputed_values)
    if valid.sum() == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    
    true_values = true_values[valid]
    imputed_values = imputed_values[valid]
    
    mse = np.mean((true_values - imputed_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_values - imputed_values))
    
    ss_res = np.sum((true_values - imputed_values) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def run_experiment(missing_rates=[0.2, 0.4], n_runs=2):
    """Run complete experiment across multiple missing rates."""
    
    print("\n" + "=" * 70)
    print("🧬 ClinDiff-PDAC: Full Experimental Validation")
    print("=" * 70)
    
    # Load data
    df_full, cols = load_data()
    
    # Initialize all methods
    methods = {
        "Mean": lambda: SimpleImputer(strategy="mean"),
        "Median": lambda: SimpleImputer(strategy="median"),
        "KNN": lambda: KNNImputer(n_neighbors=5),
        "MICE": lambda: MICEImputer(max_iter=10, random_state=42),
        "MissForest": lambda: MissForestImputer(max_iter=5, n_estimators=20, random_state=42),
        "ClinDiff-FCI": lambda: ClinDiffPDAC(
            n_features=len(cols),
            embedding_dim=32,
            diffusion_steps=50,
            feature_correlation_weight=0.3,
            num_epochs=50,
            batch_size=64
        ),
    }
    
    results = []
    
    for missing_rate in missing_rates:
        print(f"\n{'=' * 70}")
        print(f"📊 Missing Rate: {missing_rate:.0%}")
        print(f"{'=' * 70}")
        
        run_results = {name: [] for name in methods.keys()}
        
        for run in range(n_runs):
            print(f"\n  Run {run + 1}/{n_runs}...")
            
            # Introduce missing
            df_missing, mask = introduce_missing(df_full, missing_rate, random_state=42 + run)
            
            # Test each method
            for name, method_factory in methods.items():
                try:
                    start_time = time.time()
                    method = method_factory()
                    
                    if name == "ClinDiff-FCI":
                        # Diffusion model
                        df_imputed = method.fit_transform(df_missing.values)
                        df_imputed = pd.DataFrame(df_imputed, columns=cols, index=df_missing.index)
                    else:
                        # Standard imputer
                        df_imputed = method.fit_transform(df_missing)
                    
                    elapsed = time.time() - start_time
                    metrics = evaluate_imputation(df_full, df_imputed, mask, name)
                    metrics["Time"] = elapsed
                    
                    run_results[name].append(metrics)
                    print(f"    {name:15s} RMSE={metrics['RMSE']:.4f} ({elapsed:.1f}s)")
                    
                except Exception as e:
                    print(f"    {name:15s} FAILED: {str(e)[:50]}")
                    run_results[name].append({"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "Time": np.nan})
        
        # Aggregate results
        for name in methods.keys():
            rmse_vals = [r["RMSE"] for r in run_results[name] if not np.isnan(r["RMSE"])]
            time_vals = [r["Time"] for r in run_results[name] if not np.isnan(r["Time"])]
            
            results.append({
                "MissingRate": f"{missing_rate:.0%}",
                "Method": name,
                "RMSE_mean": np.mean(rmse_vals) if rmse_vals else np.nan,
                "RMSE_std": np.std(rmse_vals) if len(rmse_vals) > 1 else 0,
                "Time_mean": np.mean(time_vals) if time_vals else np.nan,
            })
    
    # Print final results table
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    pivot_rmse = results_df.pivot(index="Method", columns="MissingRate", values="RMSE_mean")
    
    print("\nRMSE by Missing Rate:")
    print(pivot_rmse.round(4).to_string())
    
    # Find best method per missing rate
    print("\n🏆 Best Method per Missing Rate:")
    for col in pivot_rmse.columns:
        best_method = pivot_rmse[col].idxmin()
        best_rmse = pivot_rmse[col].min()
        print(f"  {col}: {best_method} (RMSE={best_rmse:.4f})")
    
    return results_df


class SimpleImputer:
    """Simple mean/median imputer."""
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.values_ = None
    
    def fit(self, df):
        if self.strategy == "mean":
            self.values_ = df.mean()
        else:
            self.values_ = df.median()
        return self
    
    def transform(self, df):
        return df.fillna(self.values_)
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)


if __name__ == "__main__":
    results = run_experiment(missing_rates=[0.2, 0.4, 0.6], n_runs=3)
    print("\n✅ Full experiment completed!")

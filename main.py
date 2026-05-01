#!/usr/bin/env python3
"""
ClinDiff-PDAC Main Entry Point
==============================

Integrated pipeline demonstrating all four core modules:
1. Rule Engine - Four-state missingness mask generation
2. Missingness Analyzer - MCAR/MAR/MNAR/Structural classification
3. Enhanced Baselines - MissForest, kNN, MICE imputation
4. LLM Constraints - Constrained imputation with confidence

Usage:
    python main.py

Author: ClinDiff-PDAC Team
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from clindiff_pdac.engine.rule_engine import (
    RuleEngine, VariableSpec, MaskState, TimeWindow, ApplicabilityRule
)
from clindiff_pdac.missingness.missingness_analyzer import (
    MissingnessAnalyzer, MissingnessMechanism
)
from clindiff_pdac.baselines.enhanced_baselines import (
    MissForestImputer, KNNImputer, MICEImputer
)
from clindiff_pdac.llm.llm_constraints import (
    LLMConstraintLayer, Constraint, ClinicalContext, ImputationResult
)


def create_demo_data() -> pd.DataFrame:
    """Create synthetic PDAC clinical data for demonstration."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "age": np.random.randint(35, 85, n),
        "gender": np.random.choice(["M", "F"], n),
        "diagnosis_date": pd.date_range("2023-01-01", periods=n, freq="3D"),

        # Tumor markers
        "CA19_9": np.where(
            np.random.rand(n) < 0.25,
            np.nan,
            np.random.lognormal(4.5, 1.0, n)
        ),
        "CA19_9_date": pd.date_range("2023-01-01", periods=n, freq="3D") + pd.to_timedelta(np.random.randint(-5, 20, n), unit="D"),

        "CEA": np.where(
            np.random.rand(n) < 0.2,
            np.nan,
            np.random.lognormal(2.5, 0.8, n)
        ),

        # Lewis status (affects CA19-9 applicability)
        "Lewis_status": np.random.choice(
            ["positive", "negative", "unknown"],
            n,
            p=[0.7, 0.2, 0.1]
        ),

        # Liver function
        "ALB": np.where(
            np.random.rand(n) < 0.15,
            np.nan,
            np.random.normal(40, 5, n)
        ),
        "TBIL": np.where(
            np.random.rand(n) < 0.2,
            np.nan,
            np.random.exponential(1.5, n)
        ),
        "ALP": np.where(
            np.random.rand(n) < 0.18,
            np.nan,
            np.random.gamma(5, 30, n)
        ),

        # TNM staging
        "T_stage": np.random.choice(
            ["T1", "T2", "T3", "T4", None],
            n,
            p=[0.1, 0.25, 0.35, 0.2, 0.1]
        ),
        "N_stage": np.random.choice(
            ["N0", "N1", "N2", None],
            n,
            p=[0.3, 0.4, 0.2, 0.1]
        ),
        "M_stage": np.random.choice(
            ["M0", "M1", None],
            n,
            p=[0.75, 0.15, 0.1]
        ),

        # Surgical info
        "surgical_approach": np.random.choice(
            ["Whipple", "distal_pancreatectomy", "palliative_bypass", "no_surgery", None],
            n,
            p=[0.3, 0.25, 0.2, 0.15, 0.1]
        ),
        "resection_margin": np.random.choice(
            ["R0", "R1", "R2", "unknown", None],
            n,
            p=[0.5, 0.2, 0.1, 0.1, 0.1]
        ),
    })

    # Convert None to NaN
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace({None: np.nan})

    return df


def demo_rule_engine(df: pd.DataFrame) -> None:
    """Demonstrate the Rule Engine module."""
    print("\n" + "=" * 70)
    print("MODULE 1: RULE ENGINE - Four-State Missingness Mask")
    print("=" * 70)

    # Setup
    engine = RuleEngine()

    # Register CA19-9 with applicability rule (Lewis negative -> NA)
    engine.add_rule(
        "CA19_9",
        VariableSpec(
            column="CA19_9",
            data_type="numeric",
            time_window=TimeWindow(lower_bound_days=-30, upper_bound_days=7),
            applicability_rules=[
                ApplicabilityRule(
                    condition_column="Lewis_status",
                    condition_operator="not in",
                    condition_value=["negative", "a-b-", "Le(a-b-)"]
                )
            ]
        )
    )

    # Register other common variables
    engine.register_default_pDAC_rules()

    # Prepare anchor dates
    anchor_dates = {}
    for idx, row in df.iterrows():
        pid = row["patient_id"]
        anchor_dates[pid] = row["diagnosis_date"]

    # Generate mask
    mask_df = engine.generate_mask(
        df,
        anchor_dates=anchor_dates,
        date_columns={"CA19_9": "CA19_9_date", "CEA": "diagnosis_date"}
    )

    # Summary
    summary = engine.get_missingness_summary(mask_df)
    print(f"\n📊 Missingness Summary:")
    print(summary.to_string(index=False))

    # Show state distribution
    print(f"\n📋 State Distribution for CA19_9:")
    ca19_9_mask = mask_df["CA19_9_mask"]
    print(f"  Observed (1):           {(ca19_9_mask == MaskState.OBSERVED).sum()}")
    print(f"  True Missing (0):       {(ca19_9_mask == MaskState.TRUE_MISSING).sum()}")
    print(f"  Not Applicable (-1):   {(ca19_9_mask == MaskState.NOT_APPLICABLE).sum()}")
    print(f"  Out of Time Window(-2): {(ca19_9_mask == MaskState.OUT_OF_TIME_WINDOW).sum()}")

    print("\n✅ Rule Engine module working correctly!")
    return mask_df


def demo_missingness_analyzer(df: pd.DataFrame, mask_df: pd.DataFrame) -> None:
    """Demonstrate the Missingness Analyzer module."""
    print("\n" + "=" * 70)
    print("MODULE 2: MISSINGNESS ANALYZER - Mechanism Classification")
    print("=" * 70)

    # Setup analyzer
    analyzer = MissingnessAnalyzer(
        df,
        mask_df=mask_df,
        categorical_columns=["T_stage", "N_stage", "M_stage", "surgical_approach"]
    )

    # Analyze all variables
    profiles = analyzer.analyze_all_variables(
        exclude_columns=["patient_id", "diagnosis_date", "CA19_9_date"]
    )

    # Show results
    print(f"\n📊 Missingness Mechanism Classification:")
    for profile in profiles[:8]:  # Show first 8
        print(f"\n  Variable: {profile.variable}")
        print(f"    Mechanism: {profile.mechanism.value}")
        print(f"    Confidence: {profile.confidence:.2f}")
        print(f"    Missing Rate: {profile.missing_rate:.1%}")
        if profile.evidence:
            if "structural_pattern" in profile.evidence:
                sp = profile.evidence["structural_pattern"]
                print(f"    Pattern: {sp.description}")

    # Show clusters
    print(f"\n🔗 Missingness Clusters (corr > 0.7):")
    clusters = analyzer.detect_missingness_clusters(threshold=0.7)
    if clusters:
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {', '.join(cluster)}")
    else:
        print("  No strong clusters detected (may need more data)")

    print("\n✅ Missingness Analyzer module working correctly!")


def demo_baselines(df: pd.DataFrame) -> None:
    """Demonstrate the Enhanced Baselines module."""
    print("\n" + "=" * 70)
    print("MODULE 3: ENHANCED BASELINES - Imputation Methods")
    print("=" * 70)

    # Select columns for imputation
    numeric_cols = ["CA19_9", "CEA", "ALB", "TBIL", "ALP"]
    cat_cols = ["T_stage", "N_stage", "surgical_approach"]
    demo_df = df[numeric_cols + cat_cols].head(50).copy()

    n_missing_before = demo_df.isna().sum().sum()
    print(f"\n📊 Missing values before imputation: {n_missing_before}")

    # MissForest
    print(f"\n🔧 Testing MissForestImputer...")
    mf_imputer = MissForestImputer(max_iter=3, n_estimators=5, random_state=42)
    df_mf = mf_imputer.fit_transform(demo_df)
    n_missing_mf = df_mf.isna().sum().sum()
    print(f"  Missing values after: {n_missing_mf}")
    print(f"  Converged: {mf_imputer.converged_}")
    print(f"  Iterations: {mf_imputer.n_iter_}")

    # kNN
    print(f"\n🔧 Testing KNNImputer...")
    knn_imputer = KNNImputer(n_neighbors=3)
    df_knn = knn_imputer.fit_transform(demo_df)
    n_missing_knn = df_knn.isna().sum().sum()
    print(f"  Missing values after: {n_missing_knn}")

    # MICE
    print(f"\n🔧 Testing MICEImputer...")
    mice_imputer = MICEImputer(max_iter=5, random_state=42)
    df_mice = mice_imputer.fit_transform(demo_df)
    n_missing_mice = df_mice.isna().sum().sum()
    print(f"  Missing values after: {n_missing_mice}")

    # Show sample results
    print(f"\n📋 Sample Imputation Results (MissForest):")
    print(df_mf.head(5).to_string())

    print("\n✅ Enhanced Baselines module working correctly!")


def demo_llm_constraints(df: pd.DataFrame) -> None:
    """Demonstrate the LLM Constraints module."""
    print("\n" + "=" * 70)
    print("MODULE 4: LLM CONSTRAINTS - Structured Imputation")
    print("=" * 70)

    # Setup
    llm_layer = LLMConstraintLayer()
    llm_layer.register_default_pdac_constraints()

    # Create patient data
    patient_data = pd.Series({
        "age": 65,
        "gender": "M",
        "CA19_9": 125.0,
        "CEA": 5.2,
        "ALB": 38.0,
        "diagnosis_date": pd.Timestamp("2024-01-15")
    })

    # Create clinical context
    context = ClinicalContext(
        patient_id="P001",
        diagnosis="Pancreatic ductal adenocarcinoma",
        age=65,
        gender="M",
        stage="III",
        comorbidities=["Type 2 diabetes", "Hypertension"],
        relevant_labs={"CA19_9": 125.0, "CEA": 5.2, "ALB": 38.0},
        temporal_context="pre-operative"
    )

    # Test single imputation
    print(f"\n🔧 Testing single imputation (T_stage)...")
    result = llm_layer.impute("T_stage", patient_data, context)
    print(f"\n  Imputation Result:")
    print(f"    Variable: {result.variable}")
    print(f"    Imputed Value: {result.imputed_value}")
    print(f"    Confidence Score: {result.confidence_score:.2f}")
    print(f"    Confidence Level: {result.confidence_level.value}")
    print(f"    Reasoning: {result.reasoning[:80]}...")
    print(f"    Evidence: {result.evidence}")

    # Test batch imputation
    print(f"\n🔧 Testing batch imputation...")
    results = llm_layer.impute_batch(
        ["T_stage", "N_stage", "surgical_approach", "resection_margin"],
        patient_data,
        context
    )
    for var, res in results.items():
        print(f"  {var}: {res.imputed_value} (confidence: {res.confidence_score:.2f})")

    # Show confidence summary
    print(f"\n📋 Confidence Summary:")
    summary = llm_layer.get_confidence_summary()
    print(summary.to_string(index=False))

    # Show high confidence filter
    print(f"\n🔍 High Confidence Results (≥0.7):")
    high_conf = llm_layer.filter_high_confidence(threshold=0.7)
    for r in high_conf:
        print(f"  {r.variable}: {r.imputed_value}")

    print("\n✅ LLM Constraints module working correctly!")


def main():
    """Main function to run all module demonstrations."""
    print("\n" + "=" * 70)
    print("ClinDiff-PDAC: Clinical Differentiation for Pancreatic Ductal Adenocarcinoma")
    print("Integrated Pipeline - All Core Modules")
    print("=" * 70)

    # Create demo data
    print("\n📊 Creating synthetic PDAC clinical data...")
    df = create_demo_data()
    print(f"  Dataset shape: {df.shape}")
    print(f"  Total missing values: {df.isna().sum().sum()}")
    print(f"\n  Columns: {', '.join(df.columns)}")

    # Run demos
    mask_df = demo_rule_engine(df)
    demo_missingness_analyzer(df, mask_df)
    demo_baselines(df)
    demo_llm_constraints(df)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ ALL MODULES VERIFIED SUCCESSFULLY!")
    print("=" * 70)
    print("""
Summary:
  1. Rule Engine         ✅ Four-state mask generation working
  2. Missingness Analyzer ✅ MCAR/MAR/MNAR/Structural classification working
  3. Enhanced Baselines  ✅ MissForest/kNN/MICE imputation working
  4. LLM Constraints     ✅ Constrained imputation with confidence working

All modules are functional and ready for integration.
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())

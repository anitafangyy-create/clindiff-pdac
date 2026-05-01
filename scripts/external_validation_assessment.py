#!/usr/bin/env python3
"""
Assess whether an external cohort can support true imputation validation.

This script separates:
1. true external imputation validation on shared laboratory features; and
2. broader external utility such as clinical or molecular contextualization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from minimal_experiment import repeated_masking_evaluation  # noqa: E402


CANONICAL_ALIASES: Dict[str, str] = {
    "age_num": "age",
    "sex": "gender",
    "survival_months_num": "survival_months",
    "stage_clean": "stage_at_diagnosis",
    "AGE": "age",
    "SEX": "gender",
    "OS_MONTHS": "survival_months",
    "AJCC_PATHOLOGIC_TUMOR_STAGE": "pathological_stage",
    "AJCC_TUMOR_PATHOLOGIC_PT": "pathological_t_stage",
    "AJCC_NODES_PATHOLOGIC_PN": "pathological_n_stage",
    "AJCC_METASTASIS_PATHOLOGIC_PM": "pathological_m_stage",
    "DIABETES_DIAGNOSIS_INDICATOR": "has_diabetes",
    "HISTORY_CHRONIC_PANCREATITIS": "has_chronic_pancreatitis",
    "ALCOHOL_HISTORY_DOCUMENTED": "is_drinker",
    "SMOKING_PACK_YEARS": "smoking_duration",
}

OUTCOME_COLUMNS = {
    "survival_days",
    "survival_months",
}

HEADLINE_TARGET_GROUP = {
    "tb_before_treatment",
    "db_before_treatment",
    "ggt_before_treatment",
}

STAGE_COLUMNS = {
    "stage_at_diagnosis",
    "clinical_stage",
    "pathological_stage",
    "pathological_t_stage",
    "pathological_n_stage",
    "pathological_m_stage",
}

CLINICAL_HISTORY_COLUMNS = {
    "has_diabetes",
    "has_chronic_pancreatitis",
    "is_drinker",
    "smoking_duration",
}


@dataclass
class ValidationDecision:
    decision: str
    rationale: str
    support_level: str
    support_rationale: str
    shared_columns: List[str]
    shared_numeric_columns: List[str]
    shared_categorical_columns: List[str]
    shared_predictor_columns: List[str]
    shared_headline_targets: List[str]
    missing_headline_targets: List[str]
    has_stage_overlap: bool
    has_outcome_overlap: bool
    has_clinical_history_overlap: bool
    has_molecular_context: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal",
        default=os.path.join(ROOT, "data", "raw", "pancreatic_cancer_data_normalized_clean.csv"),
        help="Path to the internal development cohort CSV.",
    )
    parser.add_argument(
        "--external",
        default=os.path.join(ROOT, "data", "processed", "seer_pancreatic_cancer.csv"),
        help="Path to the external cohort CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "outputs", "external_validation"),
        help="Directory for generated reports.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Repeated masking runs for compatible external cohorts.",
    )
    parser.add_argument(
        "--missing-rates",
        nargs="+",
        type=float,
        default=[0.2, 0.4],
        help="Artificial missingness rates for external repeated masking.",
    )
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def canonicalize_external(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: CANONICAL_ALIASES[col] for col in df.columns if col in CANONICAL_ALIASES}
    return df.rename(columns=rename_map).copy()


def shared_columns(internal_df: pd.DataFrame, external_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    shared = sorted(set(internal_df.columns) & set(external_df.columns))
    shared_numeric = [
        col for col in shared
        if is_numeric_dtype(internal_df[col]) and is_numeric_dtype(external_df[col])
    ]
    shared_categorical = [col for col in shared if col not in shared_numeric]
    return shared, shared_numeric, shared_categorical


def predictor_columns(shared_numeric: List[str]) -> List[str]:
    return sorted(col for col in shared_numeric if col not in OUTCOME_COLUMNS)


def detect_molecular_context(external_path: str) -> bool:
    base_dir = os.path.dirname(external_path)
    molecular_markers = [
        "expression",
        "mutation",
        "mutations",
    ]
    try:
        names = os.listdir(base_dir)
    except OSError:
        return False
    lowered = [name.lower() for name in names]
    return any(any(marker in name for marker in molecular_markers) for name in lowered)


def decide_validation_mode(
    shared: List[str],
    shared_numeric: List[str],
    shared_categorical: List[str],
    external_path: str,
) -> ValidationDecision:
    predictors = predictor_columns(shared_numeric)
    shared_targets = sorted(set(shared_numeric) & HEADLINE_TARGET_GROUP)
    missing_targets = sorted(HEADLINE_TARGET_GROUP - set(shared_targets))
    has_stage_overlap = any(col in shared for col in STAGE_COLUMNS)
    has_outcome_overlap = any(col in shared for col in OUTCOME_COLUMNS)
    has_clinical_history_overlap = any(col in shared for col in CLINICAL_HISTORY_COLUMNS)
    has_molecular_context = detect_molecular_context(external_path)

    support_level = "epidemiologic_contextualization"
    support_rationale = (
        "The cohort is mainly suitable for disease-spectrum or demographic context, "
        "not for structured clinical validation of the imputation task."
    )

    if has_stage_overlap and has_outcome_overlap and has_clinical_history_overlap:
        support_level = "secondary_clinical_context"
        support_rationale = (
            "The cohort overlaps with several structured clinical variables, so it can "
            "support a secondary external clinical context analysis beyond demographics."
        )

    if support_level == "secondary_clinical_context" and has_molecular_context:
        support_level = "clinical_molecular_context"
        support_rationale = (
            "The cohort provides structured clinical overlap plus molecular assays, so it "
            "is useful as an external clinical-molecular context cohort even though it "
            "does not validate the headline imputation target group."
        )

    if len(shared_targets) >= 2 and len(predictors) >= 4:
        return ValidationDecision(
            decision="full_external_imputation_validation",
            rationale=(
                "External cohort overlaps with the headline liver biomarker group and "
                "has enough shared numeric predictors to run repeated masking."
            ),
            support_level="external_imputation_validation",
            support_rationale=(
                "This cohort supports the main ClinDiff-Lite claim and can be used for "
                "formal external imputation benchmarking."
            ),
            shared_columns=shared,
            shared_numeric_columns=shared_numeric,
            shared_categorical_columns=shared_categorical,
            shared_predictor_columns=predictors,
            shared_headline_targets=shared_targets,
            missing_headline_targets=missing_targets,
            has_stage_overlap=has_stage_overlap,
            has_outcome_overlap=has_outcome_overlap,
            has_clinical_history_overlap=has_clinical_history_overlap,
            has_molecular_context=has_molecular_context,
        )

    if len(predictors) >= 4:
        return ValidationDecision(
            decision="limited_non_headline_validation",
            rationale=(
                "External cohort has several shared numeric variables, but it does not "
                "cover the headline liver biomarker group well enough to validate the "
                "main ClinDiff-Lite claim."
            ),
            support_level=support_level,
            support_rationale=support_rationale,
            shared_columns=shared,
            shared_numeric_columns=shared_numeric,
            shared_categorical_columns=shared_categorical,
            shared_predictor_columns=predictors,
            shared_headline_targets=shared_targets,
            missing_headline_targets=missing_targets,
            has_stage_overlap=has_stage_overlap,
            has_outcome_overlap=has_outcome_overlap,
            has_clinical_history_overlap=has_clinical_history_overlap,
            has_molecular_context=has_molecular_context,
        )

    return ValidationDecision(
        decision="contextualization_only",
        rationale=(
            "External cohort lacks enough shared numeric clinical variables, especially "
            "the liver biomarker group, so it can support only cohort contextualization."
        ),
        support_level=support_level,
        support_rationale=support_rationale,
        shared_columns=shared,
        shared_numeric_columns=shared_numeric,
        shared_categorical_columns=shared_categorical,
        shared_predictor_columns=predictors,
        shared_headline_targets=shared_targets,
        missing_headline_targets=missing_targets,
        has_stage_overlap=has_stage_overlap,
        has_outcome_overlap=has_outcome_overlap,
        has_clinical_history_overlap=has_clinical_history_overlap,
        has_molecular_context=has_molecular_context,
    )


def prepare_external_numeric_frame(external_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    df = external_df[features].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return df


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_markdown(
    path: str,
    internal_path: str,
    external_path: str,
    decision: ValidationDecision,
    n_internal: int,
    n_external: int,
) -> None:
    lines = [
        "# External Validation Assessment",
        "",
        f"- Internal cohort: `{internal_path}` (n={n_internal})",
        f"- External cohort: `{external_path}` (n={n_external})",
        f"- Decision: `{decision.decision}`",
        f"- Rationale: {decision.rationale}",
        f"- Secondary support level: `{decision.support_level}`",
        f"- Secondary support rationale: {decision.support_rationale}",
        "",
        "## Shared Feature Summary",
        "",
        f"- Shared columns: {', '.join(decision.shared_columns) or 'None'}",
        f"- Shared numeric columns: {', '.join(decision.shared_numeric_columns) or 'None'}",
        f"- Shared categorical columns: {', '.join(decision.shared_categorical_columns) or 'None'}",
        f"- Shared predictor columns: {', '.join(decision.shared_predictor_columns) or 'None'}",
        f"- Shared headline liver targets: {', '.join(decision.shared_headline_targets) or 'None'}",
        f"- Missing headline liver targets: {', '.join(decision.missing_headline_targets) or 'None'}",
        f"- Stage overlap: {'Yes' if decision.has_stage_overlap else 'No'}",
        f"- Outcome overlap: {'Yes' if decision.has_outcome_overlap else 'No'}",
        f"- Clinical history overlap: {'Yes' if decision.has_clinical_history_overlap else 'No'}",
        f"- Molecular context available: {'Yes' if decision.has_molecular_context else 'No'}",
        "",
        "## Interpretation",
        "",
    ]

    if decision.decision == "full_external_imputation_validation":
        lines.extend([
            "This external cohort is compatible with the headline ClinDiff-Lite use case.",
            "You can treat repeated masking on the shared features as a genuine external imputation validation.",
        ])
    elif decision.decision == "limited_non_headline_validation":
        lines.extend([
            "This cohort supports only a secondary validation on non-headline shared features.",
            "It should not be used to claim external validation of the liver biomarker-group hypothesis.",
        ])
    else:
        if decision.support_level == "clinical_molecular_context":
            lines.extend([
                "This cohort should be described as external clinical-molecular context.",
                "It is useful for supplementary clinical and molecular interpretation, but not for external imputation accuracy validation.",
            ])
        elif decision.support_level == "secondary_clinical_context":
            lines.extend([
                "This cohort should be described as secondary clinical context.",
                "It is stronger than a demographics-only reference set, but it still cannot validate imputation accuracy for the liver biomarker task.",
            ])
        else:
            lines.extend([
                "This cohort should be described as contextualization only.",
                "It is not appropriate to claim external imputation accuracy validation from this dataset.",
            ])

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    internal_df = load_csv(args.internal)
    external_raw_df = load_csv(args.external)
    external_df = canonicalize_external(external_raw_df)

    shared, shared_numeric, shared_categorical = shared_columns(internal_df, external_df)
    decision = decide_validation_mode(shared, shared_numeric, shared_categorical, args.external)

    payload = {
        "internal_path": args.internal,
        "external_path": args.external,
        "internal_rows": int(len(internal_df)),
        "external_rows": int(len(external_df)),
        **asdict(decision),
    }

    json_path = os.path.join(args.output_dir, "compatibility_report.json")
    md_path = os.path.join(args.output_dir, "compatibility_report.md")
    write_json(json_path, payload)
    write_markdown(
        md_path,
        args.internal,
        args.external,
        decision,
        len(internal_df),
        len(external_df),
    )

    print(f"Decision: {decision.decision}")
    print(f"Rationale: {decision.rationale}")
    print(f"Support level: {decision.support_level}")
    print(f"Support rationale: {decision.support_rationale}")
    print(f"Shared numeric columns: {decision.shared_numeric_columns}")
    print(f"Shared categorical columns: {decision.shared_categorical_columns}")
    print(f"Shared headline targets: {decision.shared_headline_targets}")

    if decision.decision == "contextualization_only":
        print("Skipped repeated masking because the external cohort is not compatible.")
        return

    feature_set = decision.shared_predictor_columns
    external_numeric = prepare_external_numeric_frame(external_df, feature_set)

    if len(external_numeric.columns) < 4 or len(external_numeric) < 50:
        print("Shared external frame is too small for a stable repeated-masking benchmark.")
        return

    runs_df, summary_df = repeated_masking_evaluation(
        external_numeric,
        rates=args.missing_rates,
        repeats=args.repeats,
    )

    runs_path = os.path.join(args.output_dir, "repeated_runs.csv")
    summary_path = os.path.join(args.output_dir, "repeated_summary.csv")
    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved repeated masking runs to: {runs_path}")
    print(f"Saved repeated masking summary to: {summary_path}")


if __name__ == "__main__":
    main()

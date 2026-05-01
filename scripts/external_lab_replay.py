#!/usr/bin/env python3
"""
Scan local datasets for external laboratory compatibility and generate replay-ready reports.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from scripts.external_validation_assessment import (  # noqa: E402
    canonicalize_external,
    decide_validation_mode,
    load_csv,
    shared_columns,
)


DATASETS: List[Dict[str, str]] = [
    {
        "name": "SEER",
        "path": os.path.join(ROOT, "data", "processed", "seer_pancreatic_cancer.csv"),
        "output_dir": os.path.join(ROOT, "outputs", "external_validation", "seer"),
    },
    {
        "name": "TCGA-PAAD",
        "path": os.path.join(ROOT, "data", "TCGA_PAAD_cbio", "tcga_patient_clinical.csv"),
        "output_dir": os.path.join(ROOT, "outputs", "external_validation", "tcga"),
    },
]

HEADLINE_LABS = {
    "tb_before_treatment",
    "db_before_treatment",
    "ggt_before_treatment",
}


def main() -> None:
    internal_path = os.path.join(ROOT, "data", "raw", "pancreatic_cancer_data_normalized_clean.csv")
    internal_df = load_csv(internal_path)
    records = []

    for dataset in DATASETS:
        if not os.path.exists(dataset["path"]):
            continue
        external_raw = load_csv(dataset["path"])
        external_df = canonicalize_external(external_raw)
        shared, shared_numeric, shared_categorical = shared_columns(internal_df, external_df)
        decision = decide_validation_mode(shared, shared_numeric, shared_categorical, dataset["path"])

        record = {
            "dataset": dataset["name"],
            "path": dataset["path"],
            "decision": decision.decision,
            "support_level": decision.support_level,
            "shared_numeric_count": len(decision.shared_numeric_columns),
            "shared_headline_target_count": len(decision.shared_headline_targets),
            "shared_headline_targets": ", ".join(decision.shared_headline_targets) or "None",
            "missing_headline_targets": ", ".join(decision.missing_headline_targets) or "None",
            "is_replay_ready": decision.decision == "full_external_imputation_validation",
            "recommended_next_step": (
                "Run no-retuning repeated masking replay"
                if decision.decision == "full_external_imputation_validation"
                else "Use as context only and continue searching compatible lab cohort"
            ),
        }
        records.append(record)

    out_dir = os.path.join(ROOT, "outputs", "external_validation")
    os.makedirs(out_dir, exist_ok=True)
    summary_df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "external_lab_replay_scan.csv")
    summary_df.to_csv(csv_path, index=False)

    compatible = summary_df[summary_df["is_replay_ready"]]
    payload = {
        "internal_path": internal_path,
        "headline_labs": sorted(HEADLINE_LABS),
        "compatible_dataset_count": int(len(compatible)),
        "datasets": records,
    }
    json_path = os.path.join(out_dir, "external_lab_replay_scan.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    md_lines = [
        "# External Compatible-Lab Replay Scan",
        "",
        f"- Internal development cohort: `{internal_path}`",
        f"- Headline laboratory targets: {', '.join(sorted(HEADLINE_LABS))}",
        f"- Replay-ready compatible cohorts found: **{len(compatible)}**",
        "",
        "## Dataset-by-dataset decision",
        "",
    ]
    for record in records:
        md_lines.extend([
            f"### {record['dataset']}",
            f"- Path: `{record['path']}`",
            f"- Decision: `{record['decision']}`",
            f"- Support level: `{record['support_level']}`",
            f"- Shared numeric count: {record['shared_numeric_count']}",
            f"- Shared headline targets: {record['shared_headline_targets']}",
            f"- Missing headline targets: {record['missing_headline_targets']}",
            f"- Recommended next step: {record['recommended_next_step']}",
            "",
        ])
    md_path = os.path.join(out_dir, "external_lab_replay_scan.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")

    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()

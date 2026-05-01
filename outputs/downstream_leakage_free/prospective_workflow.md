# Leakage-free prospective downstream workflow

- Source data: `data/raw/pancreatic_cancer_data_normalized_clean.csv`
- Rows: 2419
- Columns: 123

## Prespecified workflow

1. Freeze a train/test split or temporal split before any imputation tuning.
2. Remove all outcome-derived variables from upstream imputation predictors.
3. Fit route selection and grouped refinement thresholds only on the training split.
4. Apply the frozen imputation workflow to held-out patients without retuning.
5. Compare complete-case, mean, ClinDiff-Lite, and ClinDiff-Gated under identical splits.
6. Evaluate downstream endpoints only after the imputation model is frozen.

## Current leakage audit

- Eligible predictors in the current internal dataset: 121
- Outcome-derived columns excluded upstream: survival_days, survival_months

## Interpretation

This file defines a prospective-ready downstream protocol rather than a new headline result.
Its purpose is to prevent survival-derived variables from leaking back into biomarker imputation.

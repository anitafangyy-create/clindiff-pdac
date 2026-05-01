# External Validation Protocol for ClinDiff-Lite

## Goal

This project should separate two very different claims:

1. **True external imputation validation**
2. **External cohort contextualization**

For ClinDiff-Lite, the headline claim is about the liver biomarker group:

- `tb_before_treatment`
- `db_before_treatment`
- `ggt_before_treatment`

An external dataset supports the headline claim only if it contains overlapping laboratory features from that group and enough additional shared numeric predictors to run repeated masking.

## Decision Rule

Use `scripts/external_validation_assessment.py` before writing any manuscript claim.

- `full_external_imputation_validation`
  The external cohort overlaps with at least two liver-group targets and has enough shared numeric predictors. Repeated masking on the external cohort can be reported as true external validation of imputation accuracy.
- `limited_non_headline_validation`
  The external cohort has several shared numeric variables but does not cover the liver-group targets well enough. This can support only a secondary or sensitivity analysis, not the main claim.
- `contextualization_only`
  The external cohort lacks the liver biomarkers or enough shared numeric structure. It may be used for disease-spectrum comparison, demographic comparison, stage distribution, survival context, or discussion of generalizability, but not for imputation-accuracy claims.

## Why SEER Does Not Qualify

In the current repository, SEER overlaps with the internal cohort mainly on broad variables such as age and survival. It does not contain the liver biomarker trio used by the headline ClinDiff-Lite analysis. That means SEER cannot validate whether ClinDiff-Lite improves imputation of the clinically related laboratory block.

SEER is therefore appropriate for:

- population-level cohort comparison
- stage and outcome contextualization
- narrative discussion of external generalizability

SEER is not appropriate for:

- external RMSE/MAE validation of the liver biomarker imputer
- claiming external validation of the main ClinDiff-Lite insertion

## Recommended External Validation Design

To obtain a true external validation dataset, prefer an independent EMR cohort that contains:

- `tb_before_treatment`
- `db_before_treatment`
- `ggt_before_treatment`
- age
- fasting glucose or another clinically adjacent chemistry marker
- consistent pre-treatment timing definitions

Recommended workflow:

1. Map external columns into the internal canonical names.
2. Restrict analysis to shared pre-treatment structured variables.
3. Run repeated masking on the external cohort with the same missingness rates as the internal headline analysis.
4. Compare `Mean`, `Median`, and `ClinDiff-Lite` using paired deltas and confidence intervals.
5. Report this as external imputation validation only if the shared feature set still reflects the liver biomarker group.

## Command

```bash
python3 scripts/external_validation_assessment.py
```

Outputs are written to `outputs/external_validation/`.

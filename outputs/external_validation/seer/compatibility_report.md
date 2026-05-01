# External Validation Assessment

- Internal cohort: `data/raw/pancreatic_cancer_data_normalized_clean.csv` (n=2419)
- External cohort: `data/processed/seer_pancreatic_cancer.csv` (n=124863)
- Decision: `contextualization_only`
- Rationale: External cohort lacks enough shared numeric clinical variables, especially the liver biomarker group, so it can support only cohort contextualization.
- Secondary support level: `epidemiologic_contextualization`
- Secondary support rationale: The cohort is mainly suitable for disease-spectrum or demographic context, not for structured clinical validation of the imputation task.

## Shared Feature Summary

- Shared columns: age, gender, stage_at_diagnosis, survival_months
- Shared numeric columns: age
- Shared categorical columns: gender, stage_at_diagnosis, survival_months
- Shared predictor columns: age
- Shared headline liver targets: None
- Missing headline liver targets: db_before_treatment, ggt_before_treatment, tb_before_treatment
- Stage overlap: Yes
- Outcome overlap: Yes
- Clinical history overlap: No
- Molecular context available: No

## Interpretation

This cohort should be described as contextualization only.
It is not appropriate to claim external imputation accuracy validation from this dataset.

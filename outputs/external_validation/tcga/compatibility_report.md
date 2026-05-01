# External Validation Assessment

- Internal cohort: `data/raw/pancreatic_cancer_data_normalized_clean.csv` (n=2419)
- External cohort: `data/TCGA_PAAD_cbio/tcga_patient_clinical.csv` (n=185)
- Decision: `contextualization_only`
- Rationale: External cohort lacks enough shared numeric clinical variables, especially the liver biomarker group, so it can support only cohort contextualization.
- Secondary support level: `clinical_molecular_context`
- Secondary support rationale: The cohort provides structured clinical overlap plus molecular assays, so it is useful as an external clinical-molecular context cohort even though it does not validate the headline imputation target group.

## Shared Feature Summary

- Shared columns: age, gender, has_chronic_pancreatitis, has_diabetes, is_drinker, pathological_m_stage, pathological_n_stage, pathological_stage, pathological_t_stage, smoking_duration, survival_months
- Shared numeric columns: age, survival_months
- Shared categorical columns: gender, has_chronic_pancreatitis, has_diabetes, is_drinker, pathological_m_stage, pathological_n_stage, pathological_stage, pathological_t_stage, smoking_duration
- Shared predictor columns: age
- Shared headline liver targets: None
- Missing headline liver targets: db_before_treatment, ggt_before_treatment, tb_before_treatment
- Stage overlap: Yes
- Outcome overlap: Yes
- Clinical history overlap: Yes
- Molecular context available: Yes

## Interpretation

This cohort should be described as external clinical-molecular context.
It is useful for supplementary clinical and molecular interpretation, but not for external imputation accuracy validation.

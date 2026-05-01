# External Compatible-Lab Replay Scan

- Internal development cohort: `data/raw/pancreatic_cancer_data_normalized_clean.csv`
- Headline laboratory targets: db_before_treatment, ggt_before_treatment, tb_before_treatment
- Replay-ready compatible cohorts found: **0**

## Dataset-by-dataset decision

### SEER
- Path: `data/processed/seer_pancreatic_cancer.csv`
- Decision: `contextualization_only`
- Support level: `epidemiologic_contextualization`
- Shared numeric count: 1
- Shared headline targets: None
- Missing headline targets: db_before_treatment, ggt_before_treatment, tb_before_treatment
- Recommended next step: Use as context only and continue searching compatible lab cohort

### TCGA-PAAD
- Path: `data/TCGA_PAAD_cbio/tcga_patient_clinical.csv`
- Decision: `contextualization_only`
- Support level: `clinical_molecular_context`
- Shared numeric count: 2
- Shared headline targets: None
- Missing headline targets: db_before_treatment, ggt_before_treatment, tb_before_treatment
- Recommended next step: Use as context only and continue searching compatible lab cohort


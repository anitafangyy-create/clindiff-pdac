# ClinDiff-PDAC Data Export

Export date: 2026-04-27

This package is a standalone copy of the repository `data/` directory for the ClinDiff-PDAC project.

Included content:

- `raw/`
  - Internal project raw cohort table
- `processed/`
  - Processed external cohort tables such as SEER
- `TCGA_PAAD_cbio/`
  - Existing TCGA-PAAD derivatives and local parsing report
- `public_omics/`
  - Public omics manifests, download logs, and GDC-downloaded TCGA-PAAD open-access files

Notes:

- This export is a copy, not a move. The original files remain in the project repository.
- `public_omics/ucsc_xena/TCGA-PAAD.GDC_phenotype.tsv.gz` is retained as an earlier failed direct-link fetch artifact and may contain an access-denied response rather than usable phenotype data.

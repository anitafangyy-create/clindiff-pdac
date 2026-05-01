# Public Similar Dataset Shortlist

Updated: 2026-04-27

This note records additional public or partly public pancreatic cancer / PDAC datasets that are similar to the current ClinDiff-PDAC project.

## Highest-value candidates

### 1. CPTAC-PDA

- Source: TCIA + CPTAC/PDC ecosystem
- Link: https://www.cancerimagingarchive.net/collection/cptac-pda/
- What it contains:
  - PDAC imaging collection linked to proteogenomic resources
  - MR, CT, US, PT, histopathology, whole-slide images
- Public availability:
  - Public imaging download
- Why it matters:
  - Best public candidate if the project expands toward imaging + omics integration
  - More clinically linked than generic pancreas imaging sets
- Current portal snapshot:
  - Updated `2025-02-26`
  - `168` subjects
  - about `155.24 GB`

### 2. ICGC legacy pancreatic projects

- Source: ICGC ARGO legacy access documentation
- Link: https://docs.icgc-argo.org/docs/data-access/icgc-25k-data
- What it contains:
  - Legacy ICGC projects with open and controlled molecular data
  - Open-access release bucket plus partner-repository mapping
- Public availability:
  - Some open-access release data are public
  - Controlled data still require approval
- Why it matters:
  - Best route for recovering historical pancreatic cohorts such as `PACA-AU` / `PACA-CA`
  - More relevant than TARGET for adult PDAC
- Important status note:
  - Legacy ICGC Data Portal retired in `June 2024`

### 3. TCGA-PAAD in GDC

- Source: GDC official TCGA-PAAD publication/resources pages
- Links:
  - https://gdc.cancer.gov/about-data/publications/integrated-genomic-characterization-pancreatic-ductal-adenocarcinoma
  - https://gdc.cancer.gov/resources-tcga-users/tcga-study-abbreviations
- What it contains:
  - Multi-omics PDAC cohort with clinical, mutation, RNA, methylation, CNV, imaging links, and more
- Public availability:
  - Broad open-access subset plus controlled files
- Why it matters:
  - Still the main public molecular reference cohort for PDAC
  - Already partly integrated into this repository

## Strong open transcriptomic / single-cell candidates

### 4. GEO GSE28735

- Source: NCBI GEO DataSets
- Link: https://www.ncbi.nlm.nih.gov/gds/4336
- What it contains:
  - `45` matched pairs of PDAC tumor and adjacent non-tumor tissue
  - expression profiling array data
- Why it matters:
  - Useful for external transcriptomic comparison with paired design

### 5. GEO GSE242230

- Source: NCBI GEO
- Link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242230
- What it contains:
  - treatment-naive PDAC EUS-FNB single-cell RNA-seq cohort
  - `n=25` biopsy samples
- Why it matters:
  - Strong candidate for microenvironment and cell-state external context

### 6. GEO GSE208536

- Source: NCBI GEO
- Link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE208536
- What it contains:
  - digital spatial profiling across PDAC, acinar-ductal metaplasia, and adjacent normal tissue
  - `8` patient sample sets
- Why it matters:
  - Good fit if the project expands from tabular imputation toward spatial context

### 7. OMIX002487

- Source: OMIX / NGDC
- Link: https://ngdc.cncb.ac.cn/omix/release/OMIX002487
- What it contains:
  - single-cell sequencing of pancreatic cancer with liver metastases
  - `18` paired samples from `6` patients across primary tumor, portal blood, and liver metastasis
- Public availability:
  - Open-access
  - release date `2025-02-27`
- Why it matters:
  - One of the more clinically interesting open PDAC metastatic single-cell resources

### 8. GEO GSE226762

- Source: NCBI GEO
- Link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226762
- What it contains:
  - tumor vs adjacent tissue RNA/miRNA profiling
  - plasma miRNA component
  - one PDAC single-cell component
- Why it matters:
  - More biologically and biomarker-oriented than TCGA alone

## Imaging-specific complementary datasets

### 9. TCIA Pancreas-CT

- Source: TCIA
- Link: https://www.cancerimagingarchive.net/collection/pancreas-ct/
- What it contains:
  - abdominal contrast-enhanced CT
  - `82` subjects
  - manual segmentations
- Public availability:
  - Public
- Why it matters:
  - Useful mainly as pancreas imaging control/reference data
  - Not a PDAC omics cohort, but strong if imaging preprocessing becomes relevant

### 10. CPTAC-PDA tumor annotations

- Source: TCIA analysis result
- Link: https://www.cancerimagingarchive.net/analysis-result/cptac-pda-tumor-annotations/
- What it contains:
  - annotation set derived from CPTAC-PDA
  - `103` subjects
- Why it matters:
  - Directly supports radiology / pathology AI workflows

## Controlled or partly controlled clinical-trial style resources

### 11. EGA phs003615

- Source: EGA
- Link: https://www.ega-archive.org/studies/phs003615
- What it contains:
  - advanced PDAC immunotherapy trial context
  - RNA-seq plus other molecular assays
  - clinical and demographic information referenced through dbGaP/GEO linkage
- Why it matters:
  - Useful if the scope expands into treatment response

### 12. EGA phs000516

- Source: EGA
- Link: https://ega-archive.org/studies/phs000516
- What it contains:
  - Baylor PDAC sequencing cohort
  - `24` PDAC patients
- Why it matters:
  - Historically important and linked to ICGC-style sequencing efforts

## Lower-priority for the current headline task

- TARGET:
  - Pediatric focus, not a natural match for adult PDAC
- Generic pancreas imaging without cancer biomarkers:
  - Useful for anatomy or segmentation work, not for the current lab-imputation headline
- Pure organoid or mechanistic omics without clinical overlap:
  - Useful for biological discussion, but weaker for external validation framing

## Practical recommendation for this project

Priority order for follow-up:

1. `CPTAC-PDA`
2. `ICGC legacy pancreatic cohorts`
3. `GEO GSE242230 / GSE208536 / GSE28735`
4. `OMIX002487`
5. `EGA` controlled-access PDAC trial cohorts

For the current ClinDiff-PDAC headline question, the most promising external public resources are those that add either:

- better PDAC clinical-molecular context than SEER, or
- closer overlap to liver-metastasis / treatment-response biology, or
- richer linked modalities such as imaging or proteomics

But none of the sources above should be assumed to provide a directly compatible replacement for the internal `TB / DB / GGT` laboratory imputation target without case-by-case feature inspection.

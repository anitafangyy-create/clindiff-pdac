# Public Omics Dataset Status

Updated: 2026-04-27

This folder records what public cancer omics resources are already present in this repository, what was newly fetched during this session, and which listed platforms are currently usable for this PDAC project.

## Already present in this repository

- `data/TCGA_PAAD_cbio/`
  - `tcga_patient_clinical.csv`: 185 patient records (`186` lines including header)
  - `tcga_sample_clinical.csv`: 186 sample records (`187` lines including header)
  - `tcga_mutations_all.csv`: `22410` lines
  - `tcga_expression_wide.csv`: `180` lines
  - Source note inside local report: `data/TCGA_PAAD_cbio/TCGA_PAAD数据解析报告.txt`
- `data/processed/seer_pancreatic_cancer.csv`
  - `124864` lines
- `data/raw/pancreatic_cancer_data_normalized_clean.csv`
  - Internal cohort used by this project

## Newly fetched in this session

- `cbioportal/paad_tcga_pan_can_atlas_2018_study.json`
  - From the public cBioPortal API
  - Confirms the study `paad_tcga_pan_can_atlas_2018`
  - Reported study name: `Pancreatic Adenocarcinoma (TCGA, PanCancer Atlas)`
- `ucsc_xena/tcga_paad_gdc_files.json`
  - Pulled from the public GDC API as a stable manifest for `TCGA-PAAD`
  - Contains `2000` file records in this export page
  - Major categories seen in the manifest:
    - `Simple Nucleotide Variation`: 625
    - `Copy Number Variation`: 405
    - `Sequencing Reads`: 244
    - `Biospecimen`: 176
    - `Somatic Structural Variation`: 127
    - `Structural Variation`: 125
    - `Transcriptome Profiling`: 116
    - `DNA Methylation`: 100
    - `Clinical`: 58
    - `Proteome Profiling`: 24
- `ucsc_xena/tcga_paad_gdc_files_detailed.json`
  - Same cohort-level manifest idea, but includes `file_size` and `cases.submitter_id`
- `ucsc_xena/tcga_paad_open_priority_manifest.csv`
  - Local ranked manifest for open-access files
  - Contains `786` open-access rows across all categories

## Download helper added

- `scripts/download_gdc_open_files.py`
  - Downloads public GDC files by `file_id`
  - Accepts either the JSON manifest or the local CSV manifest
  - Supports explicit `--file-id` selection and CSV `--priority` selection

## Public files downloaded into this repository

Downloaded to:

- `data/public_omics/gdc_downloads/tcga_paad_selected/`

Files downloaded successfully:

- `nationwidechildrens.org_clinical_omf_v4.0_paad.txt`
  - Clinical supplement table
  - Size: about `12 KB`
- `nationwidechildrens.org_clinical_drug_paad.txt`
  - Drug/therapy supplement table
  - Size: about `109 KB`
- `10b614d3-7608-48fb-a20a-aad19292598a.rna_seq.augmented_star_gene_counts.tsv`
  - RNA-seq gene counts for one open TCGA-PAAD sample
  - Size: about `4.0 MB`
- `TCGA-IB-7647-01A-21-A39M-20_RPPA_data.tsv`
  - RPPA protein expression table for one open sample
  - Size: about `22 KB`
- `d23a558f-463c-4047-ab15-f3df9e223d8d.methylation_array.sesame.level3betas.txt`
  - Methylation beta table for one open sample
  - Size: about `12 MB`
- `TCGA-PAAD.615b4806-0742-41cf-a334-a7e6482db9bb.gene_level_copy_number.v36.tsv`
  - Gene-level copy number table
  - Size: about `3.3 MB`

Additional high-priority CNV batch:

- `data/public_omics/gdc_downloads/tcga_paad_high_cnv_batch1/`
  - `10` additional open-access `absolute_liftover.gene_level_copy_number.v36.tsv` files
  - Total size: `34328951` bytes (about `34.3 MB`)
  - Intended use: expand sample-level open CNV coverage in small, trackable batches
- `data/public_omics/gdc_downloads/tcga_paad_high_cnv_batch2/`
  - `10` additional open-access `absolute_liftover.gene_level_copy_number.v36.tsv` files
  - Total size: `34362752` bytes (about `34.4 MB`)
  - Intended use: continue the same sample-level CNV expansion without mixing batches

Running total for high-priority open CNV batches:

- Downloaded so far: `20` files
- Combined size so far: `68691703` bytes (about `68.7 MB`)
- Remaining high-priority open CNV files not yet downloaded: `53`

## Platform-by-platform status

### 1. TCGA / cBioPortal

Usable now.

- This repo already contains curated TCGA-PAAD derivatives under `data/TCGA_PAAD_cbio/`.
- The public cBioPortal study metadata was refreshed into:
  - `data/public_omics/cbioportal/paad_tcga_pan_can_atlas_2018_study.json`

### 2. UCSC Xena

Partially usable now.

- A direct attempt to fetch a guessed Xena hub object path returned `AccessDenied`:
  - `data/public_omics/ucsc_xena/TCGA-PAAD.GDC_phenotype.tsv.gz`
- To avoid blocking on brittle hub object naming, the public `TCGA-PAAD` file manifest was fetched from the official GDC API instead:
  - `data/public_omics/ucsc_xena/tcga_paad_gdc_files.json`
- Practical interpretation for this project:
  - Xena remains a valid platform for browsing and standardized analysis.
  - For scripted bulk acquisition, GDC or cBioPortal endpoints are more stable than guessed raw Xena object URLs.

### 3. ICGC

Not directly mirrored here in this session.

- The legacy ICGC Data Portal was retired in June 2024.
- Relevant pancreatic cohorts for this topic are typically discussed as `PACA-AU` and `PACA-CA`.
- For this repository, ICGC should be treated as a candidate source that needs a fresh access route check before bulk download.

### 4. TARGET

Not applicable to this PDAC repository by default.

- TARGET is a pediatric cancer program.
- It is valuable for pediatric oncology, but it is not a standard source for adult pancreatic ductal adenocarcinoma cohort download in this project.

## Recommended use in this repository

- Use `data/TCGA_PAAD_cbio/` for immediate PDAC external clinical-molecular context work.
- Use `data/public_omics/ucsc_xena/tcga_paad_gdc_files.json` when deciding whether to add extra GDC/Xena-backed modalities such as methylation, RPPA, or additional RNA-seq artifacts.
- Use `data/public_omics/ucsc_xena/tcga_paad_open_priority_manifest.csv` plus `scripts/download_gdc_open_files.py` for repeatable follow-up downloads.
- Do not treat `TARGET` as a priority PDAC source unless the project scope explicitly expands beyond adult PDAC.
- Re-check ICGC access workflow before promising automated bulk download.

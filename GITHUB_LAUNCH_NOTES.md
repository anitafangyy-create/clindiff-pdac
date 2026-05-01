# GitHub Launch Notes

Use this file as a lightweight helper when you publish the repository.

## Suggested repository name

- `clindiff-pdac`

## Suggested short description

- `Lightweight missingness-informed imputation for targeted liver biomarker completion in PDAC structured EHR data`

## Suggested topics

- `pdac`
- `ehr`
- `missing-data`
- `imputation`
- `clinical-informatics`
- `healthcare-ai`
- `pancreatic-cancer`
- `medical-ai`

## Suggested first public commit message

```text
Prepare public ClinDiff-PDAC repository for manuscript submission
```

## Suggested release title

```text
v1.0.0-manuscript-submission
```

## Suggested release notes

```text
This release captures the manuscript-aligned public codebase for ClinDiff-PDAC.

Highlights:
- cleaned repository structure for public release
- active ClinDiff-Gated repeated-masking workflow retained
- current figures and outputs kept for the headline analysis
- internal data and local download caches excluded from public upload

Please read README.md, data/README.md, and RELEASE_CHECKLIST.md before reuse.
```

## Final manual checks before publishing

- replace the placeholder GitHub URL in `CITATION.cff`
- confirm author metadata in `CITATION.cff`
- verify `git status` before the first push
- confirm that no internal data files are tracked
- confirm the manuscript-facing result files are the intended final versions

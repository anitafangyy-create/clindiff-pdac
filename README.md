# ClinDiff-PDAC

ClinDiff-PDAC is a lightweight, missingness-informed imputation project for a narrow pancreatic cancer EHR use case: targeted completion of a physiologically linked liver biomarker trio in PDAC structured records.

The current headline workflow is centered on `ClinDiff-Gated`, a small routed imputation pipeline evaluated with repeated MCAR masking. This repository is organized as a code-first public project rather than a full local working-directory dump.

## Current project scope

- Primary focus: targeted imputation of `TB / DB / GGT` in a small structured PDAC cohort
- Headline evidence: repeated MCAR masking with a prespecified liver-trio RMSE endpoint
- Secondary materials: external cohort contextualization and an exploratory downstream survival workflow boundary
- Explicit non-goal: claiming universal EHR imputation superiority across arbitrary variables or datasets

## Repository layout

```text
clindiff-pdac/
├── src/clindiff_pdac/              # Core package modules
│   ├── baselines/                  # Pure-Python tabular imputers
│   ├── engine/                     # Four-state masking logic
│   ├── llm/                        # Optional structured constraint components
│   ├── missingness/                # Missingness profiling utilities
│   └── refinement/                 # ClinDiff-Gated liver-trio refinement
├── scripts/                        # Utility scripts, plotting, diagnostics
├── tests/                          # Lightweight regression tests
├── data/                           # Local working data storage (not for public upload)
├── figures/                        # Figure assets referenced by the paper
├── outputs/                        # Current generated experiment outputs
├── minimal_experiment.py           # Headline repeated-masking experiment
├── optimized_group_experiment.py   # Grouped-refinement benchmark
├── full_experiment.py              # Broader experiment entry point
└── main.py                         # Integrated package demo
```

## What to run first

For the current headline workflow:

```bash
python3 minimal_experiment.py
python3 scripts/plot_minimal_experiment.py
python3 scripts/plot_gated_stability.py
```

For external-cohort guardrails:

```bash
python3 scripts/external_validation_assessment.py
python3 scripts/external_lab_replay.py
```

For the exploratory downstream workflow boundary:

```bash
python3 scripts/downstream_prospective_workflow.py
```

## Active outputs

The main generated artifacts currently used in the project are:

- `outputs/gated_experiment_repeated_summary.csv`
- `outputs/gated_experiment_repeated_runs.csv`
- `outputs/gated_experiment_feature_summary.csv`
- `outputs/gated_experiment_metadata.json`
- `outputs/minimal_experiment_figure.png`
- `outputs/gated_seed_stability.png`
- `outputs/gated_paired_delta_distribution.png`

Only current outputs needed for the active workflow are kept in the main `outputs/` directory.

## Manuscript files

The submission-facing manuscript files are intentionally kept one directory above this repository, alongside the broader paper workspace. This keeps the GitHub repository focused on code, figures, and reproducible outputs.

## Citation

Citation metadata for GitHub and reference managers is provided in `CITATION.cff`.

Before making the repository public, replace the placeholder repository URL in that file with the final GitHub address.

## Public release helpers

For final repository launch preparation, see:

- `RELEASE_CHECKLIST.md`
- `CONTRIBUTING.md`
- `GITHUB_LAUNCH_NOTES.md`

## Installation

The project is Python-based and intentionally lightweight.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

```bash
pytest
```

If you only want the lightweight module checks:

```bash
python3 src/clindiff_pdac/engine/rule_engine.py
python3 src/clindiff_pdac/missingness/missingness_analyzer.py
python3 src/clindiff_pdac/baselines/enhanced_baselines.py
```

## Notes on data and reproducibility

- Raw and processed clinical data are sensitive and are not intended for public GitHub release.
- Public-data download caches are also excluded so the repository stays lightweight.
- Synthetic or limited examples may reproduce workflow structure without reproducing patient-level headline numbers.
- External contextualization outputs should not be interpreted as external imputation-accuracy validation unless a compatible laboratory panel is available.

## License

MIT License

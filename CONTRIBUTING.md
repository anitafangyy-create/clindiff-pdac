# Contributing

Thank you for your interest in ClinDiff-PDAC.

This repository is primarily being prepared as a manuscript-aligned research code release. Contributions are welcome, but please keep the project scope focused on the current public positioning:

- lightweight, missingness-informed imputation
- PDAC structured EHR use cases
- targeted liver-biomarker completion rather than broad platform claims

## Before opening an issue or pull request

Please check:

- `README.md` for the active repository layout
- `RELEASE_CHECKLIST.md` for the public-release boundaries
- `data/README.md` for data-sharing limits

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Recommended validation steps

Before proposing changes to the active workflow, run at least the core checks below:

```bash
pytest
python3 minimal_experiment.py
python3 scripts/plot_minimal_experiment.py
python3 scripts/plot_gated_stability.py
```

If your change touches external-cohort logic or contextualization:

```bash
python3 scripts/external_validation_assessment.py
python3 scripts/external_lab_replay.py
```

## Data policy

Do not include:

- internal EMR tables
- hospital-derived processed patient tables
- local public-data download caches unless they are explicitly intended for redistribution

Keep contributions aligned with the repository's public-data boundary described in `data/README.md`.

## Scope guidance

Good contribution targets:

- bug fixes in the active `ClinDiff-Gated` workflow
- reproducibility improvements
- clearer plotting and reporting scripts
- documentation cleanups
- tests covering active code paths

Less helpful for the public release branch:

- adding large experimental archives back into the repository
- reintroducing deprecated prototype scripts into the root directory
- expanding claims beyond the current manuscript-aligned scope without new evidence

## Style

- Prefer small, traceable changes.
- Keep new dependencies justified and minimal.
- Preserve the distinction between headline analyses and exploratory/supporting workflows.

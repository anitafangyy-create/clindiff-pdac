# Release Checklist

Use this checklist before making the public GitHub repository live or attaching it to a journal submission.

## Repository hygiene

- [ ] `git status` shows only the files intended for public release.
- [ ] No internal clinical datasets are staged.
- [ ] No local cache directories or virtual-environment files are staged.
- [ ] No manuscript PDFs, DOCX files, or local desktop exports are staged unless intentionally public.
- [ ] `README.md` still matches the current repository layout.
- [ ] `outputs/README.md` still matches the current public-facing outputs.

## Data boundary

- [ ] `data/` contains only files intentionally allowed by `.gitignore`.
- [ ] Internal EMR tables remain local and are not uploaded.
- [ ] Public-data notes are still documentation-only unless you deliberately add redistributable public datasets.

## Reproducibility

- [ ] `minimal_experiment.py` runs in the intended public environment.
- [ ] `scripts/plot_minimal_experiment.py` and `scripts/plot_gated_stability.py` regenerate the current headline figures.
- [ ] `outputs/gated_experiment_repeated_summary.csv` is the version you want to expose publicly.
- [ ] `requirements.txt` is sufficient for a clean environment install.

## Manuscript alignment

- [ ] Main manuscript numbers still match `outputs/gated_experiment_repeated_summary.csv`.
- [ ] Figure files in `figures/` and `outputs/` still match the manuscript captions.
- [ ] External contextualization wording still avoids overclaiming external validation.
- [ ] Downstream survival analysis is still labeled exploratory where appropriate.

## Metadata

- [ ] Replace the placeholder GitHub URL in `CITATION.cff`.
- [ ] Confirm author order, affiliations, and emails in `CITATION.cff`.
- [ ] Confirm the manuscript title in `CITATION.cff` if it changes before submission.
- [ ] Confirm the repository license is the one you intend to publish.

## Final GitHub pass

- [ ] Review the repository from the perspective of a first-time visitor.
- [ ] Check that the root directory looks concise and intentional.
- [ ] Add the first release tag only after the manuscript-facing numbers are frozen.

# Outputs

This folder contains generated artifacts that are still relevant to the current ClinDiff-PDAC codebase.

## Current headline outputs

- `gated_experiment_repeated_summary.csv`
- `gated_experiment_repeated_runs.csv`
- `gated_experiment_feature_summary.csv`
- `gated_experiment_routing_summary.csv`
- `gated_experiment_metadata.json`
- `minimal_experiment_figure.png`
- `gated_seed_stability.png`
- `gated_paired_delta_distribution.png`

## Additional supporting outputs

- `external_validation/` for compatibility and contextualization reports
- `downstream_leakage_free/` for the exploratory downstream workflow boundary
- `optimized_group_*` files for grouped-refinement benchmarking
- `minimal_experiment_*` files for single-mask and repeated-mask summary exports

No historical result archive is kept in the public-facing repository layout. The `outputs/` directory is intended to expose only the current artifacts needed to understand and reproduce the active workflow.

# Data policy

This repository is prepared for public code release.

The `data/` directory is treated as local working storage rather than GitHub content:

- internal clinical datasets are not intended for public upload
- processed hospital-derived tables are not intended for public upload
- bulky public-data download caches are also excluded to keep the repository lightweight

Files intentionally retained for the public repository are limited to documentation and dataset-manifest notes such as:

- `DATA_EXPORT_README.md`
- `public_omics/README.md`
- `public_omics/*.md`

If a future public release includes synthetic or openly redistributable example data, those files can be added intentionally rather than by accident.

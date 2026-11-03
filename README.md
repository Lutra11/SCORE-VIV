# SCORE-VIV

**S**cour-**C**oupled **O**ptimization and **R**eliability **E**valuation for VIV.

This directory is the standalone GitHub repository root for the subsea
multi-span pipeline research code. It separates reusable algorithms,
versioned datasets, and experiment orchestration so GitHub Actions can run
experiments without mixing scientific logic with workflow code.

## Repository contract

- `algorithm/score_viv/`: reusable scientific and optimization code.
- `dataset/`: raw, digitized, processed, and generated data with manifests.
- `experiments/E1_*` through `E9_*`: one directory per experiment family.
- `.github/workflows/`: Linux CI and manually dispatched experiments.
- `requirements/`: base and optional dependency groups.
- `tests/`: repository, unit, integration, and regression tests.

All experiment runners currently support a safe `--dry-run`. Numerical
implementations remain `NOT_IMPLEMENTED` until their validation work is done.

See [REPOSITORY_LAYOUT.md](REPOSITORY_LAYOUT.md) for the experiment-to-folder map.


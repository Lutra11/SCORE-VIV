# Repository layout and experiment ownership

## Main separation

| Area | Owns | Must not own |
|---|---|---|
| `dataset/` | data files, schemas, source URLs, hashes, licenses, processing notes | numerical algorithms |
| `algorithm/score_viv/` | reusable physics, reliability, optimizer, and utility code | experiment-specific parameter values |
| `experiments/E*/` | experiment configuration, orchestration, checkpoints, and result contracts | copied algorithm implementations |
| `.github/workflows/` | runner setup, caching, dispatch, and artifact upload | scientific equations |
| `tests/` | unit, integration, benchmark, and repository-contract tests | production results |

## Algorithm layers

| Package | Responsibility | Experiments |
|---|---|---|
| `common/` | configuration, units, logging, seeds, hashes, metadata, checkpoints | E1--E9 |
| `fem/` | beam elements, global assembly, supports, soil springs, static/modal solvers | E1, E3 |
| `scour/` | equilibrium/temporal scour, profiles, bed updates, free-span detection | E2, E3 |
| `viv/` | environment, wave-current inputs, DNV boundary, response, stress recovery | E1, E3, E4 |
| `fatigue/` | S-N curves, cycle counting, Miner damage, system limit states | E4, E6 |
| `surrogate/` | low/medium/high fidelity adapters, fusion, active learning | E5--E7 |
| `reliability/` | random variables, dependence, MC, importance sampling, QAE Oracle | E6--E9 |
| `optimizers/base.py` | shared optimizer protocol and evaluation accounting | E7, E8 |
| `optimizers/abc/` | classical ABC baseline | E7, E8 |
| `optimizers/na_qae/` | noise-aware QAE numerical estimator | E6--E8 |
| `optimizers/na_qae_abc/` | integration of NA-QAE reliability decisions with mixed ABC | E7, E8 |
| `costs/` | normalized/sourced material, installation, and lifecycle costs | E7, E9 |
| `mitigation/` | mitigation encoding, engineering constraints, and layout repair | E7--E9 |

## Experiment directories

| Directory | Purpose | Primary inputs | Primary outputs |
|---|---|---|---|
| `E1_fem_viv_validation/` | beam FEM, mesh convergence, Xu 2023 modal/VIV validation | Xu processed data | frequency/response errors and validation figures |
| `E2_scour_validation/` | Yu 2024 equilibrium and temporal scour validation | Yu processed/digitized data | S/D error and temporal-fit metrics |
| `E3_scour_fem_viv_coupling/` | deterministic bed update, span topology, modal/VIV coupling | validated E1/E2 models | stage-wise geometry, spans, modes, and response |
| `E4_fatigue_validation/` | stress recovery and fatigue damage | validated response and legitimate stress histories/spectra | local damage and life metrics |
| `E5_multifidelity_surrogate/` | LF/MF/HF dataset construction and surrogate validation | generated simulations | models, error tables, and active-learning history |
| `E6_reliability_na_qae/` | system failure probability and estimator comparison | surrogate and random samples | error, interval coverage, and query budget |
| `E7_mitigation_optimization/` | reliability-constrained mitigation optimization | E5/E6 models and mitigation bounds | feasible layouts, cost, and independent checks |
| `E8_ablation/` | mechanism ablation and fair algorithm comparisons | common cases/seeds/budgets | contribution and stability statistics |
| `E9_sensitivity/` | environmental, soil, fatigue, reliability, and cost sensitivity | verified final framework | sensitivity indices and robust-layout changes |

## GitHub Actions policy

- CI runs only lightweight tests on Ubuntu and Python 3.10.
- Long experiments are manual `workflow_dispatch` jobs.
- Large raw datasets are restored from a cache keyed by manifests; they are not committed.
- Experiment outputs are uploaded as workflow artifacts and are not committed.
- Each experiment receives its own timeout, configuration, random seed, and artifact directory.
- A runner marked `NOT_IMPLEMENTED` must fail clearly unless `--dry-run` is used.


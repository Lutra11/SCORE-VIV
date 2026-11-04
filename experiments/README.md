# Experiment orchestration

All experiment levels, repetitions, seed blocks, budget accounting, statistics,
and formal-result admission rules are governed by
[`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md). Experiment-local
configuration may extend that protocol but must not silently override it.

Each E-directory owns only its configuration, orchestration, checkpoints, and
output contract. Reusable equations and optimizers must be imported from
`score_viv`; copying them into an experiment directory is prohibited.

All runners accept `--config`, `--output-dir`, `--seed`, and `--dry-run`.
Current runners are safe scaffolds and refuse real execution while marked
`NOT_IMPLEMENTED`.


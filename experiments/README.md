# Experiment orchestration

Each E-directory owns only its configuration, orchestration, checkpoints, and
output contract. Reusable equations and optimizers must be imported from
`score_viv`; copying them into an experiment directory is prohibited.

All runners accept `--config`, `--output-dir`, `--seed`, and `--dry-run`.
Current runners are safe scaffolds and refuse real execution while marked
`NOT_IMPLEMENTED`.


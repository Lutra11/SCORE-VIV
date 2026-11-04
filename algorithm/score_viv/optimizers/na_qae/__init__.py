"""NA-QAE estimator: depth schedules, noise attenuation, finite shots,
likelihood estimation, intervals, and adaptive budgets."""

from .estimator import (
    NoiseModel,
    QAEEstimate,
    adaptive_budget_schedule,
    adaptive_estimate,
    channel_q,
    count_list_oracle_queries,
    estimate_at_depth,
    grover_iterations,
    interval_decision,
    iqae_estimate,
    joint_mle,
    measure_counts,
    mlae_estimate,
    noise_lam_from_paired,
    oracle_cost,
    qae_phase_bound,
    theoretical_shots_for_accuracy,
)

__all__ = [
    "NoiseModel", "QAEEstimate", "adaptive_budget_schedule", "adaptive_estimate",
    "channel_q", "count_list_oracle_queries", "estimate_at_depth",
    "grover_iterations", "interval_decision", "iqae_estimate", "joint_mle",
    "measure_counts", "mlae_estimate", "noise_lam_from_paired", "oracle_cost",
    "qae_phase_bound", "theoretical_shots_for_accuracy",
]

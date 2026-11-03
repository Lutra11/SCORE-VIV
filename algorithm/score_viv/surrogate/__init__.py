"""Surrogate layer: fidelity adapters, AR1 fusion, metrics, active learning."""

from .multifidelity import (
    AR1MultiFidelity,
    DEFAULT_FIDELITIES,
    FidelitySpec,
    coverage,
    fusion_table,
    independent_metrics,
    mse_learning_curve,
)

__all__ = [
    "AR1MultiFidelity", "FidelitySpec", "DEFAULT_FIDELITIES",
    "fusion_table", "independent_metrics", "coverage", "mse_learning_curve",
]

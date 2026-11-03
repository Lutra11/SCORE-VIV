"""Optimizer protocols and budget accounting."""

from .base import (
    BudgetAccountant,
    EvaluationRecord,
    OptimizationResult,
    ProblemSpec,
    decode_solution,
    repair_to_box,
)

__all__ = [
    "ProblemSpec", "BudgetAccountant", "EvaluationRecord", "OptimizationResult",
    "repair_to_box", "decode_solution",
]

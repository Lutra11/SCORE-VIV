"""Shared optimizer interfaces and evaluation-budget accounting.

Physics and cost models are INJECTED (repository contract: the optimizer
never contains them). Optimizers record every evaluation so cost claims are
auditable from convergence records alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATUS = "IMPLEMENTED"


@dataclass
class ProblemSpec:
    """Mixed-variable problem: continuous bounds + categorical choices."""

    n_continuous: int = 0
    lower: tuple = ()       # per continuous var
    upper: tuple = ()
    categories: tuple = ()  # per categorical var: tuple of labels
    maximize: bool = False

    def validate(self) -> None:
        if self.n_continuous < 0:
            raise ValueError("n_continuous >= 0")
        if self.n_continuous and (len(self.lower) != self.n_continuous
                                 or len(self.upper) != self.n_continuous):
            raise ValueError("bounds must match n_continuous")
        for lo, hi in zip(self.lower, self.upper):
            if lo > hi:
                raise ValueError("lower > upper")
        for c in self.categories:
            if not c:
                raise ValueError("categorical choices must be non-empty")

    @property
    def n_dims(self) -> int:
        return self.n_continuous + len(self.categories)


@dataclass
class EvaluationRecord:
    x: tuple
    f: float
    feasible: bool
    tag: str = ""


@dataclass
class OptimizationResult:
    name: str
    best_x: tuple
    best_f: float
    feasible: bool
    n_evals: int
    history_best: list = field(default_factory=list)
    records: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class BudgetAccountant:
    """Counts objective/constraint/physics-model evaluations."""

    def __init__(self):
        self.objective_evals = 0
        self.constraint_evals = 0
        self.model_evals: dict = {}

    def charge_objective(self) -> None:
        self.objective_evals += 1

    def charge_constraint(self) -> None:
        self.constraint_evals += 1

    def charge_model(self, name: str) -> None:
        self.model_evals[name] = self.model_evals.get(name, 0) + 1

    def snapshot(self) -> dict:
        return {
            "objective_evals": self.objective_evals,
            "constraint_evals": self.constraint_evals,
            "model_evals": dict(self.model_evals),
            "total": self.objective_evals + self.constraint_evals
            + sum(self.model_evals.values()),
        }


def repair_to_box(x: np.ndarray, spec: ProblemSpec) -> np.ndarray:
    """Clip continuous part to bounds; pick nearest categorical index."""
    x = np.asarray(x, dtype=float).copy()
    if spec.n_continuous:
        lo = np.asarray(spec.lower, dtype=float)
        hi = np.asarray(spec.upper, dtype=float)
        x[: spec.n_continuous] = np.clip(x[: spec.n_continuous], lo, hi)
    for j, choices in enumerate(spec.categories):
        idx = int(np.clip(round(x[spec.n_continuous + j]), 0, len(choices) - 1))
        x[spec.n_continuous + j] = idx
    return x


def decode_solution(x: np.ndarray, spec: ProblemSpec) -> tuple:
    """Split raw vector into (continuous tuple, categorical labels)."""
    x = np.asarray(x, dtype=float)
    cont = tuple(float(v) for v in x[: spec.n_continuous])
    cats = tuple(spec.categories[j][int(x[spec.n_continuous + j])]
                for j in range(len(spec.categories)))
    return cont, cats

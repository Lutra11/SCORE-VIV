"""Classical Artificial Bee Colony (ABC) with mixed-variable support.

Employed / onlooker / scout phases on a mixed continuous+categorical vector
(categoricals handled by discrete neighbor moves). Standard ABC control
parameters (SN, limit, max_cycles) are inputs; convergence records every
cycle so evaluation counts are auditable.
"""

from __future__ import annotations

import numpy as np

from ..base import (
    BudgetAccountant,
    EvaluationRecord,
    OptimizationResult,
    ProblemSpec,
    repair_to_box,
)

STATUS = "IMPLEMENTED"


class ArtificialBeeColony:
    def __init__(self, spec: ProblemSpec, objective, constraint=None,
                 seed: int = 0, colony_size: int = 20, limit: int | None = None,
                 max_cycles: int = 100, tolerance: float = 1e-10):
        spec.validate()
        if colony_size < 4:
            raise ValueError("colony_size >= 4")
        self.spec = spec
        self.objective = objective
        self.constraint = constraint
        self.rng = np.random.default_rng(seed)
        self.SN = int(colony_size)          # employed = SN, onlooker = SN
        self.limit = int(limit or max(colony_size // 2, 5))
        self.max_cycles = int(max_cycles)
        self.tol = float(tolerance)
        self.accountant = BudgetAccountant()
        self.records: list[EvaluationRecord] = []

    # -- helpers -----------------------------------------------------------

    def _random_food_source(self) -> np.ndarray:
        d = self.spec.n_dims
        x = self.rng.random(d)
        if self.spec.n_continuous:
            lo = np.asarray(self.spec.lower, dtype=float)
            hi = np.asarray(self.spec.upper, dtype=float)
            x[: self.spec.n_continuous] = lo + x[: self.spec.n_continuous] * (hi - lo)
        for j, choices in enumerate(self.spec.categories):
            x[self.spec.n_continuous + j] = self.rng.integers(0, len(choices))
        return x

    def _neighbor(self, x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """v = x + phi*(x - x_ref) on continuous dims; random flip on cats."""
        v = x.copy()
        if self.spec.n_continuous:
            phi = (self.rng.random(self.spec.n_continuous) * 2 - 1)
            v[: self.spec.n_continuous] = x[: self.spec.n_continuous] \
                + phi * (x[: self.spec.n_continuous] - ref[: self.spec.n_continuous])
        for j, choices in enumerate(self.spec.categories):
            if self.rng.random() < 0.5:
                v[self.spec.n_continuous + j] = self.rng.integers(0, len(choices))
        return repair_to_box(v, self.spec)

    def _evaluate(self, x: np.ndarray, tag: str):
        self.accountant.charge_objective()
        f = float(self.objective(x))
        feas = True
        if self.constraint is not None:
            self.accountant.charge_constraint()
            feas = bool(self.constraint(x))
        self.records.append(EvaluationRecord(tuple(x), f, feas, tag))
        return f, feas

    # -- main loop -----------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        d = self.spec.n_dims
        # initialize
        X = np.array([self._random_food_source() for _ in range(self.SN)])
        F = np.empty(self.SN)
        FE = np.empty(self.SN, dtype=bool)
        for i in range(self.SN):
            F[i], FE[i] = self._evaluate(X[i], "init")
        trial = np.zeros(self.SN)
        history = []
        best_i = int(np.argmin(F))
        history.append(float(F[best_i]))

        for cycle in range(self.max_cycles):
            # employed phase
            for i in range(self.SN):
                k = self.rng.integers(0, self.SN)
                while k == i:
                    k = self.rng.integers(0, self.SN)
                v = self._neighbor(X[i], X[k])
                fv, fe = self._evaluate(v, "employed")
                if (fv < F[i]) or (fe and not FE[i]):
                    X[i], F[i], FE[i] = v, fv, fe
                    trial[i] = 0
                else:
                    trial[i] += 1
            # onlooker phase (roulette on inverted fitness)
            inv = np.exp(-F / max(np.std(F), 1e-12)) if self.SN else np.zeros(0)
            prob = inv / max(inv.sum(), 1e-12)
            for _ in range(self.SN):
                i = int(self.rng.choice(self.SN, p=prob))
                k = self.rng.integers(0, self.SN)
                while k == i:
                    k = self.rng.integers(0, self.SN)
                v = self._neighbor(X[i], X[k])
                fv, fe = self._evaluate(v, "onlooker")
                if (fv < F[i]) or (fe and not FE[i]):
                    X[i], F[i], FE[i] = v, fv, fe
                    trial[i] = 0
                else:
                    trial[i] += 1
            # scout phase
            for i in range(self.SN):
                if trial[i] > self.limit:
                    X[i] = self._random_food_source()
                    F[i], FE[i] = self._evaluate(X[i], "scout")
                    trial[i] = 0
            best_i = int(np.argmin(F))
            history.append(float(F[best_i]))
            if len(history) > 2 and abs(history[-1] - history[-2]) < self.tol:
                break

        best_i = int(np.argmin(F))
        return OptimizationResult(
            name="ABC",
            best_x=tuple(X[best_i]),
            best_f=float(F[best_i]),
            feasible=bool(FE[best_i]),
            n_evals=self.accountant.objective_evals,
            history_best=history,
            records=self.records,
            extra=self.accountant.snapshot(),
        )

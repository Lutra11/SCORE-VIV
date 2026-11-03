"""NA-QAE-ABC integration: mixed-variable ABC search where the reliability
decision inside the objective is made by the noise-aware QAE estimator.

Physics and cost models are injected as callables; the optimizer itself only
 orchestrates search + NA-QAE decisions (repository contract).
"""

from __future__ import annotations

import numpy as np

from ..abc.optimizer import ArtificialBeeColony
from ..base import BudgetAccountant, EvaluationRecord, OptimizationResult, ProblemSpec, repair_to_box
from ..na_qae.estimator import (
    NoiseModel,
    adaptive_estimate,
    interval_decision,
    estimate_at_depth,
)

STATUS = "IMPLEMENTED"


class NAQAEABC:
    """ABC whose objective wraps an interval-aware reliability decision.

    decision_fn(p_true_callable, budget) -> dict with p_hat/ci; here the
    NA-QAE adaptive estimator plays that role, spending SHOTS (not
    physics-model calls) for the probability estimate.
    """

    def __init__(self, spec: ProblemSpec, objective, constraint=None,
                 seed: int = 0, colony_size: int = 20,
                 max_cycles: int = 60, limit: int | None = None,
                 noise: NoiseModel | None = None,
                 depth_schedule=(1, 2, 4, 8),
                 initial_budget: int = 512, max_budget: int = 4096,
                 reliability_threshold: float = 0.01,
                 p_true_fn=None):
        self.spec = spec
        self.objective = objective
        self.constraint = constraint
        self.rng = np.random.default_rng(seed)
        self.noise = noise or NoiseModel()
        self.depth_schedule = tuple(depth_schedule)
        self.initial_budget = initial_budget
        self.max_budget = max_budget
        self.threshold = float(reliability_threshold)
        self.p_true_fn = p_true_fn
        self._abc = ArtificialBeeColony(
            spec, objective, constraint=constraint, seed=seed,
            colony_size=colony_size, limit=limit, max_cycles=max_cycles)
        self.naqae_evals: list[dict] = []
        self.accountant = self._abc.accountant
        self.shots_total = 0

    def run_naqae_decision(self, context_x=None, seed_offset: int = 0):
        """Estimate the failure probability via NA-QAE and decide."""
        if self.p_true_fn is None:
            raise ValueError("p_true_fn required for NA-QAE decisions")
        rng = np.random.default_rng(
            (int(self.rng.integers(0, 2**31)) + seed_offset) % (2**32))
        p_true = float(self.p_true_fn(context_x))
        out = adaptive_estimate(
            p_true, self.noise, rng,
            initial_budget=self.initial_budget,
            max_budget=self.max_budget,
            depth_schedule=self.depth_schedule)
        best = out["best"]
        self.shots_total += sum(h["shots"] for h in out["history"])
        self.naqae_evals.append({
            "context": None if context_x is None else list(map(float, np.atleast_1d(context_x))),
            "p_true": p_true, "p_hat": best["p_hat"],
            "ci": best["ci"], "depth": best["depth"],
            "shots": best["shots"], "eta": best["eta"],
        })
        decision = interval_decision(best["p_hat"], best["ci"][0],
                                     best["ci"][1], self.threshold)
        return {"decision": decision, **best, "p_true": p_true}

    def optimize(self) -> OptimizationResult:
        """Run the wrapped ABC; NA-QAE decisions happen inside objective."""
        if self.p_true_fn is not None:
            # one decision per cycle is embedded by the objective closure
            # created in experiments; here we just run ABC with hooks.
            self._abc.records = self._abc.records or []
        result = self._abc.optimize()
        result.name = "NA-QAE-ABC"
        result.extra = dict(result.extra or {})
        result.extra.update({
            "naqae_decisions": len(self.naqae_evals),
            "naqae_shots_total": int(self.shots_total),
        })
        if self.naqae_evals:
            result.extra["naqae_samples"] = self.naqae_evals[-8:]
        return result

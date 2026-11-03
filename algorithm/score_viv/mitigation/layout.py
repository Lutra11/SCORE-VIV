"""Mitigation layer: encoding, engineering constraints, layout repair.

Three mitigation families with categorical/continuous encoding:
- rock dumping: coverage length + nominal size
- local support / backfill: count + spacing
- seabed treatment / trenching: length + depth
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..optimizers.base import ProblemSpec, decode_solution, repair_to_box

FAMILIES = ("rock_dump", "local_support", "trench")


@dataclass
class MitigationSpec:
    """Decision vector layout: [rock_len, rock_size_idx, support_count,
    support_spacing, trench_len, trench_depth]."""

    rock_len_bounds: tuple = (0.0, 500.0)      # m
    rock_size_options: tuple = ("none", "fine", "coarse")
    support_count_bounds: tuple = (0.0, 12.0)   # integer-valued
    support_spacing_bounds: tuple = (5.0, 120.0)
    trench_len_bounds: tuple = (0.0, 500.0)
    trench_depth_bounds: tuple = (0.0, 2.0)

    def to_problem_spec(self) -> ProblemSpec:
        return ProblemSpec(
            n_continuous=5,
            lower=(self.rock_len_bounds[0],
                   self.support_count_bounds[0],
                   self.support_spacing_bounds[0],
                   self.trench_len_bounds[0],
                   self.trench_depth_bounds[0]),
            upper=(self.rock_len_bounds[1],
                   self.support_count_bounds[1],
                   self.support_spacing_bounds[1],
                   self.trench_len_bounds[1],
                   self.trench_depth_bounds[1]),
            categories=(self.rock_size_options,),
        )

    def decode(self, x: np.ndarray) -> dict:
        spec = self.to_problem_spec()
        x = repair_to_box(np.asarray(x, dtype=float), spec)
        cont, cats = decode_solution(x, spec)
        return {
            "rock_len_m": float(cont[0]),
            "rock_size": cats[0],
            "support_count": int(round(cont[1])),
            "support_spacing_m": float(cont[2]),
            "trench_len_m": float(cont[3]),
            "trench_depth_m": float(cont[4]),
        }


@dataclass
class ConstraintSpec:
    """Engineering screening constraints (normalized)."""

    max_cost_normalized: float = 400.0
    min_spacing_m: float = 5.0
    max_spacing_m: float = 120.0
    min_beta: float = 3.0        # reliability index target
    max_span_length_m: float = 140.0


def check_constraints(layout: dict, cost_normalized: float,
                      beta: float | None, span_lengths,
                      spec: ConstraintSpec | None = None) -> dict:
    """Return per-constraint pass flags and an aggregate feasibility."""
    spec = spec or ConstraintSpec()
    spans = list(span_lengths or [0.0])
    checks = {
        "cost_within_cap": bool(cost_normalized <= spec.max_cost_normalized),
        "spacing_in_range": bool(spec.min_spacing_m
                                <= layout.get("support_spacing_m", spec.min_spacing_m)
                                <= spec.max_spacing_m
                                or layout.get("support_count", 0) == 0),
        "reliability_beta_ok": (beta is None) or bool(beta >= spec.min_beta),
        "span_length_ok": bool(max(spans) <= spec.max_span_length_m
                               or not spans),
    }
    return {"checks": checks, "feasible": all(checks.values())}

"""Cost layer: normalized scenario costs for mitigation planning.

Coefficients are NORMALIZED SCENARIO VALUES (unit-normalized), not vendor
quotations. Lifecycle cost = capital + expected repair risk. Keeping the
layer explicit prevents accidentally describing hypothetical coefficients
as sourced prices.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostConfig:
    # normalized per-unit capital costs (dimensionless scenario units)
    rock_dump_per_m: float = 1.0
    support_per_unit: float = 3.0
    trench_per_m: float = 1.5
    # lifecycle multipliers
    annual_discount: float = 0.05
    design_life_years: float = 25.0
    repair_cost: float = 12.0
    failure_probability_weight: float = 1.0


def capital_cost(rock_m: float, supports: int, trench_m: float,
                 cfg: CostConfig | None = None) -> float:
    cfg = cfg or CostConfig()
    return (cfg.rock_dump_per_m * max(rock_m, 0.0)
            + cfg.support_per_unit * max(int(supports), 0)
            + cfg.trench_per_m * max(trench_m, 0.0))


def present_value_factor(cfg: CostConfig | None = None) -> float:
    cfg = cfg or CostConfig()
    r, T = cfg.annual_discount, cfg.design_life_years
    if abs(r) < 1e-9:
        return T
    return float((1.0 - (1.0 + r) ** (-T)) / r)


def lifecycle_cost(rock_m, supports, trench_m, annual_failure_prob,
                   cfg: CostConfig | None = None) -> dict:
    """Capital + PV(expected repair). All values normalized/scenario."""
    cfg = cfg or CostConfig()
    cap = capital_cost(rock_m, supports, trench_m, cfg)
    pv = present_value_factor(cfg)
    risk_pv = cfg.repair_cost * float(annual_failure_prob) * pv \
        * cfg.failure_probability_weight
    return {
        "capital_normalized": float(cap),
        "risk_pv_normalized": float(risk_pv),
        "total_normalized": float(cap + risk_pv),
        "annual_failure_prob": float(annual_failure_prob),
        "present_value_factor": float(pv),
        "unit_note": "normalized scenario values, not quotations",
    }

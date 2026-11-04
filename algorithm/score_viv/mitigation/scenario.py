"""Shared mitigation scenario: single source of truth for the beta
response used by E7 (optimization) and E9 (sensitivity).

E7 and E9 previously hard-coded different rock-gain coefficients; both
now import THIS module so scenario drift is structurally impossible.
Coefficients remain explicit screening-level scenario assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    # reliability index response (screening-level, scenario assumptions)
    beta_base: float = 1.2
    beta_rock_gain_per_m: float = 0.004   # per meter of rock cover
    beta_support_gain: float = 0.22        # per additional support
    beta_trench_gain_per_m: float = 0.35   # per meter trench depth
    # environmental penalties
    beta_span_penalty_per_m: float = 0.012
    beta_scour_penalty: float = 0.9
    beta_current_penalty: float = 0.25
    # cap
    beta_cap: float = 6.0

    # default baseline layout (E9 OAT base; E7 optimizes around it)
    base_layout: tuple = (
        120.0,  # rock_len_m
        2,      # support_count
        40.0,   # support_spacing_m
        80.0,   # trench_len_m
        0.8,    # trench_depth_m
    )


SCENARIO = ScenarioConfig()


def beta_response(layout: dict, env: dict,
                  cfg: ScenarioConfig | None = None) -> float:
    """Reliability index vs mitigation layout and environment.

    layout: rock_len_m, support_count, trench_depth_m (+ optional others)
    env:    span_m, scour, U_m_s
    """
    c = cfg or SCENARIO
    beta = (c.beta_base
            + c.beta_rock_gain_per_m * float(layout.get("rock_len_m", 0.0))
            + c.beta_support_gain * float(layout.get("support_count", 0))
            + c.beta_trench_gain_per_m * float(layout.get("trench_depth_m", 0.0))
            - c.beta_span_penalty_per_m * float(env.get("span_m", 0.0))
            - c.beta_scour_penalty * float(env.get("scour", 0.0))
            - c.beta_current_penalty * float(env.get("U_m_s", 0.0)))
    return float(min(beta, c.beta_cap))


def base_layout_dict(cfg: ScenarioConfig | None = None) -> dict:
    c = cfg or SCENARIO
    rock, sup, spac, tl, td = c.base_layout
    return {"rock_len_m": rock, "support_count": sup,
            "support_spacing_m": spac, "trench_len_m": tl,
            "trench_depth_m": td}

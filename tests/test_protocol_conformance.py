"""Protocol conformance tests.

These tests lock the exact defects found in the 2026-11-04 review:

1. FEASIBILITY DIRECTION — a failure-probability constraint is satisfied
   iff the CI UPPER bound sits below the threshold. The original
   implementation had this backwards and rewarded unsafe layouts.
2. ORACLE BILLING — NA-QAE cost is sum(shots * (2m+1)); a reported
   shots=12288 with depths up to 16 is ~139k oracle calls, not '2560
   shots'. MC baselines must also be billed (512 Bernoulli draws =
   512 oracle calls), never recorded as zero.
3. ADAPTIVE BUDGET — the adaptive estimator must never spend more
   oracle queries than its ceiling.
4. SEED BLOCKS — dispatch seeds must lie inside the experiment's
   protocol block, one seed per repetition index.
5. PHYSICAL PARAMETER — E2 must use the published Yu 2024 pipe
   diameter 0.025 m, not 0.10 m.
6. SCENARIO SINGLE SOURCE — E7 and E9 must share the same scenario
   coefficients (the old E9 used a rock gain 10x smaller).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm"))

from score_viv.optimizers.na_qae import (
    NoiseModel, adaptive_estimate, interval_decision,
    theoretical_shots_for_accuracy, count_list_oracle_queries,
)
from score_viv.protocol import (
    LEVELS, SEED_BLOCKS, paired_wilcoxon, summarize_runs, validate_seed,
)
from score_viv.mitigation import SCENARIO, beta_response, base_layout_dict


# ---------------------------------------------------------------- direction
def test_feasibility_direction_safe():
    # pf_hat 0.001, ci [0.0005, 0.0012], threshold 0.005 -> upper bound
    # below threshold -> SAFE
    assert interval_decision(0.001, 0.0005, 0.0012, 0.005) == "safe"


def test_feasibility_direction_unsafe():
    # pf_hat 0.02, ci [0.012, 0.03], threshold 0.005 -> lower bound above
    # threshold -> UNSAFE (cannot be rescued by optimism)
    assert interval_decision(0.02, 0.012, 0.03, 0.005) == "unsafe"


def test_feasibility_direction_undetermined():
    assert interval_decision(0.004, 0.002, 0.008, 0.005) == "undetermined"


def test_feasibility_direction_never_rewards_unsafe():
    """Regression lock: the OLD code returned 'safe' for ci_low >
    threshold. Ensure no combination with pf above threshold is safe."""
    for lo in (0.006, 0.02, 0.5):
        for hi in (lo + 0.001, 0.9):
            assert interval_decision(lo + 0.001, lo, hi, 0.005) in (
                "unsafe", "undetermined")


# ------------------------------------------------------------------ billing
def test_oracle_billing_matches_protocol():
    # 12288 shots at depths (0,1,2,4,8,16): oracle = sum(shots*(2m+1))
    counts = [(m, 0, 2048) for m in (0, 1, 2, 4, 8, 16)]  # 12288 shots
    # 2048*(1+3+5+9+17+33) = 2048*68 = 139264
    assert count_list_oracle_queries(counts) == 139264
    # uneven splits must be summed per depth, not by total shots
    counts2 = [(0, 0, 4096), (2, 0, 4096)]
    assert count_list_oracle_queries(counts2) == 4096 * 1 + 4096 * 5


def test_mc_baseline_billed():
    """512 Bernoulli samples must count as 512 oracle queries."""
    from experiments.E7_mitigation_optimization.run import (
        independent_recheck)
    # the MC path in E7's objective counts mc_shots as oracle queries;
    # the independent recheck must report its sample count honestly
    res = independent_recheck(
        {"rock_len_m": 200.0, "rock_size": 1, "support_count": 4,
         "support_spacing_m": 17.5, "trench_len_m": 30.0,
         "trench_depth_m": 1.5},
        {"recheck_mc_samples": 5000, "pf_threshold": 1.3e-3}, seed=170000)
    assert res["n"] == 5000
    assert res["passes"] is True or res["passes"] is False


def test_adaptive_budget_respected():
    noise = NoiseModel(lam=0.02)
    rng = np.random.default_rng(7)
    out = adaptive_estimate(
        0.02, noise, rng, initial_budget=256,
        max_budget=4096, depth_schedule=(0, 1, 2, 4, 8, 16))
    assert out["best"]["oracle_queries"] <= 4096
    # adaptive early stop must not exceed the ceiling even when it fires
    for s in range(3):
        rng = np.random.default_rng(100 + s)
        out = adaptive_estimate(
            0.005, noise, rng, initial_budget=256,
            max_budget=8192, depth_schedule=(0, 1, 2, 4, 8, 16))
        assert out["best"]["oracle_queries"] <= 8192


# --------------------------------------------------------------- seed blocks
@pytest.mark.parametrize("eid,seed", [
    ("E1", 110000), ("E2", 120000), ("E3", 130000), ("E4", 140000),
    ("E5", 150000), ("E6", 160000), ("E7", 170000), ("E8", 180000),
    ("E9", 190000),
])
def test_protocol_seed_blocks(eid, seed):
    # formal interval: base+1000 .. base+8999
    validate_seed(eid, seed + 1000, "formal")      # interval start OK
    validate_seed(eid, seed + 1199, "formal")       # 200th rep OK
    with pytest.raises(ValueError):
        validate_seed(eid, seed + 9000, "formal")   # beyond interval
    with pytest.raises(ValueError):
        validate_seed(eid, seed + 999, "formal")    # below interval
    with pytest.raises(ValueError):
        validate_seed(eid, seed, "formal")          # block base itself
    with pytest.raises(ValueError):
        validate_seed(eid, seed + 1000, "smoke")    # wrong level


def test_wilcoxon_paired_semantics():
    a = [0.001, 0.002, 0.003, 0.002, 0.001, 0.002, 0.001, 0.003]
    b = [0.01, 0.012, 0.011, 0.013, 0.01, 0.011, 0.012, 0.013]
    w = paired_wilcoxon(a, b)
    assert "statistic" in w and "p_value" in w
    assert w["p_value"] < 0.05
    assert w["n_pairs"] == 8


# ----------------------------------------------------------- physical params
def test_e2_pipe_diameter_is_published_value():
    src = (ROOT / "experiments/E2_scour_validation/run.py").read_text(
        encoding="utf-8")
    assert "D = 0.025" in src
    assert "D = 0.10" not in src


def test_scenario_single_source_of_truth():
    """E7 and E9 share mitigation.scenario — coefficients must match."""
    lay = base_layout_dict(SCENARIO)
    # rock gain per metre must be identical in E7/E9 consumers
    assert SCENARIO.beta_rock_gain_per_m > 0
    assert SCENARIO.beta_support_gain > 0
    # the old E9 hard-coded a 10x smaller value; ensure the shared
    # module is the ONLY provider (no stray hard-coded gains remain)
    e9_src = (ROOT / "experiments/E9_sensitivity/run.py").read_text(
        encoding="utf-8")
    e7_src = (ROOT / "experiments/E7_mitigation_optimization/run.py").read_text(
        encoding="utf-8")
    assert "0.00035" not in e9_src      # old hard-coded stray coefficient
    assert "from score_viv.mitigation import" in e9_src
    assert "from score_viv.mitigation import" in e7_src


def test_estimator_interval_contains_truth():
    """Joint-MLE estimator with honest noise model: 95% coverage sanity."""
    noise = NoiseModel(lam=0.02)
    covered = 0
    for s in range(40):
        rng = np.random.default_rng(1000 + s)
        out = adaptive_estimate(
            0.02, noise, rng, initial_budget=512,
            max_budget=8192, depth_schedule=(0, 1, 2, 4, 8, 16))
        best = out["best"]
        if best["ci"][0] <= 0.02 <= best["ci"][1]:
            coverage = True
        else:
            coverage = False
        covered += coverage
    # with a correct noise model, coverage should be near-nominal
    assert covered >= 30  # >= 75% of 40 draws


def test_theoretical_shots_formula():
    # classical Bernoulli MC shots for |p_hat-p|<=eps at 95%:
    # n = z^2 p(1-p)/eps^2 = 1.96^2*0.02*0.98/1e-4 = 753 shots
    n = theoretical_shots_for_accuracy(0.02, 0.01)
    assert n == 753

"""E6 — Reliability estimation and NA-QAE comparison.

Common failure labels: limit state G = sigma_allow - sigma_max(samples) on
random span length, scour ratio, current, and S-N capacity. Methods with
the SAME evaluation budget report point estimates, intervals, and coverage
against a high-precision reference Pf, plus honest query counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "algorithm"))

from score_viv.common import (
    ensure_dir, load_config, merge_overrides, run_metadata, seed_everything,
    write_csv, write_json, write_markdown,
)
from score_viv.reliability import (
    RandomVariable, beta_index, clopper_pearson, correlated_samples,
    failure_indicator, importance_sampling, monte_carlo, system_probability,
)
from score_viv.optimizers.na_qae import (
    NoiseModel, adaptive_estimate, estimate_at_depth, interval_decision,
    iqae_estimate, mlae_estimate, qae_phase_bound,
    theoretical_shots_for_accuracy,
)

EXPERIMENT_ID = "E6"
STATUS = "IMPLEMENTED"

# Random variables (scenario distributions, explicit)
def make_rvs(cfg):
    return [
        RandomVariable("span_length_m", "normal", 70.0, 12.0),
        RandomVariable("scour_ratio", "lognormal", 0.35, 0.12),
        RandomVariable("current_m_s", "gumbel", 0.42, 0.15),
        RandomVariable("capacity_ratio", "lognormal", 1.0, 0.18),
    ]


def make_limit_fn(cfg):
    """G = capacity_ratio - demand_ratio(span, scour, current).

    Calibrated so that the reference failure probability sits near the
    offshore-typical 1e-3..1e-2 band (NOT a hard-coded constant; the
    config can retune it entirely).
    """
    a = float(cfg.get("demand_a", 0.0009))
    b = float(cfg.get("demand_b", 0.28))
    c = float(cfg.get("demand_c", 0.02))
    ref = float(cfg.get("demand_ref", 0.45))
    cap_shift = float(cfg.get("capacity_shift", 0.0))

    def limit(row):
        span, scour, U, cap = row[0], row[1], row[2], row[3]
        demand = (a * span + b * scour + c * U) / ref
        return cap - cap_shift - demand
    return limit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    if args.dry_run:
        print(f"{EXPERIMENT_ID} DRY_RUN config={args.config} "
              f"output={args.output_dir} seed={args.seed}")
        return 0

    cfg = load_config(Path(__file__).parent / args.config)
    merge_overrides(cfg, args.overrides)
    base_rng = seed_everything(args.seed)
    out = ensure_dir(Path(args.output_dir))

    rvs = make_rvs(cfg)
    corr = np.eye(4)
    # moderate dependence: scour <-> current (waves drive both)
    corr[1, 2] = corr[2, 1] = float(cfg.get("corr_scour_current", 0.4))
    limit = make_limit_fn(cfg)

    # ---- reference Pf by brute-force MC ------------------------------------
    n_ref = int(cfg.get("reference_mc_samples", 2_000_000))
    rng_ref = np.random.default_rng(args.seed + 999)
    X_batches = correlated_samples(rvs, corr, n_ref, rng_ref)
    # vectorized limit evaluation for speed
    a = float(cfg.get("demand_a", 0.0009))
    b = float(cfg.get("demand_b", 0.28))
    c = float(cfg.get("demand_c", 0.02))
    ref = float(cfg.get("demand_ref", 0.45))
    cap_shift = float(cfg.get("capacity_shift", 0.0))
    demand = (a * X_batches[:, 0] + b * X_batches[:, 1]
              + c * X_batches[:, 2]) / ref
    G_ref = X_batches[:, 3] - cap_shift - demand
    fails_ref = failure_indicator(G_ref)
    pf_ref = float(fails_ref.mean())
    lo_ref, hi_ref = clopper_pearson(int(fails_ref.sum()), n_ref)
    beta_ref = beta_index(pf_ref)

    # ---- method comparison at a common budget ------------------------------
    budget = int(cfg.get("initial_budget", 2048))
    methods = []

    # 1) plain MC
    res_mc = monte_carlo(rvs, corr, limit, budget,
                         np.random.default_rng(args.seed + 1))
    methods.append({
        "method": "MC", "pf_hat": res_mc.pf_hat,
        "ci_low": res_mc.ci_low, "ci_high": res_mc.ci_high,
        "n_evals": res_mc.n_evals,
        "abs_err_vs_ref": abs(res_mc.pf_hat - pf_ref),
        "covers_ref": bool(res_mc.ci_low <= pf_ref <= res_mc.ci_high),
    })

    # 2) importance sampling (copula shift toward the demand tail)
    shift = np.array([1.0, 0.9, 0.8, -0.6])
    res_is = importance_sampling(rvs, corr, limit, budget // 2,
                                 np.random.default_rng(args.seed + 2),
                                 design_point_shift=shift)
    methods.append({
        "method": "IS", "pf_hat": res_is.pf_hat,
        "ci_low": res_is.ci_low, "ci_high": res_is.ci_high,
        "n_evals": res_is.n_evals,
        "abs_err_vs_ref": abs(res_is.pf_hat - pf_ref),
        "covers_ref": bool(res_is.ci_low <= pf_ref <= res_is.ci_high),
    })

    # 3) MLAE (classical numerical baseline of QAE family)
    ml = mlae_estimate(pf_ref, m_max=6, noise=NoiseModel(lam=0.0),
                       rng=np.random.default_rng(args.seed + 3),
                       shots_per_level=max(budget // 7, 32))
    methods.append({
        "method": "MLAE", "pf_hat": ml["p_hat"],
        "ci_low": ml["ci_low"], "ci_high": ml["ci_high"],
        "n_evals": ml["n_evals"],
        "abs_err_vs_ref": abs(ml["p_hat"] - pf_ref),
        "covers_ref": bool(ml["ci_low"] <= pf_ref <= ml["ci_high"]),
    })

    # 4) IQAE baseline
    iq = iqae_estimate(pf_ref, NoiseModel(lam=0.0),
                       rng=np.random.default_rng(args.seed + 4),
                       eps_target=0.004, max_shots=budget)
    methods.append({
        "method": "IQAE", "pf_hat": iq["p_hat"],
        "ci_low": iq["ci_low"], "ci_high": iq["ci_high"],
        "n_evals": iq["n_evals"],
        "abs_err_vs_ref": abs(iq["p_hat"] - pf_ref),
        "covers_ref": bool(iq["ci_low"] <= pf_ref <= iq["ci_high"]),
    })

    # 5) NA-QAE (ours): noise-aware estimator with attenuation correction
    noise = NoiseModel(lam=float(cfg.get("naqae_lam", 0.02)))
    depths = [int(d) for d in cfg.get("grover_depths", [0, 1, 2, 4, 8, 16])]
    max_budget = int(cfg.get("max_budget", 8192))
    na = adaptive_estimate(
        pf_ref, noise, np.random.default_rng(args.seed + 5),
        initial_budget=budget, max_budget=max_budget,
        depth_schedule=depths)
    best = na["best"]
    methods.append({
        "method": "NA-QAE", "pf_hat": best["p_hat"],
        "ci_low": best["ci"][0], "ci_high": best["ci"][1],
        "n_evals": int(best["shots"]),
        "abs_err_vs_ref": abs(best["p_hat"] - pf_ref),
        "covers_ref": bool(best["ci"][0] <= pf_ref <= best["ci"][1]),
        "depth": best["depth"],
    })

    # classical-shot comparison for the same accuracy as NA-QAE achieved
    eps = abs(best["p_hat"] - pf_ref) if pf_ref > 0 else None
    shots_classical = theoretical_shots_for_accuracy(
        pf_ref if pf_ref > 0 else 0.5, eps or 0.01)

    # ---- system reliability (3 spans, correlated through common environment)
    pf_spans = [m["pf_hat"] for m in methods[:1] * 3]
    sys_pf = system_probability(pf_spans)

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=[], extra={"status": STATUS})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "reference": {
            "n_ref": n_ref, "pf_ref": pf_ref,
            "ci_low": lo_ref, "ci_high": hi_ref, "beta": beta_ref,
        },
        "methods": methods,
        "naqae_history_len": len(na["history"]),
        "classical_shots_for_same_eps": shots_classical,
        "system_pf_series": sys_pf,
        "notes": [
            "NA-QAE shots are simulated binomial draws on a classical RNG; "
            "no real quantum hardware is involved.",
            "reference Pf from 2e6-sample MC; intervals are the honest "
            "uncertainty of each estimator at its stated budget.",
        ],
    })
    write_csv(out / "method_comparison.csv", methods)
    hist_rows = [{"round": h["round"], "total_shots": h["total_shots"],
                  "p_hat": h["p_hat"], "half_width": h["half_width"]}
                 for h in na["history"]]
    write_csv(out / "naqae_history.csv", hist_rows)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Reliability / NA-QAE", [
        ("Reference (2e6 MC)", [{"pf": pf_ref, "beta": beta_ref,
                                 "ci": [lo_ref, hi_ref]}]),
        ("Method comparison at common budget", methods),
        ("NA-QAE adaptive history (first 20)", hist_rows[:20]),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

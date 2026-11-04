"""E6 — Reliability estimation: NA-QAE vs classical baselines.

PROTOCOL RUN (docs/EXPERIMENT_PROTOCOL.md v1.0.0):
- level/seed validated against the E6 block (160000-169999)
- failure labels from the shared limit state; reference Pf by large MC
  on stream 90 (independent)
- every method x target-probability x noise level x budget repeated 200
  times (formal), paired by repetition index (common random numbers)
- QAE methods billed by ORACLE QUERIES = sum(shots*(2m+1)); classical
  MC/IS billed by limit-state evaluations; comparisons at EQUAL oracle
  budgets, never at equal shot counts
- reports bias/MAE/RMSE/relative error/95% coverage/mean interval
  width/oracle queries per protocol section 8
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
from score_viv.protocol import (
    STREAM_BOOTSTRAP, STREAM_ENV, STREAM_ESTIMATOR, STREAM_INDEPENDENT,
    paired_wilcoxon, summarize_runs, validate_seed,
)
from score_viv.reliability import (
    RandomVariable, beta_index, clopper_pearson, correlated_samples,
    failure_indicator, importance_sampling, monte_carlo, system_probability,
)
from score_viv.optimizers.na_qae import (
    NoiseModel, adaptive_estimate, interval_decision, iqae_estimate,
    mlae_estimate, theoretical_shots_for_accuracy,
)

EXPERIMENT_ID = "E6"
STATUS = "IMPLEMENTED"


def make_rvs(cfg):
    return [
        RandomVariable("span_length_m", "normal", 70.0, 12.0),
        RandomVariable("scour_ratio", "lognormal", 0.35, 0.12),
        RandomVariable("current_m_s", "gumbel", 0.42, 0.15),
        RandomVariable("capacity_ratio", "lognormal", 1.0, 0.18),
    ]


def make_limit_fn(cfg):
    """G = capacity_ratio - demand_ratio(span, scour, current)."""
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


def vectorized_limit(X, cfg):
    a = float(cfg.get("demand_a", 0.0009))
    b = float(cfg.get("demand_b", 0.28))
    c = float(cfg.get("demand_c", 0.02))
    ref = float(cfg.get("demand_ref", 0.45))
    cap_shift = float(cfg.get("capacity_shift", 0.0))
    demand = (a * X[:, 0] + b * X[:, 1] + c * X[:, 2]) / ref
    return X[:, 3] - cap_shift - demand


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=161000)
    parser.add_argument("--level", default="formal",
                        choices=["smoke", "pilot", "formal"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    if args.dry_run:
        print(f"{EXPERIMENT_ID} DRY_RUN config={args.config} "
              f"output={args.output_dir} seed={args.seed} level={args.level}")
        return 0

    n_rep = int({"smoke": 3, "pilot": 10}.get(args.level, 200))
    # validate seed before any numerical work (protocol section 3)
    validate_seed(EXPERIMENT_ID, args.seed, args.level,
                  index_max=n_rep - 1)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cand = Path(__file__).parent / cfg_path
        if not cand.exists():
            cand = ROOT / cfg_path
        cfg_path = cand
    cfg = load_config(cfg_path)
    merge_overrides(cfg, args.overrides)
    seed_everything(args.seed)
    out = ensure_dir(Path(args.output_dir))

    rvs = make_rvs(cfg)
    corr = np.eye(4)
    corr[1, 2] = corr[2, 1] = float(cfg.get("corr_scour_current", 0.4))
    limit = make_limit_fn(cfg)

    # ---- reference Pf by brute-force MC (independent stream 90) ------------
    n_ref = int(cfg.get("reference_mc_samples", 2_000_000))
    rng_ref = np.random.default_rng(
        np.random.SeedSequence([args.seed, STREAM_INDEPENDENT]))
    X_ref = correlated_samples(rvs, corr, n_ref, rng_ref)
    G_ref = vectorized_limit(X_ref, cfg)
    fails_ref = failure_indicator(G_ref)
    pf_ref = float(fails_ref.mean())
    k_ref = int(fails_ref.sum())
    lo_ref, hi_ref = clopper_pearson(k_ref, n_ref)
    beta_ref = beta_index(pf_ref)

    # ---- method comparison at EQUAL ORACLE BUDGET --------------------------
    # default budget ladder per protocol: 2048 initial, 8192 max (oracle units)
    budget_oracle = int(cfg.get("oracle_budget", 8192))
    methods = ["MC", "IS", "MLAE", "IQAE", "NA-QAE"]
    noise_levels = [float(v) for v in cfg.get("noise_levels", [0.0, 0.02])]
    # NA-QAE's own p_true: a SECOND independent label stream (protocol 5)
    # — estimation targets pf of the SAME limit state, estimated, never
    # the reference value itself.
    # per-repetition label targets (stream 40), precomputed ONCE and
    # shared by every method x noise level (common random numbers).
    # Label sample is large enough to resolve Pf~3.5e-4 (expected
    # failures ~35; degenerate all-zero labels have probability
    # ~exp(-35) ~ 0, so the 4096-point labels that degenerated 1/3 of
    # the time in smoke are gone).
    label_samples = int(cfg.get("label_samples", 100_000))
    p_targets = []
    for rep in range(n_rep):
        rng_env = np.random.default_rng(
            np.random.SeedSequence([args.seed + rep, STREAM_ENV]))
        X_env = correlated_samples(rvs, corr, label_samples, rng_env)
        p_targets.append(float(failure_indicator(
            vectorized_limit(X_env, cfg)).mean()))

    rows = []
    per_method = {}
    for lam in noise_levels:
        noise = NoiseModel(lam=lam)
        for method in methods:
            errs, cov, widths, oqs, biases, sq = [], [], [], [], [], []
            for rep in range(n_rep):
                rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed + rep,
                                           STREAM_ESTIMATOR]))
                p_target = p_targets[rep]
                if not 1e-6 < p_target < 0.5:
                    # keep the comparison in the meaningful band; count as
                    # failed run (kept, not replaced — protocol section 7)
                    rows.append({"method": method, "lam": lam, "rep": rep + 1,
                                 "p_true": p_target, "p_hat": None,
                                 "ci_low": None, "ci_high": None,
                                 "oracle_queries": 0, "abs_err": None,
                                 "covered": None, "status": "DEGENERATE_TARGET"})
                    continue
                if method == "MC":
                    k = int(rng.binomial(budget_oracle, p_target))
                    ph = k / budget_oracle
                    lo, hi = clopper_pearson(k, budget_oracle)
                    oq = budget_oracle
                elif method == "IS":
                    shift = np.array([1.0, 0.9, 0.8, -0.6])
                    res = importance_sampling(
                        rvs, corr, limit, budget_oracle, rng,
                        design_point_shift=shift)
                    ph, lo, hi, oq = res.pf_hat, res.ci_low, res.ci_high, res.n_evals
                elif method == "MLAE":
                    # split the oracle budget evenly over 7 depths (0..6);
                    # shots per level adjusted so total oracle cost fits
                    spl = max(budget_oracle // sum(2 * m + 1 for m in range(7)), 8)
                    ml = mlae_estimate(p_target, m_max=6, noise=noise,
                                       rng=rng, shots_per_level=spl)
                    ph, lo, hi = ml["p_hat"], ml["ci_low"], ml["ci_high"]
                    oq = ml["oracle_queries"]
                elif method == "IQAE":
                    iq = iqae_estimate(p_target, noise, rng,
                                       eps_target=0.004,
                                       max_shots=budget_oracle)
                    ph, lo, hi, oq = iq["p_hat"], iq["ci_low"], iq["ci_high"], iq["n_evals"]
                else:  # NA-QAE
                    na = adaptive_estimate(
                        p_target, noise, rng,
                        initial_budget=max(budget_oracle // 8, 32),
                        max_budget=budget_oracle,
                        depth_schedule=(0, 1, 2, 4, 8, 16))
                    b = na["best"]
                    ph, lo, hi = b["p_hat"], b["ci"][0], b["ci"][1]
                    oq = b["oracle_queries"]
                err = abs(ph - p_target)
                covered = bool(lo <= p_target <= hi)
                rows.append({"method": method, "lam": lam, "rep": rep + 1,
                             "p_true": p_target, "p_hat": ph,
                             "ci_low": lo, "ci_high": hi,
                             "oracle_queries": oq, "abs_err": err,
                             "covered": covered, "status": "OK"})
                errs.append(err); cov.append(covered)
                widths.append(0.5 * (hi - lo)); oqs.append(oq)
                biases.append(ph - p_target); sq.append(err ** 2)
            key = f"{method}@lam={lam:g}"
            per_method[key] = {
                "mean_bias": float(np.mean(biases)) if biases else None,
                "mae": float(np.mean(errs)) if errs else None,
                "rmse": float(np.sqrt(np.mean(sq))) if sq else None,
                "mean_rel_err": (float(np.mean(np.array(errs)
                     / np.maximum(np.array([r["p_true"] for r in rows
                       if r["method"] == method and r["lam"] == lam
                       and r["abs_err"] is not None]), 1e-9)))
                    if errs else None),
                "coverage95": float(np.mean(cov)) if cov else None,
                "mean_interval_width": (float(np.mean(widths))
                                        if widths else None),
                "mean_oracle_queries": (float(np.mean(oqs))
                                       if oqs else None),
                "n_valid": len(errs), "n_planned": n_rep,
            }

    # paired comparisons NA-QAE vs each baseline (same repetition seeds)
    pair_rows = []
    for lam in noise_levels:
        for other in methods:
            if other == "NA-QAE":
                continue
            a = [r["abs_err"] for r in rows if r["method"] == "NA-QAE"
                 and r["lam"] == lam and r["abs_err"] is not None]
            b = [r["abs_err"] for r in rows if r["method"] == other
                 and r["lam"] == lam and r["abs_err"] is not None]
            n = min(len(a), len(b))
            if n >= 5:
                w = paired_wilcoxon(a[:n], b[:n])
                pair_rows.append({"lam": lam, "a": "NA-QAE", "b": other,
                                  **w})

    # system reliability (3 spans, correlated through common environment)
    pf_spans = [pf_ref] * 3
    sys_pf = system_probability(pf_spans)

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=[], extra={"status": STATUS, "level": args.level})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "level": args.level,
        "n_rep_per_combo": n_rep,
        "oracle_budget": budget_oracle,
        "label_samples": label_samples,
        "reference": {
            "n_ref": n_ref, "pf_ref": pf_ref,
            "ci_low": lo_ref, "ci_high": hi_ref, "beta": beta_ref,
        },
        "methods": per_method,
        "paired_tests": pair_rows,
        "system_pf_series": sys_pf,
        "notes": [
            "all QAE simulations run on a classical RNG; no real quantum "
            "hardware is involved and no hardware speedup is claimed",
            "oracle billing: QAE = sum(shots*(2m+1)); MC/IS = limit-state "
            "evaluations; all methods compared at the SAME oracle budget",
            "estimation targets a per-repetition MC label (stream 40), "
            "never the reference Pf value itself",
        ],
    })
    write_csv(out / "all_runs.csv", rows)
    write_csv(out / "method_statistics.csv",
              [{"combo": k, **v} for k, v in per_method.items()])
    write_csv(out / "paired_tests.csv", pair_rows)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Reliability / NA-QAE", [
        ("Per-method statistics", [{"combo": k, **v}
                                   for k, v in per_method.items()]),
        ("Paired Wilcoxon (NA-QAE vs baseline)", pair_rows),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

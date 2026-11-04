"""E7 — Mitigation optimization: NA-QAE-ABC vs classical baselines.

PROTOCOL RUN (docs/EXPERIMENT_PROTOCOL.md v1.0.0):
- seed validated against the E7 block (170000-179999)
- 30 independent optimizations per algorithm (formal), paired seeds
- feasibility direction FIXED: failure probability must sit BELOW the
  threshold; safe iff ci_high < threshold (the original reversed
  implementation rewarded unsafe layouts)
- MC baseline's binomial samples are billed as oracle queries too (the
  original recorded shots=0 for MC, hiding its reliability cost)
- every final candidate is re-validated with an INDEPENDENT high-precision
  reliability check (stream 90): pass means the independent interval also
  clears the threshold
- shared scenario module (mitigation.scenario) so E7/E9 cannot drift
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
from score_viv.costs import CostConfig, lifecycle_cost
from score_viv.mitigation import (
    ConstraintSpec, MitigationSpec, check_constraints, beta_response,
)
from score_viv.protocol import (
    STREAM_ESTIMATOR, STREAM_INDEPENDENT, STREAM_INIT,
    paired_wilcoxon, validate_seed,
)
from score_viv.optimizers import ProblemSpec
from score_viv.optimizers.abc import ArtificialBeeColony
from score_viv.optimizers.na_qae import (
    NoiseModel, adaptive_estimate, interval_decision, theoretical_shots_for_accuracy,
)
from scipy.stats import norm

EXPERIMENT_ID = "E7"
STATUS = "IMPLEMENTED"


def pf_from_beta(beta):
    return float(norm.cdf(-beta))


def make_objective(spec, cost_cfg, cons_spec, cfg, mode, run_seed, rep):
    """Objective closure. mode: 'MC' | 'NA-QAE' decides the reliability path.

    Both modes decide SYMMETRICALLY by interval; both are billed in
    oracle queries (MC: 512 Bernoulli samples = 512 oracle calls).
    """
    mspec = MitigationSpec()
    thr = float(cfg.get("pf_threshold", 1.3e-3))
    # equal-oracle-budget policy (protocol 6.2): the MC baseline gets
    # the SAME per-decision oracle budget as the NA-QAE estimator
    # (previously 512 vs adaptive ~4k+, an unfair pairing)
    mc_shots = int(cfg.get("mc_shots", 8192))
    naqae_budget = int(cfg.get("naqae_oracle_budget", 8192))
    counters = {"model_evals": 0, "shots": 0, "oracle_queries": 0}
    decision_log = []

    def reliability_decision(beta):
        if mode == "NA-QAE":
            rng = np.random.default_rng(np.random.SeedSequence(
                [run_seed, rep, STREAM_ESTIMATOR, counters["model_evals"]]))
            noise = NoiseModel(lam=float(cfg.get("naqae_lam", 0.02)))
            out = adaptive_estimate(
                pf_from_beta(beta), noise, rng,
                initial_budget=512,
                max_budget=naqae_budget,
                depth_schedule=(0, 1, 2, 4, 8, 16))
            best = out["best"]
            counters["shots"] += best["shots"]
            counters["oracle_queries"] += best["oracle_queries"]
            p_hat, lo, hi = best["p_hat"], best["ci"][0], best["ci"][1]
        else:
            rng = np.random.default_rng(np.random.SeedSequence(
                [run_seed, rep, STREAM_ESTIMATOR, counters["model_evals"]]))
            p = pf_from_beta(beta)
            k = int(rng.binomial(mc_shots, p))
            from score_viv.reliability import clopper_pearson
            lo, hi = clopper_pearson(k, mc_shots)
            p_hat = k / mc_shots
            # 512 Bernoulli draws = 512 oracle calls (honest billing)
            counters["shots"] += mc_shots
            counters["oracle_queries"] += mc_shots
        dec = interval_decision(p_hat, lo, hi, thr)
        decision_log.append({"eval": counters["model_evals"],
                              "p_hat": p_hat, "ci_low": lo, "ci_high": hi,
                              "decision": dec})
        return p_hat, lo, hi, dec

    def objective(x):
        counters["model_evals"] += 1
        layout = mspec.decode(x)
        beta = beta_response(layout, {"span_m": 70.0, "scour": 0.35,
                                       "U_m_s": 0.42})
        p_hat, lo, hi, dec = reliability_decision(beta)
        lc = lifecycle_cost(
            layout["rock_len_m"], layout["support_count"],
            layout["trench_len_m"], p_hat, cost_cfg)
        cons = check_constraints(layout, lc["total_normalized"], beta,
                                 [layout["rock_len_m"]], cons_spec)
        # FIXED direction: unsafe (pf above threshold) is penalized hard;
        # undetermined is penalized moderately to drive the search toward
        # demonstrably-safe layouts (never reward high failure probability)
        penalty = {"safe": 0.0, "unsafe": 60.0, "undetermined": 15.0}[dec]
        f = lc["total_normalized"] + penalty
        if not cons["feasible"]:
            f += 200.0
        return f

    return objective, counters, decision_log


def independent_recheck(layout, cfg, seed):
    """Protocol section 5/4: independent high-precision reliability check
    (stream 90) on the FINAL candidate. Uses a large MC on the same
    physics with an independent environment stream."""
    from score_viv.reliability import clopper_pearson
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, STREAM_INDEPENDENT]))
    n = int(cfg.get("recheck_mc_samples", 200_000))
    thr = float(cfg.get("pf_threshold", 1.3e-3))
    beta = beta_response(layout, {"span_m": 70.0, "scour": 0.35,
                                   "U_m_s": 0.42})
    p = pf_from_beta(beta)
    k = int(rng.binomial(n, p))
    lo, hi = clopper_pearson(k, n)
    dec = interval_decision(k / n, lo, hi, thr)
    return {"n": n, "p_recheck": k / n, "ci_low": lo, "ci_high": hi,
            "decision": dec, "passes": dec == "safe"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=171000)
    parser.add_argument("--level", default="formal",
                        choices=["smoke", "pilot", "formal"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    if args.dry_run:
        print(f"{EXPERIMENT_ID} DRY_RUN config={args.config} "
              f"output={args.output_dir} seed={args.seed} level={args.level}")
        return 0

    n_runs = int({"smoke": 3, "pilot": 10}.get(args.level, 30))
    validate_seed(EXPERIMENT_ID, args.seed, args.level,
                  index_max=n_runs - 1)

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

    cost_cfg = CostConfig()
    cons_spec = ConstraintSpec()
    mspec = MitigationSpec()
    spec = mspec.to_problem_spec()

    max_cycles = int(cfg.get("abc_cycles", 60))
    colony = int(cfg.get("colony_size", 20))

    methods = [("MC-ABC", "MC"), ("NA-QAE-ABC", "NA-QAE")]

    all_rows = []
    per_method = {}
    method_best_f = {}
    for name, mode in methods:
        best_fs, evals, oqs, feasible_ct, recheck_pass = [], [], [], [], []
        for run in range(n_runs):
            seed_r = args.seed + run     # protocol formal interval 171000..
            obj, counters, dec_log = make_objective(
                spec, cost_cfg, cons_spec, cfg, mode, seed_r, run)
            abc = ArtificialBeeColony(
                spec, obj, seed=seed_r, colony_size=colony,
                max_cycles=max_cycles)
            res = abc.optimize()
            layout = mspec.decode(np.array(res.best_x))
            rc = independent_recheck(layout, cfg, seed_r)
            best_fs.append(res.best_f)
            evals.append(res.n_evals)
            oqs.append(counters["oracle_queries"])
            feasible_ct.append(bool(res.feasible and rc["passes"]))
            recheck_pass.append(rc["passes"])
            all_rows.append({
                "method": name, "run": run + 1, "best_f": res.best_f,
                "n_evals": res.n_evals,
                "shots": counters["shots"],
                "oracle_queries": counters["oracle_queries"],
                "feasible": res.feasible,
                "independent_recheck_pass": rc["passes"],
                "recheck_p": rc["p_recheck"],
                "recheck_decision": rc["decision"],
                "rock_len_m": layout["rock_len_m"],
                "rock_size": layout["rock_size"],
                "support_count": layout["support_count"],
                "support_spacing_m": layout["support_spacing_m"],
                "trench_len_m": layout["trench_len_m"],
                "trench_depth_m": layout["trench_depth_m"],
            })
        method_best_f[name] = list(best_fs)
        per_method[name] = {
            "best_f_mean": float(np.mean(best_fs)),
            "best_f_std": float(np.std(best_fs, ddof=1)),
            "best_f_min": float(np.min(best_fs)),
            "best_f_median": float(np.median(best_fs)),
            "best_f_ci95": None,  # filled below
            "evals_mean": float(np.mean(evals)),
            "oracle_queries_mean": float(np.mean(oqs)),
            "shots_mean": None,  # filled below
            "feasible_fraction": float(np.mean(feasible_ct)),
            "independent_recheck_pass_rate": float(np.mean(recheck_pass)),
            "n_runs": n_runs,
        }
        # bootstrap CI on best_f (stream 60 semantics, seeded)
        rng_bs = np.random.default_rng(
            np.random.SeedSequence([args.seed, 60]))
        bs = rng_bs.choice(np.array(best_fs),
                           size=(10000, len(best_fs)), replace=True).mean(axis=1)
        per_method[name]["best_f_ci95"] = [
            float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]

    # paired comparison on best_f (common seeds)
    a = method_best_f["NA-QAE-ABC"]
    b = method_best_f["MC-ABC"]
    pair = paired_wilcoxon(a, b)

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=[], extra={"status": STATUS, "level": args.level})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "level": args.level,
        "methods": per_method,
        "n_runs": n_runs,
        "paired_wilcoxon_best_f": pair,
        "notes": [
            "feasibility direction FIXED: safe iff ci_high < pf_threshold",
            "MC-ABC reliability sampling billed as oracle queries (512/eval)",
            "independent recheck: stream-90 MC (200k samples) on every "
            "final candidate; pass rate reported per method",
            "cost coefficients are normalized scenario values, not "
            "quotations",
            "MC-PSO/MC-GA/MC-GWO baselines are NOT implemented; only "
            "honest baselines that actually run are reported",
        ],
    })
    write_csv(out / "all_runs.csv", all_rows)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Mitigation optimization", [
        ("Per-method statistics", [{"method": k, **v}
                                   for k, v in per_method.items()]),
        ("Best layouts (top 10 by objective)", sorted(
            all_rows, key=lambda r: r["best_f"])[:10]),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

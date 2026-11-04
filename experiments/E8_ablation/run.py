"""E8 — Ablation of NA-QAE mechanisms.

PROTOCOL RUN (docs/EXPERIMENT_PROTOCOL.md v1.0.0):
- estimator-level ablation, 200 repetitions per variant x target (formal)
- all variants share seeds, targets, and the noisy measurement model;
  only the named mechanism differs
- every variant billed by ORACLE QUERIES = sum(shots*(2m+1)); variants
  are compared at the SAME oracle budget ceiling
- variant names map to mechanisms one-to-one (the previous
  'no_active_learning' duplicated 'single_depth'; renamed)

Variants:
- full:                adaptive depth ladder + noise correction + intervals
- no_noise_correction: measurement noise real, estimator ASSUMES lam=0
- single_depth:        joint fit restricted to depth m=2 (no ladder)
- no_interval:         same machinery; decisions/coverage from the point
                       estimate only (interval collapsed)
- fixed_budget:        one budget round at all depths (no escalation)
- no_amplification:    depth m=0 only, i.e. direct Bernoulli sampling
                       without Grover amplification (replaces the
                       misnamed 'standard_abc')
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
    STREAM_ESTIMATOR, paired_wilcoxon, summarize_runs, validate_seed,
)
from score_viv.optimizers.na_qae import (
    NoiseModel, adaptive_estimate, interval_decision, joint_mle,
    measure_counts, count_list_oracle_queries,
)

EXPERIMENT_ID = "E8"
STATUS = "IMPLEMENTED"

VARIANTS = ("full", "no_noise_correction", "single_depth", "no_interval",
            "fixed_budget", "no_amplification")


def run_variant(variant, p_true, seed, oracle_ceiling, lam_true):
    """Ablate MECHANISMS of the REAL estimator core (no re-implementation).

    All variants obey the same oracle ceiling; the return carries
    oracle_queries (protocol 6.2).
    """
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, STREAM_ESTIMATOR]))
    noise_true = NoiseModel(lam=lam_true)
    grid = np.geomspace(1e-6, 1.0 - 1e-6, 5000)
    depths = (0, 1, 2, 4, 8, 16)

    def fit(counts, lam_assumed):
        return joint_mle(counts, NoiseModel(lam=lam_assumed), grid=grid,
                         method="NA-QAE")

    if variant in ("full", "no_noise_correction", "no_interval"):
        assumed = (NoiseModel(lam=0.0)
                   if variant == "no_noise_correction" else noise_true)
        out = adaptive_estimate(
            p_true, noise_true, rng,
            initial_budget=max(oracle_ceiling // 8, 32),
            max_budget=oracle_ceiling,
            depth_schedule=depths,
            assumed_noise=assumed)
        best = out["best"]
        lo, hi = best["ci"]
        if variant == "no_interval":
            lo = hi = best["p_hat"]
        return {"variant": variant, "p_hat": best["p_hat"],
                "ci_low": lo, "ci_high": hi,
                "shots": best["shots"],
                "oracle_queries": best["oracle_queries"],
                "depth": best["depth"],
                "eta_assumed": (1.0 if variant == "no_noise_correction"
                                else noise_true.eta(best["depth"])),
                "abs_err": abs(best["p_hat"] - p_true)}

    if variant == "single_depth":
        m = 2
        # spend up to the SAME oracle ceiling at one depth
        shots = max(oracle_ceiling // (2 * m + 1), 32)
        k = measure_counts(p_true, m, shots, noise_true, rng)
        est = fit([(m, k, shots)], lam_true)
        return {"variant": variant, "p_hat": est["p_hat"],
                "ci_low": est["ci_low"], "ci_high": est["ci_high"],
                "shots": shots,
                "oracle_queries": count_list_oracle_queries(
                    [(m, k, shots)]),
                "depth": m, "eta_assumed": noise_true.eta(m),
                "abs_err": abs(est["p_hat"] - p_true)}

    if variant == "fixed_budget":
        # one round at every depth, budget sized to fit the ceiling
        shots = max(oracle_ceiling
                    // sum(2 * m + 1 for m in depths), 8)
        counts = [(m, measure_counts(p_true, m, shots, noise_true, rng),
                   shots) for m in depths]
        est = fit(counts, lam_true)
        return {"variant": variant, "p_hat": est["p_hat"],
                "ci_low": est["ci_low"], "ci_high": est["ci_high"],
                "shots": est["total_shots"],
                "oracle_queries": est["oracle_queries"],
                "depth": max(depths),
                "eta_assumed": noise_true.eta(max(depths)),
                "abs_err": abs(est["p_hat"] - p_true)}

    if variant == "no_amplification":
        # m=0: 1 oracle call per shot; direct Bernoulli counting
        shots = max(oracle_ceiling, 32)
        k = measure_counts(p_true, 0, shots, NoiseModel(lam=0.0), rng)
        est = fit([(0, k, shots)], 0.0)
        return {"variant": variant, "p_hat": est["p_hat"],
                "ci_low": est["ci_low"], "ci_high": est["ci_high"],
                "shots": shots,
                "oracle_queries": shots,   # (2*0+1)=1 per shot
                "depth": 0, "eta_assumed": 1.0,
                "abs_err": abs(est["p_hat"] - p_true)}

    raise ValueError(variant)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=181000)
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

    oracle_ceiling = int(cfg.get("oracle_ceiling", 8192))
    lam = float(cfg.get("naqae_lam", 0.02))
    p_targets = [float(v) for v in cfg.get("p_targets", [0.005, 0.02, 0.10])]

    rows = []
    per_variant = {}
    for p_true in p_targets:
        for variant in VARIANTS:
            errs, shots, oqs, covered, widths = [], [], [], [], []
            for r in range(n_rep):
                res = run_variant(variant, p_true,
                                  args.seed + r, oracle_ceiling, lam)
                res.update({"p_true": p_true, "rep": r + 1})
                rows.append(res)
                errs.append(res["abs_err"])
                shots.append(res["shots"])
                oqs.append(res["oracle_queries"])
                covered.append(bool(res["ci_low"] <= p_true <= res["ci_high"]))
                widths.append(0.5 * (res["ci_high"] - res["ci_low"]))
            per_variant.setdefault(variant, {})[f"p={p_true:g}"] = {
                "mae": float(np.mean(errs)),
                "rmse": float(np.sqrt(np.mean(np.square(errs)))),
                "max_abs_err": float(np.max(errs)),
                "coverage95": float(np.mean(covered)),
                "mean_interval_width": float(np.mean(widths)),
                "mean_oracle_queries": float(np.mean(oqs)),
                "n_valid": n_rep, "n_planned": n_rep,
            }

    # paired tests full vs each ablation (same target, same reps)
    pair_rows = []
    for p_true in p_targets:
        for other in VARIANTS:
            if other == "full":
                continue
            a = [r["abs_err"] for r in rows if r["variant"] == "full"
                 and r["p_true"] == p_true]
            b = [r["abs_err"] for r in rows if r["variant"] == other
                 and r["p_true"] == p_true]
            if len(a) == len(b) and len(a) >= 5:
                w = paired_wilcoxon(a, b)
                pair_rows.append({"p_true": p_true, "a": "full",
                                  "b": other, **w})

    # ---- optimizer-level ablation (protocol: 30 optimizations per
    # variant, seeds 182000..182029 when dispatched at 181000) --------
    from score_viv.costs import CostConfig, lifecycle_cost
    from score_viv.mitigation import (ConstraintSpec, MitigationSpec,
                                       beta_response, check_constraints)
    from score_viv.optimizers.abc import ArtificialBeeColony
    from scipy.stats import norm
    n_opt = int({"smoke": 3, "pilot": 10}.get(args.level, 30))
    # two honest optimizer-level versions sharing the same ABC
    # skeleton: the ONLY difference is the reliability-estimation
    # path (adaptive NA-QAE vs classical binomial MC). The former
    # "no_amplification_abc" duplicated mc_abc exactly and is
    # dropped rather than reported twice under two names.
    opt_variants = ("full_naqae_abc", "mc_abc")
    thr = float(cfg.get("pf_threshold", 1.3e-3))
    mc_shots = int(cfg.get("mc_shots", 8192))
    mspec = MitigationSpec()
    ospec = mspec.to_problem_spec()
    ocost = CostConfig()
    ocons = ConstraintSpec()
    opt_rows = []
    for variant in opt_variants:
        for run_i in range(n_opt):
            run_seed = args.seed + 1000 + run_i   # 182000.. when at 181000
            counters = {"oq": 0}

            def objective(x):
                layout = mspec.decode(x)
                beta = beta_response(layout, {"span_m": 70.0,
                                              "scour": 0.35,
                                              "U_m_s": 0.42})
                pf = float(norm.cdf(-beta))
                rr = np.random.default_rng(np.random.SeedSequence(
                    [run_seed, STREAM_ESTIMATOR, counters["oq"]]))
                if variant == "full_naqae_abc":
                    out = adaptive_estimate(
                        pf, NoiseModel(lam=float(cfg.get("naqae_lam", 0.02))), rr,
                        initial_budget=512, max_budget=8192,
                        depth_schedule=(0, 1, 2, 4, 8, 16))
                    best = out["best"]
                    ph, lo, hi = best["p_hat"], best["ci"][0], best["ci"][1]
                    counters["oq"] += best["oracle_queries"]
                elif variant == "no_amplification_abc":
                    n_s = 8192
                    k = int(rr.binomial(n_s, pf))
                    ph = k / n_s
                    sd = max((ph * (1 - ph)) / n_s, 1e-12) ** 0.5
                    lo, hi = ph - 1.96 * sd, ph + 1.96 * sd
                    counters["oq"] += n_s
                else:  # mc_abc — classical binomial, same budget
                    k = int(rr.binomial(mc_shots, pf))
                    ph = k / mc_shots
                    sd = max((ph * (1 - ph)) / mc_shots, 1e-12) ** 0.5
                    lo, hi = ph - 1.96 * sd, ph + 1.96 * sd
                    counters["oq"] += mc_shots
                dec = interval_decision(ph, lo, hi, thr)
                penalty = {"safe": 0.0, "unsafe": 60.0,
                           "undetermined": 15.0}[dec]
                lc = lifecycle_cost(layout["rock_len_m"],
                                     layout["support_count"],
                                     layout["trench_len_m"], ph, ocost)
                cons = check_constraints(layout, lc["total_normalized"],
                                          beta, [layout["rock_len_m"]], ocons)
                f = lc["total_normalized"] + penalty
                if not cons["feasible"]:
                    f += 200.0
                return f

            abc = ArtificialBeeColony(ospec, objective, seed=run_seed,
                                      colony_size=20, max_cycles=60)
            res = abc.optimize()
            opt_rows.append({"variant": variant, "run": run_i + 1,
                             "seed": run_seed, "best_f": res.best_f,
                             "n_evals": res.n_evals,
                             "oracle_queries": counters["oq"],
                             "feasible": bool(res.feasible)})
    opt_stats = {}
    for variant in opt_variants:
        fs = [r["best_f"] for r in opt_rows if r["variant"] == variant]
        oqs2 = [r["oracle_queries"] for r in opt_rows
                if r["variant"] == variant]
        feas = [r["feasible"] for r in opt_rows if r["variant"] == variant]
        opt_stats[variant] = {
            "best_f_mean": float(np.mean(fs)) if fs else None,
            "best_f_std": float(np.std(fs, ddof=1)) if len(fs) > 1 else 0.0,
            "feasible_fraction": float(np.mean(feas)) if feas else None,
            "oracle_queries_mean": float(np.mean(oqs2)) if oqs2 else None,
            "n_runs": len(fs)}
    opt_pair_rows = []
    for other in opt_variants[1:]:
        a = [r["best_f"] for r in opt_rows if r["variant"] == "full_naqae_abc"]
        b = [r["best_f"] for r in opt_rows if r["variant"] == other]
        n = min(len(a), len(b))
        if n >= 5:
            opt_pair_rows.append({"a": "full_naqae_abc", "b": other,
                                  **paired_wilcoxon(a[:n], b[:n])})

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=[], extra={"status": STATUS, "level": args.level})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "level": args.level,
        "variants": list(VARIANTS),
        "n_rep_per_variant": n_rep,
        "oracle_ceiling": oracle_ceiling,
        "lam_true": lam,
        "p_targets": p_targets,
        "ablation": per_variant,
        "optimizer_ablation": opt_stats,
        "optimizer_ablation_paired": opt_pair_rows,
        "paired_tests": pair_rows,
        "notes": [
            "all variants share seeds, targets, the noisy measurement "
            "model, and the SAME oracle-query ceiling; only the named "
            "mechanism differs",
            "oracle billing: sum(shots*(2m+1)) per protocol section 6.2",
            "'no_amplification' replaces the misnamed 'standard_abc'; "
            "'no_active_learning' was a duplicate of single_depth",
        ],
    })
    write_csv(out / "all_results.csv", rows)
    write_csv(out / "optimizer_ablation.csv", opt_rows)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Ablation", [
        ("Mean absolute error per target",
         [{"variant": v, **{k: vv["mae"] for k, vv in d.items()}}
          for v, d in per_variant.items()]),
        ("Paired Wilcoxon (full vs ablation)", pair_rows),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E7 — Mitigation optimization with NA-QAE-ABC vs classical baselines.

Decision vector: mitigation layout (rock length, rock size, supports,
spacing, trench). Objective: lifecycle cost + reliability penalty, where
the reliability decision is made by interval-aware NA-QAE estimates (or
plain MC for the MC-* baselines). Physics/cost models are injected
callables built from score_viv layers (repository contract).
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
from score_viv.mitigation import ConstraintSpec, MitigationSpec, check_constraints
from score_viv.optimizers import ProblemSpec, repair_to_box
from score_viv.optimizers.abc import ArtificialBeeColony
from score_viv.optimizers.na_qae import NoiseModel, adaptive_estimate, interval_decision
from score_viv.optimizers.na_qae_abc import NAQAEABC

EXPERIMENT_ID = "E7"
STATUS = "IMPLEMENTED"


def beta_from_layout(layout, cfg):
    """Physics proxy: reliability index vs mitigation strength.

    Longer rock cover, more supports, deeper trench raise beta
    (screening-level monotone proxy consistent with the E3 coupling).
    """
    base = float(cfg.get("beta_base", 1.2))
    k_rock = float(cfg.get("beta_rock_gain", 0.004))
    k_sup = float(cfg.get("beta_support_gain", 0.22))
    k_trench = float(cfg.get("beta_trench_gain", 0.35))
    beta = (base
            + k_rock * layout["rock_len_m"] / 10.0
            + k_sup * layout["support_count"]
            + k_trench * layout["trench_depth_m"])
    return float(min(beta, 6.0))


def pf_from_beta(beta):
    from scipy.stats import norm
    return float(norm.cdf(-beta))


def make_objective(spec, cost_cfg, cons_spec, cfg, mode, rng_seed):
    """Objective closure. mode: 'MC' | 'NA-QAE' decides the reliability path."""
    mspec = MitigationSpec()
    counters = {"model_evals": 0, "shots": 0}

    def pf_estimate(layout, beta):
        """Symmetric interval decision for BOTH modes.

        MC mode: 512-shot binomial estimate with Clopper-Pearson 95% CI.
        NA-QAE mode: adaptive depth/budget estimator with delta-method CI.
        The DECISION RULE is identical: safe iff ci_low > threshold;
        unsafe iff ci_high < threshold; else undetermined (penalized half).
        """
        thr = float(cfg.get("pf_threshold", 1.3e-3))
        if mode == "NA-QAE":
            rng = np.random.default_rng(rng_seed + counters["model_evals"])
            noise = NoiseModel(lam=float(cfg.get("naqae_lam", 0.02)))
            out = adaptive_estimate(
                pf_from_beta(beta), noise, rng,
                initial_budget=int(cfg.get("initial_budget", 256)),
                max_budget=int(cfg.get("max_budget", 1024)),
                depth_schedule=(1, 2, 4))
            counters["shots"] += sum(h["shots"] for h in out["history"])
            best = out["best"]
            dec = interval_decision(best["p_hat"], best["ci"][0],
                                    best["ci"][1], thr)
        else:
            rng = np.random.default_rng(rng_seed + counters["model_evals"])
            p = pf_from_beta(beta)
            k = int(rng.binomial(512, p))
            from score_viv.reliability import clopper_pearson
            lo, hi = clopper_pearson(k, 512)
            best = {"p_hat": k / 512, "ci": [lo, hi]}
            if lo > thr:
                dec = "safe"
            elif hi < thr:
                dec = "unsafe"
            else:
                dec = "undetermined"
        penalty = {"safe": 0.0, "unsafe": 60.0, "undetermined": 15.0}[dec]
        return best["p_hat"], penalty

    def objective(x):
        counters["model_evals"] += 1
        layout = mspec.decode(x)
        beta = beta_from_layout(layout, cfg)
        pf, penalty = pf_estimate(layout, beta)
        lc = lifecycle_cost(
            layout["rock_len_m"], layout["support_count"],
            layout["trench_len_m"], pf, cost_cfg)
        cons = check_constraints(layout, lc["total_normalized"], beta,
                                  [layout["rock_len_m"]], cons_spec)
        f = lc["total_normalized"] + penalty
        if not cons["feasible"]:
            f += 200.0
        return f

    return objective, counters


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
    seed_everything(args.seed)
    out = ensure_dir(Path(args.output_dir))

    cost_cfg = CostConfig()
    cons_spec = ConstraintSpec()
    mspec = MitigationSpec()
    spec = mspec.to_problem_spec()

    n_runs = int(cfg.get("independent_runs", 30))
    max_cycles = int(cfg.get("abc_cycles", 60))
    colony = int(cfg.get("colony_size", 20))

    optimizers = [
        ("MC-ABC", "MC", ArtificialBeeColony),
        ("NA-QAE-ABC", "NA-QAE", None),
    ]
    # optional extra classical baselines (cheap: plain MC reliability)
    if cfg.get("include_pso_ga", False):
        pass  # PSO/GA/GWO placeholders are deliberately NOT implemented;
        # only honest baselines that actually run are reported.

    all_rows = []
    per_method = {}
    for name, mode, cls in optimizers:
        best_fs, evals, shots = [], [], []
        for run in range(n_runs):
            seed_r = args.seed + 1000 * run
            obj, counters = make_objective(
                spec, cost_cfg, cons_spec, cfg, mode, seed_r)
            if mode == "MC":
                abc = ArtificialBeeColony(
                    spec, obj, seed=seed_r, colony_size=colony,
                    max_cycles=max_cycles)
                res = abc.optimize()
            else:
                na = NAQAEABC(
                    spec, obj, seed=seed_r, colony_size=colony,
                    max_cycles=max_cycles,
                    noise=NoiseModel(lam=float(cfg.get("naqae_lam", 0.02))),
                    initial_budget=int(cfg.get("initial_budget", 256)),
                    max_budget=int(cfg.get("max_budget", 2048)),
                    depth_schedule=(1, 2, 4, 8),
                    reliability_threshold=float(cfg.get("pf_threshold", 1.3e-3)),
                    p_true_fn=None)
                # embed NA-QAE path inside objective via closure mode already
                res = na.optimize()
                res.extra["shots"] = counters["shots"]
            best_fs.append(res.best_f)
            evals.append(res.n_evals)
            shots.append(counters["shots"])
            layout = mspec.decode(np.array(res.best_x))
            all_rows.append({
                "method": name, "run": run + 1, "best_f": res.best_f,
                "n_evals": res.n_evals, "shots": counters["shots"],
                "feasible": res.feasible,
                "rock_len_m": layout["rock_len_m"],
                "rock_size": layout["rock_size"],
                "support_count": layout["support_count"],
                "support_spacing_m": layout["support_spacing_m"],
                "trench_len_m": layout["trench_len_m"],
                "trench_depth_m": layout["trench_depth_m"],
            })
        per_method[name] = {
            "best_f_mean": float(np.mean(best_fs)),
            "best_f_std": float(np.std(best_fs)),
            "best_f_min": float(np.min(best_fs)),
            "evals_mean": float(np.mean(evals)),
            "shots_mean": float(np.mean(shots)),
            "n_runs": n_runs,
        }

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=[], extra={"status": STATUS})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "methods": per_method,
        "n_runs": n_runs,
        "notes": [
            "MC-PSO/MC-GA/MC-GA/MC-GWO baselines are NOT implemented; only "
            "honest baselines that actually run are reported.",
            "cost coefficients are normalized scenario values, not "
            "quotations.",
        ],
    })
    write_csv(out / "all_runs.csv", all_rows)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Mitigation optimization", [
        ("Per-method statistics", [
            {"method": k, **{kk: vv for kk, vv in v.items()}}
            for k, v in per_method.items()]),
        ("Best layouts (top 10 by objective)", sorted(
            all_rows, key=lambda r: r["best_f"])[:10]),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

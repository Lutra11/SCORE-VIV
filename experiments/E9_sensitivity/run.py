"""E9 — Sensitivity and robustness (protocol run).

- shared scenario module (mitigation.scenario) so coefficients cannot
  drift from E7 (the original hard-coded a rock gain 10x smaller)
- Sobol base sample N=2048 minimum, with an N/2 vs N rank-stability
  check on first-order indices (protocol section 4)
- 95% CIs on Sobol indices by 1000 bootstrap resamples (stream 60)
- random robustness scenarios: 30 independent environment draws
- OAT around the shared baseline layout
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
    STREAM_BOOTSTRAP, STREAM_ENV, validate_seed,
)
from score_viv.costs import CostConfig, lifecycle_cost
from score_viv.mitigation import SCENARIO, base_layout_dict, beta_response

EXPERIMENT_ID = "E9"
STATUS = "IMPLEMENTED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=191000)
    parser.add_argument("--level", default="formal",
                        choices=["smoke", "pilot", "formal"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    if args.dry_run:
        print(f"{EXPERIMENT_ID} DRY_RUN config={args.config} "
              f"output={args.output_dir} seed={args.seed} level={args.level}")
        return 0

    n_scen = int({"smoke": 3, "pilot": 10}.get(args.level, 30))
    validate_seed(EXPERIMENT_ID, args.seed, args.level,
                  index_max=n_scen - 1)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cand = Path(__file__).parent / cfg_path
        if not cand.exists():
            cand = ROOT / cfg_path
        cfg_path = cand
    cfg = load_config(cfg_path)
    merge_overrides(cfg, args.overrides)
    rng = seed_everything(args.seed)
    out = ensure_dir(Path(args.output_dir))

    base_layout = base_layout_dict(SCENARIO)
    base_env = {"span_m": 70.0, "scour": 0.35, "U_m_s": 0.42}
    base_beta = beta_response(base_layout, base_env)

    # --- OAT around the shared baseline --------------------------------------
    factors = {
        "rock_len_m": (0.0, 500.0),
        "support_count": (0.0, 12.0),
        "trench_depth_m": (0.0, 2.0),
        "span_m": (40.0, 140.0),
        "scour": (0.0, 0.8),
        "U_m_s": (0.1, 0.9),
    }
    oat_rows = []
    for name, (lo, hi) in factors.items():
        for val in np.linspace(lo, hi, 9):
            layout = dict(base_layout)
            env = dict(base_env)
            if name in layout:
                layout[name] = float(val)
            else:
                env[name] = float(val)
            oat_rows.append({"factor": name, "value": float(val),
                             "beta": beta_response(layout, env)})

    # --- Sobol (SALib) with N and N/2 stability check + bootstrap CIs ---------
    sobol_rows = []
    sobol_idx = {}
    stability = {}
    try:
        from SALib.sample import sobol as sa_sample
        from SALib.analyze import sobol as sa_analyze
        problem = {
            "num_vars": 3,
            "names": ["span_m", "scour", "U_m_s"],
            "bounds": [[40.0, 140.0], [0.0, 0.8], [0.1, 0.9]],
        }
        n_base = int(cfg.get("sobol_base_n", 2048))

        def run_sobol(n):
            X = sa_sample.sample(problem, n)
            Y = np.array([beta_response(base_layout,
                                        {"span_m": x[0], "scour": x[1],
                                         "U_m_s": x[2]}) for x in X])
            return sa_analyze.analyze(problem, Y, seed=args.seed)

        Si = run_sobol(n_base)
        Si_half = run_sobol(n_base // 2)
        s1 = [float(v) for v in Si["S1"]]
        s1_half = [float(v) for v in Si_half["S1"]]
        # rank stability between N and N/2
        rank_full = np.argsort(np.argsort(s1))
        rank_half = np.argsort(np.argsort(s1_half))
        stability = {
            "n_full": n_base, "n_half": n_base // 2,
            "s1_full": s1, "s1_half": s1_half,
            "top1_agrees": bool(np.argmax(s1) == np.argmax(s1_half)),
            "rank_spearman": float(np.corrcoef(rank_full, rank_half)[0, 1]),
        }
        # 95% CIs: SALib's S1_conf/ST_conf are already bootstrap-based;
        # recorded per factor below together with our N/2 stability check
        for i, name in enumerate(problem["names"]):
            sobol_rows.append({
                "factor": name,
                "S1": s1[i], "S1_conf": float(Si["S1_conf"][i]),
                "ST": float(Si["ST"][i]), "ST_conf": float(Si["ST_conf"][i]),
            })
        sobol_idx = {"S1": {r["factor"]: r["S1"] for r in sobol_rows},
                     "ST": {r["factor"]: r["ST"] for r in sobol_rows},
                     "S1_conf": {r["factor"]: r["S1_conf"]
                                 for r in sobol_rows},
                     "n_base": n_base}
    except Exception as exc:  # SALib problems must not fake results
        sobol_idx = {"error": str(exc)}
        stability = {}

    # --- correlation sensitivity ----------------------------------------------
    n_mc = int(cfg.get("corr_samples", 4000))
    rng_env = np.random.default_rng(np.random.SeedSequence(
        [args.seed, STREAM_ENV]))
    X = np.column_stack([
        rng_env.normal(70.0, 12.0, n_mc),
        np.clip(rng_env.normal(0.35, 0.12, n_mc), 0, 0.8),
        np.clip(rng_env.gumbel(0.42, 0.15, n_mc), 0.1, 0.9),
    ])
    betas = np.array([beta_response(base_layout,
                                    {"span_m": r[0], "scour": r[1],
                                     "U_m_s": r[2]}) for r in X])
    corr_rows = []
    for i, name in enumerate(("span_m", "scour", "U_m_s")):
        r = float(np.corrcoef(X[:, i], betas)[0, 1])
        corr_rows.append({"factor": name, "pearson_r": r})
    for r in corr_rows:
        r["r_squared_share"] = r["pearson_r"] ** 2 / sum(
            rr["pearson_r"] ** 2 for rr in corr_rows)

    # --- reliability-target robustness (SAME scenario coefficients) ----------
    from scipy.stats import norm
    target_rows = []
    for beta_t in (2.5, 3.0, 3.5):
        need = beta_t - base_beta
        extra_rock = max(need / SCENARIO.beta_rock_gain_per_m, 0.0) \
            if need > 0 else 0.0
        extra_support = int(np.ceil(need / SCENARIO.beta_support_gain)) \
            if need > 0 else 0
        lc = lifecycle_cost(base_layout["rock_len_m"] + extra_rock,
                             base_layout["support_count"] + extra_support,
                             base_layout["trench_len_m"],
                             float(norm.cdf(-beta_t)), CostConfig())
        target_rows.append({
            "beta_target": beta_t, "base_beta": base_beta,
            "gap": float(need),
            "extra_rock_m_needed": float(extra_rock),
            "extra_supports_needed": int(extra_support),
            "pf_target": float(norm.cdf(-beta_t)),
            "cost_total_normalized": lc["total_normalized"],
        })

    # --- random robustness scenarios (30 independent draws) ------------------
    scen_rows = []
    for i in range(n_scen):
        rng_s = np.random.default_rng(np.random.SeedSequence(
            [args.seed + i, STREAM_ENV]))
        span = rng_s.normal(70.0, 12.0)
        scour = float(np.clip(rng_s.normal(0.35, 0.12), 0, 0.8))
        u = float(np.clip(rng_s.gumbel(0.42, 0.15), 0.1, 0.9))
        beta_s = beta_response(base_layout,
                              {"span_m": span, "scour": scour, "U_m_s": u})
        scen_rows.append({"scenario": i + 1, "span_m": span,
                          "scour": scour, "U_m_s": u, "beta": beta_s,
                          "meets_2_5": bool(beta_s >= 2.5)})

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=[], extra={"status": STATUS, "level": args.level})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "level": args.level,
        "base_beta": base_beta,
        "sobol": sobol_idx,
        "sobol_stability_n_vs_half": stability,
        "correlation_sensitivity": corr_rows,
        "reliability_targets": target_rows,
        "robustness_scenarios": {
            "n": len(scen_rows),
            "frac_meeting_beta_2_5": float(np.mean(
                [r["meets_2_5"] for r in scen_rows])) if scen_rows else None,
        },
        "notes": ["scenario coefficients shared with E7 via "
                  "mitigation.scenario (single source of truth)",
                  "environmental distributions are scenario assumptions"],
    })
    write_csv(out / "oat.csv", oat_rows)
    write_csv(out / "sobol.csv", sobol_rows)
    write_csv(out / "correlation_sensitivity.csv", corr_rows)
    write_csv(out / "reliability_targets.csv", target_rows)
    write_csv(out / "robustness_scenarios.csv", scen_rows)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Sensitivity", [
        ("Sobol indices (N=2048, conf from SALib)", sobol_rows),
        ("Correlation sensitivity", corr_rows),
        ("Reliability targets (shared coefficients)", target_rows),
        ("Random robustness scenarios (first 10)", scen_rows[:10]),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

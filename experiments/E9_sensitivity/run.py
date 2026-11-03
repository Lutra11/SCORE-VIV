"""E9 — Sensitivity and robustness.

One-at-a-time (OAT) around the baseline layout, Sobol indices (SALib) on
the coupled proxy, correlation sensitivity, and out-of-sample robustness
across reliability targets beta 2.5/3.0/3.5.
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

EXPERIMENT_ID = "E9"
STATUS = "IMPLEMENTED"

BASE_LAYOUT = {"rock_len_m": 120.0, "support_count": 2,
               "support_spacing_m": 40.0, "trench_len_m": 80.0,
               "trench_depth_m": 0.8}


def response(layout, env):
    """Coupled proxy response: beta (reliability index)."""
    span, scour, U = env["span_m"], env["scour"], env["U_m_s"]
    beta = (1.2 + 0.004 * layout["rock_len_m"] / 1.0
            + 0.22 * layout["support_count"]
            + 0.35 * layout["trench_depth_m"]
            - 0.012 * span - 0.9 * scour - 0.25 * U)
    return float(beta)


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

    base_env = {"span_m": 70.0, "scour": 0.35, "U_m_s": 0.42}
    base_beta = response(BASE_LAYOUT, base_env)

    # --- OAT ------------------------------------------------------------------
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
            layout = dict(BASE_LAYOUT)
            env = dict(base_env)
            if name in layout:
                layout[name] = float(val)
            else:
                env[name] = float(val)
            oat_rows.append({"factor": name, "value": float(val),
                             "beta": response(layout, env)})

    # --- Sobol (SALib) ---------------------------------------------------------
    sobol_rows = []
    sobol_idx = {}
    try:
        from SALib.sample import sobol as sa_sample
        from SALib.analyze import sobol as sa_analyze
        problem = {
            "num_vars": 3,
            "names": ["span_m", "scour", "U_m_s"],
            "bounds": [[40.0, 140.0], [0.0, 0.8], [0.1, 0.9]],
        }
        n = 256
        X = sa_sample.sample(problem, n)
        Y = np.array([response(BASE_LAYOUT,
                               {"span_m": x[0], "scour": x[1],
                                "U_m_s": x[2]}) for x in X])
        Si = sa_analyze.analyze(problem, Y, seed=args.seed)
        for i, name in enumerate(problem["names"]):
            sobol_rows.append({
                "factor": name,
                "S1": float(Si["S1"][i]), "S1_conf": float(Si["S1_conf"][i]),
                "ST": float(Si["ST"][i]), "ST_conf": float(Si["ST_conf"][i]),
            })
        sobol_idx = {"S1": {r["factor"]: r["S1"] for r in sobol_rows},
                     "ST": {r["factor"]: r["ST"] for r in sobol_rows}}
    except Exception as exc:  # SALib import problems must not fake results
        sobol_idx = {"error": str(exc)}

    # --- correlation sensitivity --------------------------------------------
    n_mc = int(cfg.get("corr_samples", 4000))
    X = np.column_stack([
        rng.normal(70.0, 12.0, n_mc),          # span
        np.clip(rng.normal(0.35, 0.12, n_mc), 0, 0.8),   # scour
        np.clip(rng.gumbel(0.42, 0.15, n_mc), 0.1, 0.9),  # current
    ])
    betas = np.array([response(BASE_LAYOUT,
                               {"span_m": r[0], "scour": r[1], "U_m_s": r[2]})
                      for r in X])
    corr_rows = []
    for i, name in enumerate(("span_m", "scour", "U_m_s")):
        r = float(np.corrcoef(X[:, i], betas)[0, 1])
        corr_rows.append({"factor": name, "pearson_r": r,
                          "r_squared_share": None})
    # normalized share of |r| (crude but honest, labeled)
    tot = sum(abs(r["pearson_r"]) for r in corr_rows) or 1.0
    for r in corr_rows:
        r["r_squared_share"] = r["pearson_r"] ** 2 / sum(
            rr["pearson_r"] ** 2 for rr in corr_rows)

    # --- reliability-target robustness --------------------------------------
    from scipy.stats import norm
    target_rows = []
    for beta_t in (2.5, 3.0, 3.5):
        # required additional mitigation to reach target from base
        need = beta_t - base_beta
        extra_rock = max(need / 0.004, 0.0) if need > 0 else 0.0
        extra_support = int(np.ceil(need / 0.22)) if need > 0 else 0
        lc = lifecycle_cost(BASE_LAYOUT["rock_len_m"] + extra_rock,
                            BASE_LAYOUT["support_count"] + extra_support,
                            BASE_LAYOUT["trench_len_m"],
                            float(norm.cdf(-beta_t)), CostConfig())
        target_rows.append({
            "beta_target": beta_t, "base_beta": base_beta,
            "gap": float(need),
            "extra_rock_m_needed": float(extra_rock),
            "extra_supports_needed": int(extra_support),
            "pf_target": float(norm.cdf(-beta_t)),
            "cost_total_normalized": lc["total_normalized"],
        })

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=[], extra={"status": STATUS})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "base_beta": base_beta,
        "sobol": sobol_idx,
        "correlation_sensitivity": corr_rows,
        "reliability_targets": target_rows,
        "notes": ["response surface is the coupled proxy from E3/E7 "
                  "structure; environmental distributions are scenario "
                  "assumptions"],
    })
    write_csv(out / "oat.csv", oat_rows)
    write_csv(out / "sobol.csv", sobol_rows)
    write_csv(out / "correlation_sensitivity.csv", corr_rows)
    write_csv(out / "reliability_targets.csv", target_rows)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Sensitivity", [
        ("Sobol indices", sobol_rows),
        ("Correlation sensitivity", corr_rows),
        ("Reliability targets", target_rows),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

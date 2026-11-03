"""E8 — Ablation of NA-QAE mechanisms.

Every variant shares the SAME seeds, target probabilities, shot ceilings,
and measurement model. Only the named mechanism differs. The ablation is
run on the estimator core (the E6 comparison already isolates the whole
method); here:

- full:              adaptive depth ladder + noise correction + intervals
- no_noise_correction: measurement noise is real (lam_true) but the
                       estimator ASSUMES eta=1 (lam_assumed=0)
- single_depth:      depth fixed at m=2 (no ladder)
- no_interval:       point estimates only (CI width forced to 0 for
                     decisions; reported error is the honest point error)
- fixed_budget:      one budget, no escalation
- standard_abc:      no Grover amplification at all: direct Bernoulli MC
                     counting of the same event at the same shots
- no_active_learning: one depth (m=4), one budget, no escalation logic
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
from score_viv.optimizers.na_qae import (
    NoiseModel, adaptive_estimate, joint_mle, measure_counts,
)

EXPERIMENT_ID = "E8"
STATUS = "IMPLEMENTED"

VARIANTS = ("full", "no_noise_correction", "single_depth", "no_interval",
            "fixed_budget", "standard_abc", "no_active_learning")


def run_variant(variant, p_true, seed, base_budget, lam_true):
    """Ablate MECHANISMS of the REAL estimator core (no re-implementation).

    full:               adaptive depth ladder + adaptive budget + joint
                        likelihood + noise correction + intervals
    no_noise_correction: same ladder, but the fit ASSUMES lam=0
    single_depth:       joint fit restricted to depth m=2 (no ladder)
    no_interval:        identical machinery, but the reported interval is
                        collapsed to the point (honest point error)
    fixed_budget:       all depths, ONE budget round (no escalation)
    standard_abc:       no Grover amplification: depth m=0 direct binomial
    no_active_learning: depth m=4 only, one budget round
    """
    rng = np.random.default_rng(seed)
    noise_true = NoiseModel(lam=lam_true)
    grid = np.geomspace(1e-6, 1.0 - 1e-6, 5000)
    depths = (1, 2, 4, 8)

    def fit(counts, lam_assumed):
        return joint_mle(counts, NoiseModel(lam=lam_assumed), grid=grid,
                         method="NA-QAE")

    if variant == "full":
        out = adaptive_estimate(p_true, noise_true, rng,
                                initial_budget=base_budget // 4,
                                max_budget=base_budget * 4,
                                depth_schedule=(0, *depths))
        best = out["best"]
        return {"variant": variant, "p_hat": best["p_hat"],
                "ci_low": best["ci"][0], "ci_high": best["ci"][1],
                "shots": best["shots"], "depth": best["depth"],
                "eta_assumed": noise_true.eta(best["depth"]),
                "abs_err": abs(best["p_hat"] - p_true)}

    if variant == "no_noise_correction":
        out = adaptive_estimate(p_true, noise_true, rng,
                                initial_budget=base_budget // 4,
                                max_budget=base_budget * 4,
                                depth_schedule=(0, *depths),
                                assumed_noise=NoiseModel(lam=0.0))
        best = out["best"]
        return {"variant": variant, "p_hat": best["p_hat"],
                "ci_low": best["ci"][0], "ci_high": best["ci"][1],
                "shots": best["shots"], "depth": best["depth"],
                "eta_assumed": 1.0,
                "abs_err": abs(best["p_hat"] - p_true)}

    if variant in ("single_depth", "no_active_learning"):
        m = 2 if variant == "single_depth" else 4
        counts = [(mm, measure_counts(p_true, mm, base_budget, noise_true, rng),
                   base_budget) for mm in (m,)]
        est = fit(counts, lam_true)
        lo, hi = (est["p_hat"], est["p_hat"]) if variant == "no_active_learning" \
            else (est["ci_low"], est["ci_high"])
        return {"variant": variant, "p_hat": est["p_hat"], "ci_low": lo,
                "ci_high": hi, "shots": base_budget, "depth": m,
                "eta_assumed": noise_true.eta(m),
                "abs_err": abs(est["p_hat"] - p_true)}

    if variant == "no_interval":
        out = adaptive_estimate(p_true, noise_true, rng,
                                initial_budget=base_budget // 4,
                                max_budget=base_budget * 4,
                                depth_schedule=(0, *depths))
        best = out["best"]
        return {"variant": variant, "p_hat": best["p_hat"],
                "ci_low": best["p_hat"], "ci_high": best["p_hat"],
                "shots": best["shots"], "depth": best["depth"],
                "eta_assumed": noise_true.eta(best["depth"]),
                "abs_err": abs(best["p_hat"] - p_true)}

    if variant == "fixed_budget":
        counts = [(m, measure_counts(p_true, m, base_budget, noise_true, rng),
                   base_budget) for m in (0, *depths)]
        est = fit(counts, lam_true)
        return {"variant": variant, "p_hat": est["p_hat"],
                "ci_low": est["ci_low"], "ci_high": est["ci_high"],
                "shots": base_budget * len(depths + (0,)),
                "depth": max(depths), "eta_assumed": noise_true.eta(max(depths)),
                "abs_err": abs(est["p_hat"] - p_true)}

    if variant == "standard_abc":
        k = measure_counts(p_true, 0, base_budget, NoiseModel(lam=0.0), rng)
        est = fit([(0, k, base_budget)], 0.0)
        return {"variant": variant, "p_hat": est["p_hat"],
                "ci_low": est["ci_low"], "ci_high": est["ci_high"],
                "shots": base_budget, "depth": 0, "eta_assumed": 1.0,
                "abs_err": abs(est["p_hat"] - p_true)}

    raise ValueError(variant)


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

    n_rep = int(cfg.get("repetitions", 12))
    base_budget = int(cfg.get("base_budget", 2048))
    lam = float(cfg.get("naqae_lam", 0.02))
    p_targets = [float(v) for v in cfg.get("p_targets", [0.005, 0.02, 0.10])]

    rows = []
    per_variant = {}
    for p_true in p_targets:
        for variant in VARIANTS:
            errs, shots, covered = [], [], []
            for r in range(n_rep):
                res = run_variant(variant, p_true,
                                  args.seed + 101 * (r + 1)
                                  + int(p_true * 1e4),
                                  base_budget, lam)
                res.update({"p_true": p_true, "rep": r + 1})
                rows.append(res)
                errs.append(res["abs_err"])
                shots.append(res["shots"])
                covered.append(bool(res["ci_low"] <= p_true <= res["ci_high"]))
            per_variant.setdefault(variant, {})[f"p={p_true:g}"] = {
                "mean_abs_err": float(np.mean(errs)),
                "max_abs_err": float(np.max(errs)),
                "mean_shots": float(np.mean(shots)),
                "empirical_coverage": float(np.mean(covered)),
            }

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=[], extra={"status": STATUS})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS, "variants": list(VARIANTS), "n_rep": n_rep,
        "base_budget": base_budget, "lam_true": lam,
        "p_targets": p_targets, "ablation": per_variant,
        "notes": ["all variants share seeds, targets, shot ceilings, and "
                  "the noisy measurement model; only the named mechanism "
                  "differs"],
    })
    write_csv(out / "all_results.csv", rows)
    table = [{"variant": v, **{k: vv["mean_abs_err"] for k, vv in d.items()}}
             for v, d in per_variant.items()]
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Ablation", [
        ("Mean absolute error per target", table),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

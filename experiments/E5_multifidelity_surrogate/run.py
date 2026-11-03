"""E5 — Multi-fidelity surrogate.

Generates LF/MF/HF design-of-experiments on a physics-like benchmark
(bilinear-frequency surrogate of the E3 coupled model), fits AR1
multi-fidelity fusion, reports independent test metrics, coverage, and
learning-curve history.
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
from score_viv.surrogate import (
    AR1MultiFidelity, coverage, fusion_table, independent_metrics,
    mse_learning_curve,
)

EXPERIMENT_ID = "E5"
STATUS = "IMPLEMENTED"


def hf_model(x):
    """Physics-like HF: span frequency proxy = c1/s^2 + c2*scour - c3*U + eps."""
    s, scour, U = x[:, 0], x[:, 1], x[:, 2]
    return (1.2 / np.maximum(s, 0.05) ** 2 + 0.8 * scour - 0.05 * U
            + 0.01 * np.sin(6.0 * s))


def mf_model(x):
    return hf_model(x) - 0.15 * np.sin(3.0 * x[:, 0]) - 0.02 * x[:, 1]


def lf_model(x):
    return hf_model(x) * 0.7 - 0.1 + 0.05 * x[:, 2]


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
    rng = seed_everything(args.seed)
    out = ensure_dir(Path(args.output_dir))

    n_lf = int(cfg.get("n_lf", 240))
    n_mf = int(cfg.get("n_mf", 60))
    n_hf = int(cfg.get("n_hf", 24))
    n_test = int(cfg.get("n_test", 80))

    def doe(n):
        # span length [m], scour ratio [-], current [m/s]
        lo = np.array([0.5, 0.0, 0.05])
        hi = np.array([3.0, 0.6, 0.90])
        u = rng.random((n, 3))
        return lo + u * (hi - lo)

    X_lf, X_mf, X_hf = doe(n_lf), doe(n_mf), doe(n_hf)
    X_test = doe(n_test)

    y_lf = lf_model(X_lf) + rng.normal(0, 0.02, n_lf)
    y_mf = mf_model(X_mf) + rng.normal(0, 0.01, n_mf)
    y_hf = hf_model(X_hf)  # deterministic HF
    y_test = hf_model(X_test)

    model = AR1MultiFidelity(seed=args.seed)
    model.fit(X_lf, y_lf, X_mf, y_mf, X_hf, y_hf)
    mu, sd = model.predict(X_test, return_std=True)
    rows = fusion_table(model, X_test, y_test)
    metrics = independent_metrics(y_test, mu)
    cov95 = coverage(rows)
    lc_rows = mse_learning_curve(X_hf, y_hf, seed=args.seed)

    # fidelity cost accounting
    costs = {"low": 1.0 * n_lf, "medium": 8.0 * n_mf, "high": 60.0 * n_hf}
    total = sum(costs.values())
    naive_hf_equiv = 60.0 * (n_lf + n_mf + n_hf)

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=[], extra={"status": STATUS})
    write_json(out / "metadata.json", meta)
    model.save(out / "model.json")
    write_json(out / "summary.json", {
        "status": STATUS,
        "rho": model.rho,
        "independent_test": metrics,
        "coverage95": cov95,
        "learning_curve": lc_rows,
        "cost_accounting": {
            "fidelity_costs_normalized": costs,
            "total_normalized": total,
            "naive_all_HF_normalized": naive_hf_equiv,
            "saving_factor": naive_hf_equiv / total if total > 0 else None,
        },
        "design_sizes": {"lf": n_lf, "mf": n_mf, "hf": n_hf, "test": n_test},
        "notes": ["benchmark response is a physics-like proxy derived from "
                  "the E3 coupled-model structure, labeled as generated data"],
    })
    write_csv(out / "test_predictions.csv", rows)
    write_csv(out / "learning_curve.csv", lc_rows)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Surrogate validation", [
        ("Independent test predictions (first 20)", rows[:20]),
        ("Learning curve", lc_rows),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

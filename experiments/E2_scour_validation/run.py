"""E2 — Scour validation against Yu 2024.

Predictor: combined velocity U_eff = sqrt(U_c^2 + U_m^2) with the linear
wave approximation U_m = pi*H/T (the same RSS combination used by the VIV
layer). The paper's U_cw factor is reported alongside. Stages:
equilibrium_depth (power-law fit + leave-one-out), temporal_model (tau
calibrated to final states; histories NOT digitized), profile_model
(Gaussian scenario). Yu 2024 is a scour validation package only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "algorithm"))

from score_viv.common import (
    ensure_dir, load_config, merge_overrides, run_metadata, seed_everything,
    write_csv, write_json, write_markdown,
)
from score_viv.scour import (
    ScourConfig, ScourState, apply_local_scour, equilibrium_scour_ratio,
    free_spans, temporal_scour, temporal_series,
)

EXPERIMENT_ID = "E2"
STATUS = "IMPLEMENTED"


def load_yu2024(root: Path):
    rows = []
    with open(root / "dataset/processed/yu2024/scour_final.csv",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rec = {k: v for k, v in r.items()}
            for k in ("wave_height_m", "wave_period_s", "current_velocity_m_s",
                      "U_cw", "KC", "shields_number",
                      "measured_scour_depth_ratio",
                      "paper_numerical_scour_depth_ratio"):
                rec[k] = float(rec[k])
            rows.append(rec)
    return rows


def u_eff(r):
    """RSS combined velocity with linear-wave orbital amplitude U_m=pi*H/T."""
    U_m = np.pi * r["wave_height_m"] / r["wave_period_s"]
    return float(np.hypot(r["current_velocity_m_s"], U_m))


def fit_power_law(xs, ys):
    """Least-squares S/D = a * x^b in log space."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = (xs > 0) & (ys > 0)
    if mask.sum() < 2:
        return None
    b, log_a = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)
    return float(np.exp(log_a)), float(b)


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
    seed_everything(args.seed)
    out = ensure_dir(Path(args.output_dir))

    data = load_yu2024(ROOT)
    for r in data:
        r["U_eff"] = u_eff(r)
    xs = [r["U_eff"] for r in data]
    ys = [r["measured_scour_depth_ratio"] for r in data]

    # --- stage 1: equilibrium regression + leave-one-out ----------------------
    fit = fit_power_law(xs, ys)
    a_fit, b_fit = (fit if fit else (None, None))
    default_cfg = ScourConfig()
    eq_rows = []
    loo_errs = []
    for i, r in enumerate(data):
        keep_x = [xs[j] for j in range(len(data)) if j != i]
        keep_y = [ys[j] for j in range(len(data)) if j != i]
        loo = fit_power_law(keep_x, keep_y)
        pred_fit = a_fit * r["U_eff"] ** b_fit if fit else None
        pred_loo = (loo[0] * r["U_eff"] ** loo[1]) if loo else None
        err_loo = abs(pred_loo - r["measured_scour_depth_ratio"]) if loo else None
        if err_loo is not None:
            loo_errs.append(err_loo)
        eq_rows.append({
            "test_id": r["test_id"], "U_eff_m_s": r["U_eff"],
            "U_cw_paper": r["U_cw"], "KC": r["KC"],
            "measured_S_D": r["measured_scour_depth_ratio"],
            "paper_numerical_S_D": r["paper_numerical_scour_depth_ratio"],
            "fit_pred_S_D": pred_fit, "loo_pred_S_D": pred_loo,
            "loo_abs_err": err_loo,
        })

    pred_fit_all = np.array([a_fit * x ** b_fit for x in xs]) if fit else None
    rmse_fit = (float(np.sqrt(np.mean((pred_fit_all - np.array(ys)) ** 2)))
                if fit else None)
    rmse_paper = float(np.sqrt(np.mean(
        (np.array([r["paper_numerical_scour_depth_ratio"] for r in data])
         - np.array(ys)) ** 2)))
    # default coefficient form evaluated on U_eff
    rmse_default = float(np.sqrt(np.mean(
        (np.array([equilibrium_scour_ratio(x, default_cfg) for x in xs])
         - np.array(ys)) ** 2)))

    # --- stage 2: temporal calibration (final states only) --------------------
    T_end = 3600.0
    D = 0.10  # Yu 2024 model pipe diameter [m] (scenario for time scaling)
    U_typ = float(np.mean(xs))
    tau_grid = np.geomspace(60.0, 3600.0, 25)
    best_tau, best_err = None, None
    for tau in tau_grid:
        c = ScourConfig(tau_s_per_D=tau)
        errs = [abs(temporal_scour(T_end, r["measured_scour_depth_ratio"],
                                   D, r["U_eff"], c)
                    - r["measured_scour_depth_ratio"])
                / max(r["measured_scour_depth_ratio"], 1e-9)
                for r in data]
        m = float(np.mean(errs))
        if best_err is None or m < best_err:
            best_tau, best_err = float(tau), m
    temporal_rows = temporal_series(
        np.linspace(0.0, T_end, 13), float(np.mean(ys)), D, U_typ,
        ScourConfig(tau_s_per_D=best_tau))

    # --- stage 3: profile model (scenario) -------------------------------------
    x_grid = np.linspace(-3.0, 3.0, 121)
    D_p = 0.5
    S_scenario = 0.35 * D_p
    bed = ScourState(x_grid=x_grid, bed_elevation=np.zeros_like(x_grid))
    bed2 = apply_local_scour(bed, 0.0, S_scenario)
    prof_rows = [{"x_m": float(x), "bed_m": float(z)}
                 for x, z in zip(x_grid, bed2.bed_elevation)]
    spans = free_spans(np.column_stack([x_grid, bed2.bed_elevation]), D_p)

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=["dataset/processed/yu2024/scour_final.csv"],
                        extra={"status": STATUS})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "n_tests": len(data),
        "predictor": "U_eff = sqrt(U_c^2 + (pi*H/T)^2)",
        "fitted_a": a_fit, "fitted_b": b_fit,
        "rmse_fit": rmse_fit,
        "rmse_paper_numerical": rmse_paper,
        "rmse_default_coeffs_on_Ueff": rmse_default,
        "loo_mean_abs_err": float(np.mean(loo_errs)) if loo_errs else None,
        "best_tau_s_per_D": best_tau,
        "temporal_mean_rel_err_at_Tend": best_err,
        "temporal_limitation": "Test-2/Test-6 histories not digitized; tau "
                               "calibrated on final states only",
        "free_spans_scenario": [
            {"start_m": s[0], "end_m": s[1], "length_m": s[2],
             "max_gap_m": s[3]} for s in spans],
    })
    write_csv(out / "equilibrium_validation.csv", eq_rows)
    write_csv(out / "temporal_curve.csv", temporal_rows)
    write_csv(out / "profile_model.csv", prof_rows)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Scour validation", [
        ("Equilibrium S/D (U_eff power law + LOO)", eq_rows),
        ("Temporal approach (calibrated tau)", temporal_rows),
        ("Profile scenario (every 6th station)", prof_rows[::6]),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

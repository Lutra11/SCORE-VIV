"""E3 — Scour-FEM-VIV coupling (one-way, deterministic).

Stage-wise bed update from the validated E2 scour model, free-span
detection, support rebuild, modal analysis per stage, and VIV screening
of the evolving spans. Deterministic single chain: scour -> geometry ->
FEM -> VIV, repeated over stages.
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
from score_viv.fem import BeamFEM, BeamModel, Support
from score_viv.scour import ScourConfig, ScourState, apply_local_scour, free_spans
from score_viv.viv import (
    EnvironmentState, ScreeningConfig, classify_span, mass_ratio,
    reduced_velocity, response_amplitude_AD,
)

EXPERIMENT_ID = "E3"
STATUS = "IMPLEMENTED"

# Scenario pipeline (prototype scale, explicit assumptions)
PIPE = {
    "D_m": 0.5, "EI_Nm2": 1.2e7, "mass_per_length_kg_m": 350.0,
    "route_length_m": 600.0, "E_Pa": 2.1e11,
}


def build_supports(bed_x, bed_z, D, pin_gap=0.25):
    """Supports where bed touches the pipe (gap < pin_gap*D)."""
    gap = -bed_z
    supports = [Support(x=0.0, kind="pin", label="route-start")]
    for x, g in zip(bed_x, gap):
        if g < pin_gap * D:
            supports.append(Support(x=float(x), kind="pin",
                                     label=f"bed@{x:.1f}"))
    supports.append(Support(x=PIPE["route_length_m"], kind="pin",
                            label="route-end"))
    return supports


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

    n_stages = int(cfg.get("stages_count", 5))
    U_current = float(cfg.get("current_velocity_m_s", 0.35))
    U_wave = float(cfg.get("wave_orbital_velocity_m_s", 0.25))
    Ucw = float(cfg.get("Ucw", 0.4))
    D = PIPE["D_m"]

    x_grid = np.linspace(0.0, PIPE["route_length_m"], 241)
    bed = ScourState(x_grid=x_grid, bed_elevation=np.zeros_like(x_grid))
    # initial bathymetry: two broad depressions (free-spanning regions)
    # plus a low berm field; deterministic from the route scenario
    for xc, width, depth in ((180.0, 90.0, 0.55 * D), (420.0, 120.0, 0.75 * D)):
        sigma = width / 6.0
        bed.bed_elevation -= depth * np.exp(
            -0.5 * ((x_grid - xc) / sigma) ** 2)
    bed.bed_elevation += 0.05 * np.sin(2 * np.pi * x_grid / 150.0)

    scour_cfg = ScourConfig(eq_a=0.60, eq_b=0.28,
                           tau_s_per_D=1800.0, temporal_n=1.0)
    S_eq = 0.60 * Ucw ** 0.28 * D
    t_stage_s = float(cfg.get("stage_duration_s", 7200.0))

    stage_rows = []
    span_hist = []
    t_cum = 0.0
    for k in range(n_stages):
        t_cum += t_stage_s
        # local scour deepens at pit centers AND at the shoulders of the
        # existing depressions (shoulder scour lengthens the free spans);
        # shoulder pits use WIDE sigma so the depression flanks migrate
        from score_viv.scour import temporal_scour
        s_ratio = temporal_scour(t_cum, S_eq / D, D,
                                 U_current + U_wave, scour_cfg)
        centers = [150.0, 180.0, 210.0, 390.0, 420.0, 450.0]
        for x0 in centers:
            bed = apply_local_scour(bed, x0, s_ratio * D, scour_cfg)
        # additional shoulder widening: flat scour band at the flanks
        flank = ((np.abs(x_grid - 180.0) > 40.0) & (np.abs(x_grid - 180.0) < 80.0)) \
            | ((np.abs(x_grid - 420.0) > 50.0) & (np.abs(x_grid - 420.0) < 100.0))
        bed.bed_elevation[flank] -= 0.55 * D * min(1.0, (k + 1) / n_stages)
        spans = free_spans(np.column_stack([bed.x_grid, bed.bed_elevation]),
                           D, gap_criterion=0.5)
        span_hist.append({"stage": k + 1, "n_spans": len(spans),
                          "max_span_m": max((s[2] for s in spans), default=0.0),
                          "max_gap_m": max((s[3] for s in spans), default=0.0)})
        # FEM on the longest span region (route modeled full length)
        supports = build_supports(bed.x_grid, bed.bed_elevation, D)
        model = BeamModel(length=PIPE["route_length_m"], EI=PIPE["EI_Nm2"],
                          mass_per_length=PIPE["mass_per_length_kg_m"],
                          supports=supports, outer_diameter=D,
                          elastic_modulus=PIPE["E_Pa"])
        fem = BeamFEM(model, 120)
        freqs, shapes = fem.natural_modes(2)
        # VIV screening on mode 1
        m_star = mass_ratio(PIPE["mass_per_length_kg_m"], D)
        state = EnvironmentState(current_velocity=U_current,
                                  wave_orbital_velocity=U_wave, wave_period=1.5)
        U_tot = float(np.hypot(U_current, U_wave))
        Vr = reduced_velocity(U_tot, freqs[0], D)
        cls = classify_span(Vr, ScreeningConfig())
        A_D = response_amplitude_AD(Vr, m_star, 0.008, ScreeningConfig())
        stage_rows.append({
            "stage": k + 1, "t_s": t_cum, "n_supports": len(supports),
            "n_spans": len(spans),
            "max_span_m": max((s[2] for s in spans), default=0.0),
            "f1_hz": float(freqs[0]), "f2_hz": float(freqs[1]),
            "reduced_velocity": float(Vr), "classification": cls,
            "amplitude_AD": float(A_D),
            "scour_depth_max_m": float(np.max(-bed.bed_elevation)),
        })

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=["dataset/processed/yu2024/scour_final.csv"],
                        extra={"status": STATUS})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "coupling": "one_way_deterministic",
        "n_stages": n_stages,
        "f1_trend_hz": [r["f1_hz"] for r in stage_rows],
        "max_span_trend_m": [r["max_span_m"] for r in stage_rows],
        "notes": ["scenario pipeline parameters are explicit assumptions "
                  "(prototype scale), not published data"],
    })
    write_csv(out / "stage_evolution.csv", stage_rows)
    span_rows = []
    for h in span_hist:
        span_rows.append(h)
    write_csv(out / "span_history.csv", span_rows)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Coupled evolution", [
        ("Stage-wise geometry, modal and VIV evolution", stage_rows),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

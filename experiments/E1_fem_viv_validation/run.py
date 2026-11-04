"""E1 — FEM and VIV validation.

PROTOCOL RUN: deterministic benchmark cases; mesh convergence with >= 4
levels; seed validated against the E1 block.

Stages: beam_element, analytical_modal, mesh_convergence, multispan_modal
(Xu 2023 double/triple spans with SHOULDER SEGMENTS + soil springs,
full 5.6 m pipe), viv_response. Comparisons against Xu 2023 measured
frequencies remain PENDING until digitization completes; the multispan
model now builds the FULL pipe: shoulder segments beyond the pinned span
boundaries, and bed springs in the shoulder/contact zones, instead of
ideal pins at the span ends.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "algorithm"))

from score_viv.common import (
    ensure_dir, load_config, run_metadata, seed_everything, write_csv,
    write_json, write_markdown, merge_overrides,
)
from score_viv.protocol import validate_seed
from score_viv.fem import (
    BeamFEM, BeamModel, Support,
    cantilever_frequencies, mesh_convergence, ss_beam_frequencies,
)
from score_viv.viv import (
    EnvironmentState, ScreeningConfig, classify_span, mass_ratio,
    reduced_velocity, response_curves, stress_from_mode_shape,
)

EXPERIMENT_ID = "E1"
STATUS = "IMPLEMENTED"

# bed/soil spring stiffness for the shoulder-contact model [N/m/m]
# (scenario assumption; the Xu 2023 paper does not publish soil stiffness)
SHOULDER_SPRING_PER_M = 400.0


def load_xu2023(root: Path) -> dict:
    proc = root / "dataset/processed/xu2023"
    with open(proc / "pipeline_parameters.csv", encoding="utf-8") as fh:
        rows = fh.read().splitlines()
    params = {}
    for line in rows[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        params[parts[0]] = {
            "value": float(parts[2]), "unit": parts[3],
            "status": parts[4], "source_type": parts[5],
        }
    spans = []
    with open(proc / "span_geometries.csv", encoding="utf-8") as fh:
        for i, line in enumerate(fh.read().splitlines()):
            if i == 0 or not line.strip():
                continue
            p = line.split(",")
            spans.append({"case_id": p[0], "span_type": p[1],
                          "span_ratio": p[2],
                          "lengths_m": [float(v) for v in p[3:6] if v],
                          "status": p[6]})
    return {"params": params, "spans": spans}


def build_multispan_model(L_total, EI, m_s, lengths, spring_per_m):
    """FULL-pipe multispan model.

    The Xu 2023 tests use a 5.6 m pipe with a number of pinned span
    boundaries and pipe shoulders extending beyond them into the
    support frames. Modeling choice (explicit, honest):
    - span boundaries: ideal pins (the test frames impose w = 0 there)
    - shoulder segments: the remaining pipe length beyond the outermost
      pins, carrying distributed translational springs (bed contact)
    """
    nodes = [0.0]
    for ln in lengths:
        nodes.append(nodes[-1] + ln)
    span_end = nodes[-1]
    supports = []
    for x in nodes:
        supports.append(Support(x=x, kind="pin"))
    # shoulder: remaining pipe on each side, grounded by bed springs
    shoulder_l = max((L_total - span_end) / 2.0, 0.0)
    n_spring = 8
    springs = []
    for side in (0, 1):
        for i in range(n_spring):
            if side == 0:
                x = -(i + 0.5) * shoulder_l / n_spring
            else:
                x = span_end + (i + 0.5) * shoulder_l / n_spring
            k = spring_per_m * (L_total - span_end) / n_spring
            supports.append(Support(x=x, kind="spring", stiffness=k,
                                    label=f"bed{side}"))
            springs.append(x)
    # BeamFEM snaps supports to the nearest node; extend the domain by
    # giving BeamModel the full length and offsetting span positions.
    x0 = -shoulder_l
    shifted = []
    for s in supports:
        shifted.append(Support(x=s.x - x0, kind=s.kind,
                               stiffness=s.stiffness, label=s.label))
    # drop springs that fall outside [0, L_total]
    shifted = [s for s in shifted if 0.0 <= s.x <= L_total]
    model = BeamModel(length=L_total, EI=EI, mass_per_length=m_s,
                      supports=shifted)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=111000)
    parser.add_argument("--level", default="formal",
                        choices=["smoke", "pilot", "formal"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides",
                        help="key.sub=value config overrides")
    args = parser.parse_args()

    if args.dry_run:
        print(f"{EXPERIMENT_ID} DRY_RUN config={args.config} "
              f"output={args.output_dir} seed={args.seed} level={args.level}")
        return 0

    validate_seed(EXPERIMENT_ID, args.seed, args.level)
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
    data = load_xu2023(ROOT)
    p = data["params"]
    L = p["model_total_length"]["value"]        # 5.6 m
    D = p["outer_diameter"]["value"]            # 0.016 m
    EI = p["flexural_rigidity"]["value"]        # 17.45 N m^2
    m_s = p["mass_per_length_in_air"]["value"]  # 0.3821 kg/m
    m_star_pub = p["mass_ratio"]["value"]       # 1.90

    inputs = [
        "dataset/processed/xu2023/pipeline_parameters.csv",
        "dataset/processed/xu2023/span_geometries.csv",
    ]
    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir,
                         ROOT, inputs=inputs, extra={"status": STATUS,
                                                     "level": args.level})

    # ---- stage 1: beam element + analytic modal -----------------------------
    cantilever = BeamModel(length=L, EI=EI, mass_per_length=m_s,
                           supports=[Support(x=0.0, kind="clamp")])
    ss = BeamModel(length=L, EI=EI, mass_per_length=m_s,
                   supports=[Support(x=0.0, kind="pin"),
                             Support(x=L, kind="pin")])
    n_el = 64
    f_ss_fem = BeamFEM(ss, n_el).natural_modes(3)[0]
    f_cant_fem = BeamFEM(cantilever, n_el).natural_modes(3)[0]
    f_ss_ref = ss_beam_frequencies(EI, m_s, L, 3)
    f_cant_ref = cantilever_frequencies(EI, m_s, L, 3)

    analytic_rows = []
    for j in range(3):
        analytic_rows.append({
            "mode": j + 1,
            "fem_ss_hz": float(f_ss_fem[j]),
            "analytic_ss_hz": float(f_ss_ref[j]),
            "err_ss_pct": float(abs(f_ss_fem[j] - f_ss_ref[j])
                                / f_ss_ref[j] * 100),
            "fem_cant_hz": float(f_cant_fem[j]),
            "analytic_cant_hz": float(f_cant_ref[j]),
            "err_cant_pct": float(abs(f_cant_fem[j] - f_cant_ref[j])
                                  / f_cant_ref[j] * 100),
        })

    # ---- stage 2: mesh convergence (>= 4 levels) ----------------------------
    conv_rows = mesh_convergence(
        lambda n: BeamFEM(ss, n), [4, 8, 16, 32, 64, 128], n_modes=2)

    # ---- stage 3: multispan modal — FULL pipe with shoulders ---------------
    multi_rows = []
    for case in data["spans"]:
        model = build_multispan_model(
            L, EI, m_s, case["lengths_m"], SHOULDER_SPRING_PER_M)
        fem = BeamFEM(model, 128)
        freqs, _ = fem.natural_modes(3)
        span_sum = sum(case["lengths_m"])
        multi_rows.append({
            "case_id": case["case_id"], "span_type": case["span_type"],
            "span_ratio": case["span_ratio"],
            "span_total_m": span_sum,
            "shoulder_total_m": L - span_sum,
            "total_len_m": L,
            "f1_hz": float(freqs[0]), "f2_hz": float(freqs[1]),
            "f3_hz": float(freqs[2]),
            "measured_f1_hz": None,   # digitization PENDING
            "f1_error_pct": None,
            "validation_status": "PENDING_DIGITIZATION",
        })

    # ---- stage 4: VIV screening response ------------------------------------
    f_ref = multi_rows[0]["f1_hz"]
    m_star_calc = mass_ratio(m_s, D) - 1.0
    m_star_full = mass_ratio(m_s, D)
    cfg_screen = ScreeningConfig(
        velocity_combination=cfg.get("screening", {}).get(
            "velocity_combination", "rss"))
    state = EnvironmentState(current_velocity=0.3,
                              wave_orbital_velocity=0.15,
                              wave_period=1.5)
    rows_viv = response_curves(np.linspace(0.05, 0.9, 18), f_ref, D,
                               m_star_full, 0.008, state, cfg_screen)

    # stress recovery on the dominant mode of the first case
    first = data["spans"][0]
    model = build_multispan_model(L, EI, m_s, first["lengths_m"],
                                  SHOULDER_SPRING_PER_M)
    fem = BeamFEM(model, 128)
    freqs, shapes = fem.natural_modes(1)
    E_assumed = 70.0e9  # scenario assumption [Pa] (not published)
    stress = stress_from_mode_shape(shapes[0], fem.x,
                                    amplitude=0.55 * D, D=D, E=E_assumed)

    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "level": args.level,
        "stages_completed": cfg.get("stages"),
        "max_f1_error_analytic_pct": max(
            (r["err_ss_pct"] for r in analytic_rows), default=0.0),
        "mesh_converged_rel_err": conv_rows[-2]["rel_err_1"],
        "multispan_cases": len(multi_rows),
        "multispan_measured_pending": True,
        "multispan_model": "full-pipe: pins at span boundaries + "
                          f"shoulder segments ({L} m total) with bed "
                          f"springs ({SHOULDER_SPRING_PER_M} N/m/m "
                          "scenario value)",
        "mass_ratio_published": m_star_pub,
        "mass_ratio_calculated": m_star_calc,
        "viv_stress_recovery": stress,
        "notes": [
            "Xu 2023 measured frequencies/responses: digitization PENDING; "
            "modal errors vs measurements are not yet computable.",
            "shoulder soil-spring stiffness is a scenario value (the "
            "paper does not publish it); sensitivity to it belongs to E9.",
            "E assumption for stress recovery is a scenario value, not "
            "a published parameter.",
        ],
    })
    write_csv(out / "analytic_modal_validation.csv", analytic_rows)
    write_csv(out / "mesh_convergence.csv", conv_rows)
    write_csv(out / "multispan_modal.csv", multi_rows)
    write_csv(out / "viv_response_sweep.csv", rows_viv)
    write_json(out / "stress_recovery.json", stress)
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — FEM/VIV validation", [
        ("Analytic modal validation (simply supported / cantilever)",
         analytic_rows),
        ("Mesh convergence", conv_rows),
        ("Multispan modal (full-pipe with shoulders)", multi_rows),
        ("VIV screening sweep (first case, mode 1)", rows_viv),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

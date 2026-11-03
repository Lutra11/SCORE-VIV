"""E4 — Fatigue validation.

Two HONEST stress routes (never RMS->history fabrication):
1. synthetic stress HISTORY (with published-form S-N) -> rainflow -> Miner;
2. spectral narrowband Rayleigh route from an RMS amplitude series.
Both use the same S-N family; cross-check of the two routes is the
validation output. Xu 2023 raw strain histories: NOT_CONFIRMED_PUBLIC,
so no measured-fatigue comparison is claimed.
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
from score_viv.fatigue import (
    SNCurve, damage_from_history, extract_cycles, spectral_damage,
    system_limit_state,
)
from score_viv.viv import EnvironmentState, ScreeningConfig, response_amplitude_AD, reduced_velocity

EXPERIMENT_ID = "E4"
STATUS = "IMPLEMENTED"

# S-N defaults are the DNV-style air curve FAMILY; using them for seawater
# requires the -3 slope/knee corrections which are calibration inputs.
SN_DEFAULTS = dict(log_a1=11.764, m1=3.0, log_a2=15.606, m2=5.0,
                   stress_ref_mpa=0.530, thickness_mm=20.0, t_ref_mm=25.0,
                   thickness_exponent=0.20)


def synthetic_stress_history(rng, n_blocks=40, base_amp_mpa=8.0,
                              period_s=4.0, fs_hz=8.0, duration_s=3600.0):
    """Deterministic-from-seed synthetic stress history [Pa].

    Multi-amplitude harmonic blocks: fatigue loads with a physical basis
    (VIV at lock-in produces near-harmonic stress); NOT a measured trace.
    """
    t = np.arange(0, duration_s, 1.0 / fs_hz)
    sig = np.zeros_like(t)
    amps = base_amp_mpa * (0.6 + 0.8 * rng.random(n_blocks))
    block_len = max(len(t) // n_blocks, 1)
    for b in range(n_blocks):
        s = b * block_len
        e = min((b + 1) * block_len, len(t))
        if s >= len(t):
            break
        # slowly varying frequency around the VIV period
        f = 1.0 / (period_s * (0.85 + 0.3 * rng.random()))
        sig[s:e] = amps[b] * 1e6 * np.sin(2 * np.pi * f * t[s:e])
    return t, sig


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

    sn_cfg = {**SN_DEFAULTS, **cfg.get("sn_curve_overrides", {})}
    curve = SNCurve(**sn_cfg)

    # S-N curve self-consistency table (MPa rows, API takes Pa)
    sn_rows = []
    for S_mpa in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0):
        N = float(curve.allowable_cycles(S_mpa * 1e6))
        sn_rows.append({"stress_range_mpa": S_mpa,
                        "allowable_cycles": N,
                        "log10_N": float(np.log10(N))})

    # Route 1: synthetic history -> rainflow
    t, sig = synthetic_stress_history(rng)
    ranges, counts, means = extract_cycles(sig)
    dmg_hist = damage_from_history(sig, curve)
    n_cycles = int(len(ranges))
    big_cycles = sorted(zip(ranges / 1e6, counts), reverse=True)[:8]
    hist_rows = [{"range_mpa": float(r), "count": float(c)}
                 for r, c in big_cycles]

    # Route 2: spectral from block RMS amplitudes (same synthetic signal)
    block_rms = []
    block_len = max(len(sig) // 40, 1)
    for b in range(40):
        s = b * block_len
        e = min((b + 1) * block_len, len(sig))
        if s >= len(sig):
            break
        block_rms.append(float(np.sqrt(np.mean(sig[s:e] ** 2))))
    duration_s = float(t[-1] - t[0])
    # damage of the SAME exposure window as route 1 (1 hour)
    dmg_spec_window = spectral_damage(np.asarray(block_rms), 4.0,
                                       duration_s, curve)
    # annualized by scaling to a full year of identical exposure
    window_hours = duration_s / 3600.0
    dmg_spec_annual = dmg_spec_window * (8760.0 / window_hours) \
        if window_hours > 0 else 0.0
    dmg_hist_annual = dmg_hist * (8760.0 / window_hours) \
        if window_hours > 0 else 0.0

    # multi-span system damage (3 spans, scaled exposure scenario)
    span_scale = (1.0, 0.8, 0.6)
    damages = [dmg_spec_annual * s * 25.0 for s in span_scale]  # 25y exposure
    sys_state = system_limit_state(damages, limit=1.0)

    meta = run_metadata(EXPERIMENT_ID, cfg, args.seed, args.output_dir, ROOT,
                        inputs=["dataset/processed/xu2023/pipeline_parameters.csv"],
                        extra={"status": STATUS})
    write_json(out / "metadata.json", meta)
    write_json(out / "summary.json", {
        "status": STATUS,
        "route1_damage_window": dmg_hist,
        "route2_damage_window": dmg_spec_window,
        "routes_ratio_window": (dmg_hist / dmg_spec_window
                                if dmg_spec_window > 0 else None),
        "route1_damage_annualized": dmg_hist_annual,
        "route2_damage_annualized": dmg_spec_annual,
        "n_rainflow_cycles": n_cycles,
        "system_25y": sys_state,
        "history_note": "synthetic (seeded), not measured; Xu 2023 raw strain "
                        "histories NOT_CONFIRMED_PUBLIC",
        "sn_note": "S-N family defaults; seawater correction is a "
                   "calibration input",
    })
    write_csv(out / "sn_curve.csv", sn_rows)
    write_csv(out / "top_cycles.csv", hist_rows)
    write_json(out / "block_rms_series.json",
               {"block_rms_mpa": block_rms, "duration_s": duration_s})
    write_markdown(out / "report.md", f"{EXPERIMENT_ID} — Fatigue validation", [
        ("S-N curve (Pa inputs, MPa rows)", sn_rows),
        ("Largest rainflow cycles (route 1)", hist_rows),
    ])
    print(f"{EXPERIMENT_ID} complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scour layer: equilibrium and temporal scour models with bed updates and
free-span detection.

Models follow the reduced-order, literature-standard families:

- equilibrium depth (current):   S/D = 2.2 * (1 - U/U_c)^2.5  (Sumer 2002 form)
- equilibrium depth (waves/wave-current): S/D = a * (U_cw)^b with default
  coefficients from the Yu 2024 regression context; these are CALIBRATION
  COEFFICIENTS fitted per dataset, not universal constants.
- temporal: S(t) = S_eq * (1 - exp(-(t/tau)^n)) (Sumer-style exponential law)
- profile: Gaussian ridge centered at span node with width from scour width S_w.

This layer does not contain the FEM or the optimizer (repository contract).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCOUR_MODEL_VERSION = "reduced-order-v1"


@dataclass
class ScourConfig:
    """Coefficients of the reduced-order scour family (calibratable)."""

    # equilibrium, wave-dominated: S/D = a * Ucw^b
    eq_a: float = 0.60
    eq_b: float = 0.28
    # temporal law: S/S_eq = 1 - exp(-(t/tau)^n)
    tau_s_per_D: float = 1800.0    # characteristic time [s] (scaled by U,D)
    temporal_n: float = 1.0
    # profile: width of scour hole relative to depth
    width_factor: float = 4.0
    ridge_sigma_factor: float = 0.35
    max_scour_depth_ratio: float = 1.0    # cap relative to D


@dataclass
class ScourState:
    """Bed geometry along x: ridge crest, shoulder-to-shoulder span nodes."""

    x_grid: np.ndarray            # bed stations [m]
    bed_elevation: np.ndarray     # initial bed (0 = undisturbed seabed) [m]


def equilibrium_scour_ratio(Ucw: float, cfg: ScourConfig | None = None) -> float:
    """S_eq/D for combined wave-current motion (dimensionless)."""
    cfg = cfg or ScourConfig()
    if Ucw <= 0:
        return 0.0
    return float(cfg.eq_a * Ucw ** cfg.eq_b)


def equilibrium_current_scour_ratio(U, Uc, cfg: ScourConfig | None = None) -> float:
    """S_eq/D for steady current with critical velocity Uc (Sumer form)."""
    if U <= 0 or Uc <= 0:
        return 0.0
    if U >= Uc:
        return 2.2  # live-bed cap of the reference form
    return float(2.2 * (1.0 - U / Uc) ** 2.5)


def temporal_scour(t, S_eq_ratio, D, U, cfg: ScourConfig | None = None) -> float:
    """S(t)/D via exponential approach; tau scales with 1/U."""
    cfg = cfg or ScourConfig()
    if t <= 0 or S_eq_ratio <= 0 or U <= 0:
        return 0.0
    tau = cfg.tau_s_per_D * D / max(U, 1e-9)
    return float(S_eq_ratio * (1.0 - np.exp(-((t / tau) ** cfg.temporal_n))))


def scour_width(S, cfg: ScourConfig | None = None) -> float:
    """Scour-hole width ~ width_factor * depth (dimensionally consistent)."""
    cfg = cfg or ScourConfig()
    return float(cfg.width_factor * max(S, 0.0))


def gaussian_profile(x, x0, S, cfg: ScourConfig | None = None) -> np.ndarray:
    """Local scour depression as a Gaussian pit (negative elevation)."""
    cfg = cfg or ScourConfig()
    sigma = max(cfg.ridge_sigma_factor * scour_width(S, cfg), 1e-9)
    return -S * np.exp(-0.5 * ((np.asarray(x, dtype=float) - x0) / sigma) ** 2)


def apply_local_scour(state: ScourState, x0: float, S: float,
                      cfg: ScourConfig | None = None) -> ScourState:
    """Pit subtracted from current bed; returns a NEW state."""
    bed_new = state.bed_elevation - gaussian_profile(
        state.x_grid, x0, S, cfg)
    return ScourState(x_grid=state.x_grid.copy(),
                     bed_elevation=bed_new)


def free_spans(bed: np.ndarray, D: float, gap_criterion: float = 0.5):
    """Detect free spans where local gap between pipe and bed >= criterion*D.

    bed is (n_stations, 2) array of (x, elevation); pipe assumed at z=0.
    Returns list of (start_x, end_x, length, max_gap) tuples.
    """
    bed = np.asarray(bed, dtype=float)
    if bed.ndim != 2 or bed.shape[1] != 2:
        raise ValueError("bed must be (n, 2): x, elevation")
    gap = -bed[:, 1]  # bed below z=0 -> positive gap
    free = gap >= gap_criterion * D
    spans = []
    start = None
    for i, is_free in enumerate(free):
        if is_free and start is None:
            start = i
        elif not is_free and start is not None:
            seg = gap[start:i]
            spans.append((float(bed[start, 0]), float(bed[i - 1, 0]),
                          float(bed[i - 1, 0] - bed[start, 0]),
                          float(np.max(seg))))
            start = None
    if start is not None:
        seg = gap[start:]
        spans.append((float(bed[start, 0]), float(bed[-1, 0]),
                      float(bed[-1, 0] - bed[start, 0]),
                      float(np.max(seg))))
    return spans


@dataclass
class TemporalResult:
    t_s: float
    scour_ratio: float


def temporal_series(t_values, S_eq_ratio, D, U, cfg: ScourConfig | None = None):
    """Temporal series rows for calibration plots."""
    rows = []
    for t in np.asarray(t_values, dtype=float):
        rows.append({
            "t_s": float(t),
            "scour_depth_ratio": temporal_scour(t, S_eq_ratio, D, U, cfg),
        })
    return rows

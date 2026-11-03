"""VIV screening layer: wave-current states, reduced velocity, span
classification, a reduced (screening-level) CF response model, and stress
recovery from mode shapes.

All thresholds and response coefficients are configuration parameters with
literature-typical defaults; they are screening values, not quotations from
a standard. Calibration against digitized Xu 2023 figures is tracked as
PENDING in the experiment outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RESPONSE_MODEL_VERSION = "screening-v1"


@dataclass
class ScreeningConfig:
    """Screening thresholds (documented defaults, config-overridable)."""

    onset_in_line: float = 1.8
    onset_cross_flow: float = 3.0
    end_cross_flow: float = 8.0
    peak_reduced_velocity: float = 5.5
    response_width: float = 1.6
    ad_peak: float = 0.55
    griffin_exponent: float = 0.25
    mstar_zeta_reference: float = 0.05
    ad_min: float = 0.05
    ad_max: float = 1.1
    velocity_combination: str = "rss"


@dataclass
class EnvironmentState:
    """Flow state at pipe level."""

    current_velocity: float = 0.0        # U_c [m/s]
    wave_orbital_velocity: float = 0.0   # U_m amplitude [m/s]
    wave_period: float = 1.5             # [s]
    water_density: float = 1025.0        # [kg/m^3]


def combined_velocity(state: EnvironmentState, how: str | None = None) -> float:
    how = how or "rss"
    if how == "rss":
        return float(np.hypot(state.current_velocity,
                               state.wave_orbital_velocity))
    if how == "sum":
        return state.current_velocity + state.wave_orbital_velocity
    raise ValueError(f"velocity_combination must be rss|sum: {how}")


def reduced_velocity(U: float, f_n: float, D: float) -> float:
    """Vr = U / (f_n * D)."""
    if f_n <= 0 or D <= 0:
        raise ValueError("f_n and D must be positive")
    return U / (f_n * D)


def mass_ratio(mass_per_length: float, D: float,
               water_density: float = 1025.0) -> float:
    """m* including internal added mass: m/(rho*pi*D^2/4) + 1."""
    if D <= 0:
        raise ValueError("D must be positive")
    displaced = water_density * np.pi * D**2 / 4.0
    return mass_per_length / displaced + 1.0


def stability_parameter(m_star: float, zeta: float) -> float:
    return m_star * zeta


def classify_span(Vr: float, cfg: ScreeningConfig | None = None) -> str:
    """Screening classification: none / in_line / cross_flow."""
    cfg = cfg or ScreeningConfig()
    if Vr < cfg.onset_in_line:
        return "no_viv"
    if Vr < cfg.onset_cross_flow:
        return "in_line"
    return "cross_flow"


def response_amplitude_AD(Vr: float, m_star: float, zeta: float,
                          cfg: ScreeningConfig | None = None) -> float:
    """Reduced CF response model: A/D = A_pk * gauss(Vr) for onset<=Vr<=end."""
    cfg = cfg or ScreeningConfig()
    if Vr < cfg.onset_cross_flow or Vr > cfg.end_cross_flow:
        return 0.0
    kS = stability_parameter(max(m_star, 1e-3), max(zeta, 1e-6))
    factor = (cfg.mstar_zeta_reference / kS) ** cfg.griffin_exponent
    a_pk = float(np.clip(cfg.ad_peak * factor, cfg.ad_min, cfg.ad_max))
    g = np.exp(-0.5 * ((Vr - cfg.peak_reduced_velocity) / cfg.response_width) ** 2)
    return float(a_pk * g)


def response_curves(U_values, f_n: float, D: float, m_star: float, zeta: float,
                    state: EnvironmentState | None = None,
                    cfg: ScreeningConfig | None = None):
    """Sweep current velocities -> Vr, classification, A/D table rows."""
    cfg = cfg or ScreeningConfig()
    state = state or EnvironmentState()
    rows = []
    for U in np.asarray(U_values, dtype=float):
        st = EnvironmentState(
            current_velocity=float(U),
            wave_orbital_velocity=state.wave_orbital_velocity,
            wave_period=state.wave_period,
            water_density=state.water_density,
        )
        U_tot = combined_velocity(st, cfg.velocity_combination)
        Vr = reduced_velocity(U_tot, f_n, D)
        rows.append({
            "current_m_s": float(U),
            "total_velocity_m_s": float(U_tot),
            "reduced_velocity": float(Vr),
            "classification": classify_span(Vr, cfg),
            "amplitude_AD": response_amplitude_AD(Vr, m_star, zeta, cfg),
        })
    return rows


def stress_from_mode_shape(shape, x_grid, amplitude, D, E,
                            wall_thickness=None) -> dict:
    """Recover harmonic stress from a mode shape scaled to physical amplitude.

    shape is normalized to max|w| = 1 (unitless); amplitude is the physical
    peak amplitude [m]. Returns amplitude and RMS stress [Pa].
    """
    from scipy.interpolate import CubicSpline

    if len(shape) != len(x_grid) or len(shape) < 3:
        raise ValueError("shape and x_grid must match with >=3 points")
    cs = CubicSpline(np.asarray(x_grid, dtype=float),
                     np.asarray(shape, dtype=float))
    kappa = cs(np.asarray(x_grid, dtype=float), 2)
    kappa_scaled = kappa * amplitude
    c = 0.5 * D if wall_thickness is None else 0.5 * (D - wall_thickness) + wall_thickness
    sigma_amp = E * c * kappa_scaled
    i_max = int(np.argmax(np.abs(sigma_amp)))
    return {
        "sigma_amplitude_pa_max": float(np.max(np.abs(sigma_amp))),
        "sigma_rms_pa_max": float(np.max(np.abs(sigma_amp)) / np.sqrt(2.0)),
        "location_of_max_m": float(x_grid[i_max]),
        "max_curvature_per_m": float(np.max(np.abs(kappa_scaled))),
        "fiber_distance_m": float(c),
    }

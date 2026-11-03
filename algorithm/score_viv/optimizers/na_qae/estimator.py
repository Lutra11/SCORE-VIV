"""NA-QAE probability estimator (numerical, noise-aware).

MEASUREMENT MODEL
-----------------
Grover depth m gives M = 2m+1 oracle calls per shot. For true amplitude
angle theta = arcsin(sqrt(p)), the ideal 'good' probability is

    a(p, m) = sin^2( (2m+1) * theta )

A depolarizing channel attenuates the register toward the uniform measure:

    q(p, m) = eta(m) * a(p, m) + (1 - eta(m)) * 1/2,
    eta(m)  = exp(-lam * 2^m)

Finite shots sample k ~ Binomial(shots, q).

ESTIMATION
----------
A single-depth arcsin inversion is ambiguous (phase wrapping when
(2m+1)*theta crosses pi/2), so NA-QAE estimates p by a CROSS-DEPTH
JOINT LIKELIHOOD: counts from a ladder of depths are combined,

    L(p) = sum_m [ k_m log q(p, m) + (n_m - k_m) log(1 - q(p, m)) ]

evaluated with the ASSUMED noise model (the noise-aware correction),
maximized over a log-spaced grid; the 95% interval is the likelihood-
ratio region L(p) >= L_max - 1.92. The shallow depths anchor the branch
(they wrap for no p), the deep ones contribute the Grover amplification
(Fisher information grows ~ M^2 while M*theta < pi/2), and the adaptive
budget ladder escalates shots only while the interval is still wide.

Ideal (noiseless) QAE accuracy after M Grover iterations is bounded by
2*pi/M (phase error bound; standard result). Everything here runs on a
classical RNG; this is NOT a claim of real quantum-hardware acceleration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2, norm

_Z95 = float(norm.ppf(0.975))
_LR95 = float(chi2.ppf(0.95, 1) / 2.0)      # 1.92: 95% LR half-drop


def theoretical_shots_for_accuracy(p, eps, alpha=0.05):
    """Classical Bernoulli MC shots for |p_hat - p| <= eps at 1-alpha."""
    if p is None or not (0 < p < 1) or eps <= 0:
        return None
    z = float(norm.ppf(1 - alpha / 2))
    return int(math.ceil(z ** 2 * p * (1 - p) / eps ** 2))


def qae_phase_bound(m: int) -> float:
    """Ideal QAE phase-error bound 2*pi/M with M = 2m+1 Grover calls."""
    if m < 0:
        raise ValueError("Grover depth must be >= 0")
    return 2.0 * math.pi / max(2 * int(m) + 1, 1)


def grover_iterations(m: int) -> int:
    return 2 * int(m) + 1


@dataclass
class NoiseModel:
    """Depolarizing attenuation eta(m) = exp(-lam * 2^m)."""

    lam: float = 0.02

    def eta(self, m: int) -> float:
        return float(math.exp(-self.lam * (2.0 ** m)))


def _clip01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def channel_q(p, m: int, noise: NoiseModel):
    """q(p, m); scalar or ndarray p (principal branch)."""
    theta = np.arcsin(np.sqrt(np.clip(np.asarray(p, dtype=float), 0.0, 1.0)))
    a = np.sin(grover_iterations(m) * theta) ** 2
    eta = noise.eta(m)
    q = eta * a + (1.0 - eta) * 0.5
    if np.ndim(p) == 0:
        return float(q)
    return q


def measure_counts(p_true: float, m: int, shots: int, noise: NoiseModel,
                   rng: np.random.Generator) -> int:
    """One batch of 'good' counts from the noisy amplified register."""
    q = channel_q(float(p_true), m, noise)
    return int(rng.binomial(int(shots), min(max(q, 0.0), 1.0)))


@dataclass
class QAEEstimate:
    depth: int
    shots: int
    p_hat: float
    var_hat: float
    ci_low: float
    ci_high: float
    eta: float
    ideal_bound: float
    method: str = "NA-QAE"


def _grid(n: int = 5000) -> np.ndarray:
    return np.geomspace(1e-6, 1.0 - 1e-6, n)


def joint_mle(counts, assumed_noise: NoiseModel, grid: np.ndarray | None = None,
              method: str = "NA-QAE") -> dict:
    """Cross-depth joint maximum-likelihood estimate from count tuples.

    counts: iterable of (depth m, good counts k, shots n). The likelihood
    uses ``assumed_noise`` — pass a mismatched model to express the
    no-correction ablation honestly.
    """
    counts = [(int(m), int(k), int(n)) for m, k, n in counts if int(n) > 0]
    if not counts:
        raise ValueError("no counts")
    if grid is None:
        grid = _grid()
    loglik = np.zeros_like(grid)
    for m, k, n in counts:
        q = np.clip(channel_q(grid, m, assumed_noise), 1e-12, 1 - 1e-12)
        loglik += k * np.log(q) + (n - k) * np.log(1.0 - q)
    j = int(np.argmax(loglik))
    p_hat = float(grid[j])
    lmax = float(loglik[j])
    ok = np.where(loglik >= lmax - _LR95)[0]
    lo = float(grid[ok[0]]) if ok.size else 0.0
    hi = float(grid[ok[-1]]) if ok.size else 1.0
    if lo > p_hat:
        lo = p_hat
    if hi < p_hat:
        hi = p_hat
    half = 0.5 * (hi - lo)
    var_hat = (half / _Z95) ** 2
    return {"p_hat": p_hat, "ci_low": lo, "ci_high": hi,
            "half_width": half, "var_hat": var_hat, "method": method,
            "n_depths": len({m for m, _, _ in counts}),
            "total_shots": int(sum(n for _, _, n in counts))}


def estimate_at_depth(p_true_or_rng, m: int, shots: int, noise: NoiseModel,
                      rng: np.random.Generator | None = None,
                      p_true: float | None = None,
                      assumed_noise: NoiseModel | None = None) -> QAEEstimate:
    """Single-depth estimate (kept for baselines/ablation diagnostics).

    Uses the de-biased arcsin inversion with an ambiguity guard; the
    PRODUCTION NA-QAE path is ``adaptive_estimate`` (joint likelihood).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if p_true is None:
        p_true = float(p_true_or_rng)
    if not (0.0 <= p_true <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if shots <= 0:
        raise ValueError("shots must be positive")
    k = measure_counts(p_true, m, shots, noise, rng)
    out = joint_mle([(m, k, shots)], assumed_noise or noise,
                    method=f"NA-QAE(m={m})")
    return QAEEstimate(m, shots, out["p_hat"], out["var_hat"],
                       out["ci_low"], out["ci_high"], (assumed_noise or noise).eta(m),
                       qae_phase_bound(m), method=out["method"])


def adaptive_budget_schedule(initial_budget: int, max_budget: int,
                             growth: float = 2.0) -> list[int]:
    budgets, b = [], int(initial_budget)
    while b <= max_budget:
        budgets.append(b)
        b = max(int(b * growth), b + 1)
    if not budgets or budgets[-1] != int(max_budget):
        budgets.append(int(max_budget))
    return budgets


def adaptive_estimate(p_true: float, noise: NoiseModel,
                      rng: np.random.Generator,
                      initial_budget: int = 2048,
                      max_budget: int = 8192,
                      depth_schedule=(0, 1, 2, 4, 8, 16),
                      stop_half_width: float | None = None,
                      assumed_noise: NoiseModel | None = None,
                      rel_stop: float = 0.15,
                      abs_stop: float = 2e-3) -> dict:
    """Adaptive depth x budget ladder with the cross-depth joint MLE.

    Each round measures ``s`` shots at EVERY depth of the schedule,
    refits the joint likelihood on ALL counts so far, and stops when the
    interval is decisively tight (half <= max(abs_stop, rel_stop * p_hat))
    or the shot ceiling is reached.
    """
    if not (0.0 <= p_true <= 1.0):
        raise ValueError("p must be in [0, 1]")
    depths = tuple(int(m) for m in depth_schedule)
    if not depths:
        raise ValueError("empty depth schedule")
    assumed = assumed_noise or noise
    grid = _grid()

    s = max(int(initial_budget), 8)
    counts: list[tuple[int, int, int]] = []
    history = []
    total = 0
    best = None

    while total + s * len(depths) <= int(max_budget) or not counts:
        round_shots = 0
        for m in depths:
            k = measure_counts(p_true, m, s, noise, rng)
            counts.append((m, k, s))
            round_shots += s
        total += round_shots
        est = joint_mle(counts, assumed, grid=grid)
        best = {"p_hat": est["p_hat"], "half_width": est["half_width"],
                "depth": max(depths), "shots": total,
                "ci": [est["ci_low"], est["ci_high"]],
                "eta": noise.eta(max(depths)), "var_hat": est["var_hat"]}
        history.append({"round": len(history) + 1, "shots": round_shots,
                        "total_shots": total, "p_hat": est["p_hat"],
                        "half_width": est["half_width"]})
        target = (stop_half_width if stop_half_width is not None
                  else max(abs_stop, rel_stop * est["p_hat"]))
        if est["half_width"] <= target:
            return {"best": best, "history": history, "stopped_early": True}
        s *= 2
        if total >= int(max_budget):
            break

    if best is None:  # single-round ceiling path
        est = joint_mle(counts, assumed, grid=grid)
        best = {"p_hat": est["p_hat"], "half_width": est["half_width"],
                "depth": max(depths), "shots": total,
                "ci": [est["ci_low"], est["ci_high"]],
                "eta": noise.eta(max(depths)), "var_hat": est["var_hat"]}
    return {"best": best, "history": history, "stopped_early": False}


def interval_decision(p_hat: float, ci_low: float, ci_high: float,
                      threshold: float) -> str:
    if ci_low > threshold:
        return "safe"
    if ci_high < threshold:
        return "unsafe"
    return "undetermined"


def mlae_estimate(p_true: float, m_max: int, noise: NoiseModel,
                  rng: np.random.Generator, shots_per_level: int = 100,
                  grid_size: int = 5000) -> dict:
    """Maximum-Likelihood Amplitude Estimation baseline.

    Depths 0..m_max, fixed uniform shots per level through the SAME noisy
    amplified channel; joint grid MLE (non-adaptive ladder).
    """
    depths = list(range(0, int(m_max) + 1))
    grid = np.geomspace(1e-6, 1.0 - 1e-6, grid_size)
    counts = []
    for m in depths:
        k = measure_counts(p_true, m, shots_per_level, noise, rng)
        counts.append((m, k, shots_per_level))
    est = joint_mle(counts, noise, grid=grid, method="MLAE")
    return {**est, "depths": depths, "shots_per_level": shots_per_level,
            "n_evals": shots_per_level * len(depths)}


def iqae_estimate(p_true: float, noise: NoiseModel,
                  rng: np.random.Generator, eps_target: float = 2e-3,
                  max_shots: int = 2048, shots_per_step: int = 128) -> dict:
    """Iterative QAE baseline: deepen until CI half-width <= eps.

    Uses the joint likelihood over ALL depths visited so far (the
    literature IQAE refits one depth; we keep the shared joint core and
    the difference to NA-QAE is the depth-growth rule, not the fitter).
    """
    depths = [0, 1, 2, 4]
    counts = []
    shots_used = 0
    est = None
    grid = _grid()
    while depths and shots_used < max_shots:
        m = depths.pop(0)
        k = measure_counts(p_true, m, shots_per_step, noise, rng)
        counts.append((m, k, shots_per_step))
        shots_used += shots_per_step
        est = joint_mle(counts, noise, grid=grid, method="IQAE")
        if est["half_width"] <= eps_target:
            break
        nxt = min(2 * m, 16) if m > 0 else 1
        if nxt == m:
            break
        depths.insert(0, nxt)
    if est is None:
        est = joint_mle([(0, 0, 1)], noise, grid=grid, method="IQAE")
    return {"p_hat": est["p_hat"], "ci_low": est["ci_low"],
            "ci_high": est["ci_high"], "n_evals": shots_used,
            "method": "IQAE"}


def noise_lam_from_paired(pairs) -> float | None:
    """Learn lam from (depth, eta_measured) pairs via log-linear fit."""
    if not pairs or len(pairs) < 2:
        return None
    xs = np.array([2.0 ** m for m, _ in pairs], dtype=float)
    ys = np.log(np.clip([e for _, e in pairs], 1e-9, 1.0))
    if np.allclose(xs, xs[0]):
        return None
    slope, _ = np.polyfit(xs, ys, 1)
    return float(-slope)

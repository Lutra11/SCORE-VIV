"""Reliability layer: random variables, dependence, Monte Carlo and
importance sampling, confidence intervals, and system limit states.

Failure-probability estimators return point estimates AND interval bounds,
with explicit evaluation counts, so no method can claim unearned precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass
class RandomVariable:
    name: str
    dist: str                 # "normal" | "lognormal" | "uniform" | "gumbel"
    mean: float = 0.0
    std: float = 1.0
    lower: float | None = None
    upper: float | None = None

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.dist == "normal":
            return rng.normal(self.mean, self.std, n)
        if self.dist == "lognormal":
            sigma = np.sqrt(np.log(1 + (self.std / max(self.mean, 1e-12)) ** 2))
            mu = np.log(max(self.mean, 1e-12)) - 0.5 * sigma ** 2
            return rng.lognormal(mu, sigma, n)
        if self.dist == "uniform":
            lo, hi = self.lower or 0.0, self.upper or 1.0
            return rng.uniform(lo, hi, n)
        if self.dist == "gumbel":
            beta = self.std * np.sqrt(6) / np.pi
            mu = self.mean - 0.5772 * beta
            return rng.gumbel(mu, beta, n)
        raise ValueError(f"unknown dist: {self.dist}")


def correlated_samples(rvs, corr_matrix, n: int, rng: np.random.Generator):
    """Gaussian-copula correlated samples for a list of RandomVariable."""
    k = len(rvs)
    corr = np.asarray(corr_matrix, dtype=float)
    if corr.shape != (k, k):
        raise ValueError("corr_matrix must be k x k")
    L = np.linalg.cholesky(corr + 1e-12 * np.eye(k))
    z = rng.standard_normal((n, k)) @ L.T
    out = np.zeros((n, k))
    for j, rv in enumerate(rvs):
        # map through marginal via inverse-CDF on the normal copula scale
        u = norm.cdf(z[:, j])
        out[:, j] = _ppf(rv, u)
    return out


def _ppf(rv: RandomVariable, u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 1e-12, 1 - 1e-12)
    if rv.dist == "normal":
        return norm.ppf(u, rv.mean, rv.std)
    if rv.dist == "lognormal":
        sigma = np.sqrt(np.log(1 + (rv.std / max(rv.mean, 1e-12)) ** 2))
        mu = np.log(max(rv.mean, 1e-12)) - 0.5 * sigma ** 2
        return np.exp(norm.ppf(u, mu, sigma))
    if rv.dist == "uniform":
        lo, hi = rv.lower or 0.0, rv.upper or 1.0
        return lo + u * (hi - lo)
    if rv.dist == "gumbel":
        beta = rv.std * np.sqrt(6) / np.pi
        mu = rv.mean - 0.5772 * beta
        return mu - beta * np.log(-np.log(u))
    raise ValueError(f"unknown dist: {rv.dist}")


def limit_state_values(samples: np.ndarray, fn) -> np.ndarray:
    """Vectorized G(x) = fn(sample_row) with failure G <= 0."""
    return np.asarray([fn(row) for row in samples], dtype=float)


def failure_indicator(G: np.ndarray) -> np.ndarray:
    return (np.asarray(G, dtype=float) <= 0.0).astype(float)


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial CI (Clopper-Pearson) via Beta quantiles."""
    from scipy.stats import beta as _beta
    if n <= 0:
        return (None, None)
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def wald_interval(k: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return (None, None)
    p = k / n
    z = norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / n)
    return (float(max(p - z * se, 0.0)), float(min(p + z * se, 1.0)))


@dataclass
class MCResult:
    method: str
    n_evals: int
    n_fail: int
    pf_hat: float
    ci_low: float
    ci_high: float
    alpha: float = 0.05
    extra: dict = field(default_factory=dict)


def monte_carlo(rvs, corr_matrix, limit_fn, n: int, rng,
                alpha: float = 0.05) -> MCResult:
    X = correlated_samples(rvs, corr_matrix, n, rng)
    G = limit_state_values(X, limit_fn)
    fails = failure_indicator(G)
    k = int(fails.sum())
    lo, hi = clopper_pearson(k, n, alpha)
    return MCResult("monte_carlo", n, k, k / n if n else 0.0,
                    lo or 0.0, hi or 1.0, alpha)


def importance_sampling(rvs, corr_matrix, limit_fn, n: int, rng,
                        design_point_shift=None, alpha: float = 0.05) -> MCResult:
    """IS with a user-supplied mean shift (delta method style).

    Weights: ratio of target density to shifted density on the standard
    normal copula scale, computed row-wise from the joint Gaussian logpdf.
    """
    k_vars = len(rvs)
    corr = np.asarray(corr_matrix, dtype=float)
    X = correlated_samples(rvs, corr, n, rng)
    G = limit_state_values(X, limit_fn)
    fails = failure_indicator(G)
    if design_point_shift is None:
        # fallback: plain MC with wider reporting
        kk = int(fails.sum())
        lo, hi = clopper_pearson(kk, n, alpha)
        return MCResult("importance_sampling(uniform-IS)", n, kk,
                        kk / n if n else 0.0, lo or 0.0, hi or 1.0, alpha,
                        {"note": "no shift supplied; identical to MC"})
    # shift on copula scale: use the supplied shift as mean of proposal
    shift = np.asarray(design_point_shift, dtype=float).ravel()
    z = rng.standard_normal((n, k_vars))
    L = np.linalg.cholesky(corr + 1e-12 * np.eye(k_vars))
    z_shift = shift + z
    Xp = np.zeros((n, k_vars))
    for j, rv in enumerate(rvs):
        Xp[:, j] = _ppf(rv, norm.cdf(z_shift @ L.T)[:, j])
    Gp = limit_state_values(Xp, limit_fn)
    failsp = failure_indicator(Gp)
    # weights via multivariate normal logpdf difference
    cov = corr + 1e-12 * np.eye(k_vars)
    log_w = (_mvn_logpdf(z_shift, np.zeros(k_vars), cov)
             - _mvn_logpdf(z_shift, shift, cov))
    w = np.exp(np.clip(log_w, -50, 50))
    est = float(np.sum(failsp * w) / n)
    var_est = float(np.sum((failsp * w) ** 2) / n - est ** 2) / n
    z975 = norm.ppf(1 - alpha / 2)
    sd = np.sqrt(max(var_est, 0.0))
    return MCResult(
        "importance_sampling", 2 * n, int(failsp.sum()), est,
        float(max(est - z975 * sd, 0.0)), float(min(est + z975 * sd, 1.0)),
        alpha, {"shift": shift.tolist(), "ess": float(np.sum(w) ** 2 / np.sum(w ** 2))})


def _mvn_logpdf(x, mean, cov):
    k = x.shape[1]
    diff = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    quad = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return -0.5 * (k * np.log(2 * np.pi) + logdet + quad)


def system_probability(fail_probs, series: bool = True) -> float:
    """Series system: P(any span fails) = 1 - prod(1 - p_i)."""
    p = np.asarray(fail_probs, dtype=float)
    if series:
        return float(1.0 - np.prod(1.0 - p))
    return float(np.prod(p))


def beta_index(pf: float) -> float | None:
    if pf is None or not (0.0 < pf < 1.0):
        return None
    return float(-norm.ppf(pf))

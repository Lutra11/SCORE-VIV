"""Surrogate layer: fidelity adapters, co-kriging-style fusion, prediction
intervals, active learning, and independent test metrics.

The auto-regressive (AR1) multi-fidelity scheme:
    y_HF(x) = rho * y_MF(x) + delta(x)
A Gaussian-process or polynomial ridge represents each level; the fused
model predicts HF cheaply with uncertainty. Coefficients are fitted on
paired design points; models are saved as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class FidelitySpec:
    name: str            # "low" | "medium" | "high"
    cost_per_call: float  # normalized compute cost (arbitrary units)
    noise_std: float = 0.0


DEFAULT_FIDELITIES = (
    FidelitySpec("low", 1.0, 0.02),
    FidelitySpec("medium", 8.0, 0.01),
    FidelitySpec("high", 60.0, 0.0),
)


class _GPLevel:
    """GP surrogate for one level with predictive std."""

    def __init__(self, noise_floor: float = 1e-6):
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(1) * 0.5) \
            + WhiteKernel(noise_level=noise_floor)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=0,
            n_restarts_optimizer=3)
        self.fitted = False

    def fit(self, X, y):
        self.gp.fit(np.asarray(X, dtype=float),
                    np.asarray(y, dtype=float).ravel())
        self.fitted = True
        return self

    def predict(self, X, return_std: bool = False):
        if return_std:
            mu, sd = self.gp.predict(np.asarray(X, dtype=float),
                                     return_std=True)
            return mu, sd
        return self.gp.predict(np.asarray(X, dtype=float))


class AR1MultiFidelity:
    """y_HF = rho * y_MF + delta; delta by GP; MF/LF by GP."""

    def __init__(self, seed: int = 0):
        self.lf = _GPLevel()
        self.mf = _GPLevel()
        self.delta = _GPLevel()
        self.rho = None
        self.seed = int(seed)

    def fit(self, X_lf, y_lf, X_mf, y_mf, X_hf, y_hf):
        self.lf.fit(X_lf, y_lf)
        self.mf.fit(X_mf, y_mf)
        # rho on paired (subset of) MF/HF points
        Xh = np.asarray(X_hf, dtype=float)
        yh = np.asarray(y_hf, dtype=float).ravel()
        mu_mf = self.mf.predict(Xh)
        denom = float(np.sum((mu_mf - mu_mf.mean()) ** 2))
        if denom > 1e-12:
            self.rho = float(np.sum((yh - yh.mean()) * (mu_mf - mu_mf.mean()))
                            / denom)
        else:
            self.rho = 1.0
        resid = yh - self.rho * mu_mf
        self.delta.fit(Xh, resid)
        self.X_hf, self.y_hf = Xh, yh
        return self

    def predict(self, X, return_std: bool = False):
        X = np.asarray(X, dtype=float)
        mu_mf, sd_mf = self.mf.predict(X, return_std=True)
        mu_d, sd_d = self.delta.predict(X, return_std=True)
        mu = self.rho * mu_mf + mu_d
        sd = np.sqrt((self.rho * sd_mf) ** 2 + sd_d**2)
        if return_std:
            return mu, sd
        return mu

    def to_dict(self):
        return {
            "kind": "AR1MultiFidelity",
            "rho": self.rho,
            "n_lf": int(np.asarray(self.X_hf).shape[0]) if hasattr(self, "X_hf") else 0,
        }

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2),
                              encoding="utf-8")


def fusion_table(model: AR1MultiFidelity, X_test, y_test) -> list[dict]:
    mu, sd = model.predict(X_test, return_std=True)
    rows = []
    for i in range(len(y_test)):
        rows.append({
            "x": float(X_test[i][0]) if X_test.ndim > 1 else float(X_test[i]),
            "y_true": float(y_test[i]),
            "y_pred": float(mu[i]),
            "std": float(sd[i]),
            "abs_err": float(abs(mu[i] - y_test[i])),
        })
    return rows


def independent_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test": int(y_true.size),
    }


def coverage(rows, z: float = 1.96) -> float:
    """Empirical coverage of ~95% (z*sigma) intervals."""
    inside = [abs(r["y_pred"] - r["y_true"]) <= z * max(r["std"], 1e-12)
              for r in rows]
    return float(np.mean(inside)) if inside else 0.0


def mse_learning_curve(X, y, fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
                       seed: int = 0) -> list[dict]:
    """Active-learning-style history: RMSE vs training fraction."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] < 8:
        return []
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed)
    rows = []
    for frac in fractions:
        n = max(int(frac * len(X_tr)), 2)
        gp = _GPLevel()
        gp.fit(X_tr[:n], y_tr[:n])
        pred = gp.predict(X_te)
        rows.append({
            "train_fraction": float(frac),
            "n_train": int(n),
            "rmse": float(np.sqrt(np.mean((pred - y_te) ** 2))),
        })
    return rows

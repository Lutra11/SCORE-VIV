"""SCORE-VIV unified experiment protocol (v1.0.0).

Implements docs/EXPERIMENT_PROTOCOL.md:
- per-experiment seed blocks and level->seed-range mapping (section 3)
- independent sub-random streams via SeedSequence (section 3.2)
- oracle-query accounting for QAE simulations (section 6.2)
- run-level statistics + paired Wilcoxon with Holm correction (section 8)
- batch/output layout and manifest helpers (section 10)

The protocol is ENFORCED: runners validate their seed against the declared
level before doing any numerical work, and every estimator result carries
oracle_queries so budget comparisons cannot silently use shot counts.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROTOCOL_NAME = "SCORE-VIV-EXPERIMENT-PROTOCOL"
PROTOCOL_VERSION = "1.0.0"

LEVELS = ("smoke", "pilot", "formal")

# experiment -> base of its 10,000-int seed block (section 3.1)
SEED_BLOCKS = {
    "E1": 110000,
    "E2": 120000,
    "E3": 130000,
    "E4": 140000,
    "E5": 150000,
    "E6": 160000,
    "E7": 170000,
    "E8": 180000,
    "E9": 190000,
}
_BLOCK = 10000

# stream ids (section 3.2)
STREAM_DATA = 10
STREAM_SPLIT = 20
STREAM_INIT = 30
STREAM_ENV = 40
STREAM_ESTIMATOR = 50
STREAM_BOOTSTRAP = 60
STREAM_INDEPENDENT = 90


def run_seed(experiment_id: str, level: str, index: int = 0) -> int:
    """Protocol seed: smoke block / pilot block / formal main interval."""
    base = SEED_BLOCKS[experiment_id]
    if level == "smoke":
        return base + 0 + int(index)          # 110000..110002
    if level == "pilot":
        return base + 100 + int(index)        # 110100..110109
    if level == "formal":
        return base + 1000 + int(index)       # 111000..118999
    raise ValueError(f"level must be one of {LEVELS}: {level}")


def validate_seed(experiment_id: str, seed: int, level: str,
                  index_max: int | None = None) -> None:
    """Raise unless seed sits in the declared level interval."""
    base = SEED_BLOCKS[experiment_id]
    lo, hi = base, base + _BLOCK - 1
    if not (lo <= int(seed) <= hi):
        raise ValueError(
            f"{experiment_id} seed {seed} outside its block [{lo}, {hi}]")
    if level == "smoke":
        rng = (base + 0, base + 2)
    elif level == "pilot":
        rng = (base + 100, base + 109)
    elif level == "formal":
        rng = (base + 1000, base + 8999)
    else:
        raise ValueError(f"level must be one of {LEVELS}: {level}")
    if not (rng[0] <= int(seed) <= rng[1]):
        raise ValueError(
            f"{experiment_id} seed {seed} invalid for level={level} "
            f"(allowed {rng[0]}..{rng[1]})")
    if index_max is not None:
        if int(seed) - rng[0] > int(index_max):
            raise ValueError(
                f"{experiment_id} seed index {int(seed) - rng[0]} exceeds "
                f"declared repetitions-1 = {index_max}")


def stream_rng(seed: int, stream_id: int) -> np.random.Generator:
    """Independent sub-random stream (section 3.2; no python hash())."""
    ss = np.random.SeedSequence([int(seed), int(stream_id)])
    return np.random.default_rng(ss)


# ---- oracle-query accounting (section 6.2) ---------------------------------


def grover_iterations(m: int) -> int:
    return 2 * int(m) + 1


def oracle_queries(counts) -> int:
    """Sum of shots*(2m+1) over (m, k, shots) measurement tuples."""
    total = 0
    for m, _k, n in counts:
        total += int(n) * grover_iterations(int(m))
    return int(total)


# ---- run statistics (section 8) ---------------------------------------------


def summarize_runs(values, n_planned: int | None = None,
                   bootstrap_resamples: int = 10000,
                   rng: np.random.Generator | None = None,
                   confidence: float = 0.95) -> dict:
    """mean/std/median/IQR/min/max + 95% bootstrap CI; failed runs counted
    as the protocol requires (n_valid vs n_planned, no silent deletion)."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    n_valid = int(v.size)
    out = {
        "n_valid": n_valid,
        "n_planned": int(n_planned if n_planned is not None else n_valid),
        "mean": float(v.mean()) if n_valid else None,
        "std": float(v.std(ddof=1)) if n_valid > 1 else None,
        "median": float(np.median(v)) if n_valid else None,
        "iqr": (float(np.percentile(v, 25)), float(np.percentile(v, 75)))
        if n_valid else (None, None),
        "min": float(v.min()) if n_valid else None,
        "max": float(v.max()) if n_valid else None,
        "ci95": [None, None],
    }
    if n_valid >= 2:
        rng = rng or np.random.default_rng(0)
        bs = rng.choice(v, size=(bootstrap_resamples, n_valid), replace=True)
        means = bs.mean(axis=1)
        a = (1.0 - confidence) / 2.0
        out["ci95"] = [float(np.quantile(means, a)),
                       float(np.quantile(means, 1.0 - a))]
    return out


def paired_wilcoxon(a, b, alpha: float = 0.05):
    """Two-sided Wilcoxon signed-rank on paired differences + effect size.

    Returns dict with statistic, p_value, median_diff, and rank-biserial
    effect size; falls back to a sign test note when all differences tie.
    """
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired arrays must be same non-empty shape")
    d = a - b
    nz = d[d != 0]
    if nz.size == 0:
        return {"statistic": None, "p_value": 1.0,
                "median_diff": 0.0, "rank_biserial": 0.0,
                "n_pairs": int(a.size), "note": "all differences zero"}
    res = wilcoxon(a, b, zero_method="wilcox",
                   alternative="two-sided", mode="auto")
    # rank-biserial effect size r = 1 - 2W/(n(n+1)/2) on signed ranks
    n = nz.size
    w = float(res.statistic)
    r_rb = 1.0 - 2.0 * w / (n * (n + 1) / 2.0)
    return {
        "statistic": w,
        "p_value": float(res.pvalue),
        "significant_at_alpha": bool(res.pvalue < alpha),
        "median_diff": float(np.median(d)),
        "rank_biserial": float(np.clip(r_rb, -1.0, 1.0)),
        "n_pairs": int(a.size),
    }


def holm_adjust(pvals):
    """Holm step-down adjusted p-values (section 8)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    m = p.size
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted


# ---- batch layout + manifests (section 10) ---------------------------------


def batch_id(level: str, git_sha: str, date: str) -> str:
    """e.g. formal-20261104-protocol-v1.0.0-0a56c97"""
    return f"{level}-{date}-protocol-v{PROTOCOL_VERSION}-{git_sha}"


def seed_dir(batch: str, experiment_id: str, case_id: str,
             method_id: str, seed: int) -> Path:
    p = Path("outputs") / batch / experiment_id / case_id / method_id
    return p / f"seed_{seed}"


def environment_lock(path: Path) -> dict:
    """OS/CPU/Python/dependency versions for the run record (section 6.3)."""
    try:
        import numpy, scipy, sklearn  # noqa: F401
        deps = {"numpy": numpy.__version__, "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__}
        try:
            import SALib
            deps["SALib"] = SALib.__version__
        except Exception:
            pass
        try:
            import rainflow
            deps["rainflow"] = rainflow.__version__
        except Exception:
            pass
    except Exception:
        deps = {}
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": __import__("os").cpu_count(),
        "dependencies": deps,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def input_manifest(path: Path, entries) -> dict:
    """entries: iterable of (relative_path, sha256, unit, source)."""
    doc = {"inputs": [
        {"path": str(p), "sha256": str(h), "unit": str(u), "source": str(s)}
        for p, h, u, s in entries
    ]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def working_tree_state(repo_root) -> dict:
    """git commit + dirty flag so results carry their exact code state."""
    root = Path(repo_root)
    def _git(*args):
        try:
            r = subprocess.run(["git", *args], cwd=str(root),
                               capture_output=True, text=True, timeout=15)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    return {
        "commit": _git("rev-parse", "HEAD"),
        "short": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


PROTOCOL_BLOCK = {
    "name": PROTOCOL_NAME,
    "version": PROTOCOL_VERSION,
    "seed_policy": "experiment-block-v1",
    "common_random_numbers": True,
    "record_git_commit": True,
    "record_environment": True,
    "hash_algorithm": "sha256",
    "confidence_level": 0.95,
    "alpha": 0.05,
    "bootstrap_resamples": 10000,
    "multiple_comparison": "holm",
    "qae_budget": {
        "initial_shots": 2048,
        "max_shots": 8192,
        "oracle_query_rule": "sum(shots * (2*m + 1))",
    },
    "result_policy": {
        "keep_failed_runs": True,
        "replace_failed_seed": False,
        "independent_final_validation": True,
    },
}


@dataclass
class ProtocolCheck:
    """Formal-admission check record (section 11)."""
    protocol_version_used: str
    config_locked: bool
    prerequisites_met: bool
    tests_passed: bool
    no_placeholders: bool
    correct_public_data: bool
    required_sample_sizes: bool
    fair_budget_accounting: bool
    correct_oracle_formula: bool
    e7_direction_tested: bool
    snapshots_saved: bool
    aggregation_reproducible: bool
    conclusion_scope_honest: bool

    @property
    def admitted(self) -> bool:
        return all([
            self.protocol_version_used == PROTOCOL_VERSION,
            self.config_locked, self.prerequisites_met, self.tests_passed,
            self.no_placeholders, self.correct_public_data,
            self.required_sample_sizes, self.fair_budget_accounting,
            self.correct_oracle_formula, self.e7_direction_tested,
            self.snapshots_saved, self.aggregation_reproducible,
            self.conclusion_scope_honest,
        ])

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["admitted_as_formal"] = self.admitted
        return d

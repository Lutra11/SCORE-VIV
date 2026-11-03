"""Shared runtime helpers: seeds, hashes, git revision, run metadata."""

from __future__ import annotations

import datetime as _dt
import hashlib
import random
import subprocess
from pathlib import Path

import numpy as np

PACKAGE = "score_viv"
PACKAGE_VERSION = "0.1.0"


def make_rng(seed: int) -> np.random.Generator:
    """Deterministic numpy Generator from an integer seed."""
    return np.random.default_rng(int(seed))


def seed_everything(seed: int) -> np.random.Generator:
    """Seed python and numpy legacy RNG, return a modern Generator."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    return make_rng(seed)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes(paths, root) -> dict:
    """SHA-256 of existing files (paths relative to root)."""
    root = Path(root)
    out = {}
    for p in paths:
        f = (root / p) if not Path(p).is_absolute() else Path(p)
        if f.is_file():
            out[str(p)] = _sha256(f)
    return out


def git_revision(root=".") -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def run_metadata(experiment_id, config, seed, output_dir, repo_root,
                 inputs=None, extra=None) -> dict:
    """Standard run metadata block written next to every experiment output."""
    meta = {
        "experiment_id": experiment_id,
        "package": PACKAGE,
        "package_version": PACKAGE_VERSION,
        "python": f"{__import__('sys').version.split()[0]}",
        "numpy": np.__version__,
        "config": config,
        "seed": int(seed),
        "output_dir": str(output_dir),
        "git_revision": git_revision(repo_root),
        "started_utc": now_iso(),
        "input_hashes": input_hashes(inputs or [], repo_root),
    }
    if extra:
        meta.update(extra)
    return meta

"""Repository-layout contract for the GitHub experiment repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    "E1_fem_viv_validation",
    "E2_scour_validation",
    "E3_scour_fem_viv_coupling",
    "E4_fatigue_validation",
    "E5_multifidelity_surrogate",
    "E6_reliability_na_qae",
    "E7_mitigation_optimization",
    "E8_ablation",
    "E9_sensitivity",
)


def test_required_top_level_areas_exist() -> None:
    required = {
        ".github/workflows",
        "dataset/raw",
        "dataset/digitized",
        "dataset/processed",
        "dataset/generated",
        "dataset/manifests",
        "algorithm/score_viv/common",
        "algorithm/score_viv/fem",
        "algorithm/score_viv/scour",
        "algorithm/score_viv/viv",
        "algorithm/score_viv/fatigue",
        "algorithm/score_viv/surrogate",
        "algorithm/score_viv/reliability",
        "algorithm/score_viv/optimizers/abc",
        "algorithm/score_viv/optimizers/na_qae",
        "algorithm/score_viv/optimizers/na_qae_abc",
        "algorithm/score_viv/costs",
        "algorithm/score_viv/mitigation",
        "experiments",
        "requirements",
        "scripts",
        "tests",
    }
    missing = sorted(path for path in required if not (ROOT / path).exists())
    assert missing == []


def test_every_experiment_has_readme_config_runner_and_output_contract() -> None:
    for experiment in EXPERIMENTS:
        base = ROOT / "experiments" / experiment
        assert (base / "README.md").is_file(), experiment
        assert (base / "configs" / "default.yaml").is_file(), experiment
        assert (base / "run.py").is_file(), experiment
        assert (base / "outputs" / "README.md").is_file(), experiment


def test_every_experiment_runner_supports_safe_dry_run() -> None:
    for experiment in EXPERIMENTS:
        runner = ROOT / "experiments" / experiment / "run.py"
        completed = subprocess.run(
            [sys.executable, str(runner), "--dry-run"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert experiment.split("_")[0] in completed.stdout
        assert "DRY_RUN" in completed.stdout


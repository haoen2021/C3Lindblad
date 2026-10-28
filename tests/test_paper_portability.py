"""The comparison scripts consume and produce data inside an external workdir."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("calculation", ["mbpt_green", "mbpt_static"])
def test_moved_mbpt_comparison_reads_workdir_data(tmp_path, calculation):
    scripts = tmp_path / "moved calculation scripts"
    shutil.copytree(Path(__file__).resolve().parents[1] / "examples", scripts)
    work = tmp_path / "work"
    inputs = work / "results/inchmoss_fig6_mbpt_v1"
    inputs.mkdir(parents=True)
    # The continuum FFT uses an even grid with the upper endpoint excluded.
    omega = np.linspace(-8.0, 8.0, 64, endpoint=False)
    green = np.eye(2)[None] / (omega[:, None, None] + 0.5j)
    for method in ("2B", "GW"):
        np.savez(inputs / f"{method}_refined.npz", omega=omega, G_R=green,
                 defect_history=np.array([1e-10]))
    result = subprocess.run(
        [sys.executable, str(scripts / "run.py"), "--work-dir", str(work),
         f"inchmoss.comparison.{calculation}"],
        cwd=tmp_path, capture_output=True, text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert result.returncode == 0, result.stderr
    outputs = work / "inchmoss/comparison/results"
    for method in ("2B", "GW"):
        if calculation == "mbpt_green":
            with np.load(outputs / f"greens/{method}/greens.npz") as saved:
                assert saved["G_tau"].shape == (501, 2, 2)
                assert np.all(np.isfinite(saved["G_tau"]))
        else:
            assert (outputs / f"{method}.json").is_file()
    assert not list(scripts.rglob("*.npz"))

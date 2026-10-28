"""Half-filling driver defaults and compatibility with the release launcher."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_release_launcher_exposes_half_filling_response(tmp_path):
    # Absolute ``examples`` imports break the launcher's private package alias.
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "examples/run.py"),
         "--work-dir", str(tmp_path), "kanamori.half_filling", "--help"],
        cwd=tmp_path, env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "S13" in result.stdout and "S135" in result.stdout


def test_release_driver_rejects_larger_cap_before_starting(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "examples/run.py"),
         "--work-dir", str(tmp_path), "kanamori.half_filling", "--beta", "8",
         "--bath", str(tmp_path / "bath.npz"), "--output", str(output),
         "--cap", "2048"],
        cwd=tmp_path, env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "--cap" in result.stderr and "1024" in result.stderr
    assert not output.exists()


@pytest.fixture
def bath(tmp_path):
    path = tmp_path / "bath.npz"
    arrays = {}
    for branch in ("lesser", "greater"):
        arrays[f"H_{branch}"] = np.diag(np.linspace(-1., 1., 5))
        arrays[f"damping_{branch}"] = np.diag(np.linspace(.4, .8, 5))
        arrays[f"g_{branch}"] = np.linspace(.1, .2, 5)
    np.savez(path, **arrays)
    return path


def test_default_model_is_half_filled_and_preserves_bath(bath):
    from examples.kanamori.half_filling import build_model

    half_filled = build_model(bath)
    chemical_potential = build_model(bath, convention="chemical_potential")
    np.testing.assert_allclose(half_filled.E[:2, :2], -2.5 * np.eye(2))
    np.testing.assert_array_equal(half_filled.E[2:, :], chemical_potential.E[2:, :])
    np.testing.assert_array_equal(half_filled.E[:2, 2:], chemical_potential.E[:2, 2:])
    np.testing.assert_array_equal(half_filled.gamma_loss, chemical_potential.gamma_loss)
    np.testing.assert_array_equal(half_filled.gamma_gain, chemical_potential.gamma_gain)


def test_quadratic_reconstruction_uses_half_filled_impurity(bath):
    from examples.kanamori.half_filling import build_model, reconstruct

    model = build_model(bath)
    shifts = np.array([-.4 + .05j, .3 + .05j, .7j])
    generator = model.E - 1j * (model.gamma_loss + model.gamma_gain)
    raw = np.asarray([
        np.linalg.inv(z * np.eye(model.n_sites) - generator)[:2, :2]
        for z in shifts
    ])
    observed, self_energy = reconstruct(model, raw, shifts)
    # Independent analytic resolvent of a D=2 semicircular bath, t=1.
    root = np.sqrt(shifts**2 - 4)
    root = np.where(root.imag < 0, -root, root)
    bethe = (shifts - root) / 2
    expected = np.asarray([
        np.linalg.inv((z + 2.5) * np.eye(2) - g * np.ones((2, 2)))
        for z, g in zip(shifts, bethe)
    ])
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(self_energy, 0, atol=1e-12, rtol=0)

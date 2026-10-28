"""Check scientific conventions and movable, JSON-driven paper calculations."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def test_moved_exchange_example_uses_adjacent_parameters(tmp_path):
    scripts = tmp_path / "renamed examples"
    shutil.copytree(Path(__file__).resolve().parents[1] / "examples", scripts)
    path = scripts / "parameters.json"
    parameters = json.loads(path.read_text())
    parameters["exchange_model"]["epsilon"] = 7.25
    path.write_text(json.dumps(parameters))
    work = tmp_path / "work"
    run = subprocess.run(
        [
            sys.executable,
            str(scripts / "run.py"),
            "--work-dir",
            str(work),
            "exchange_model",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert run.returncode == 0, run.stderr
    with np.load(work / "results/exchange_model/responses.npz") as saved:
        z = saved["omega"] + 0.1j
        expected = 1 / (z - 7.25 - 0.25 / (z - 7.25))
        np.testing.assert_allclose(saved["Exact"], expected)
    assert not list(scripts.rglob("*.npz"))


def test_junction_transfer_strengths_against_fermionic_operators():
    from examples.shared.fock import fock_annihilators
    from examples.junction.observables import strengths

    a = fock_annihilators(4)
    n = [x.conj().T @ x for x in a]
    nh, nl = n[0] + n[1], n[2] + n[3]
    szh, szl = (n[0] - n[1]) / 2, (n[2] - n[3]) / 2
    sph, spl = a[0].conj().T @ a[1], a[2].conj().T @ a[3]
    dot = szh @ szl + (sph @ spl.conj().T + sph.conj().T @ spl) / 2
    pauli = [
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.diag([1, -1]),
    ]
    transfer = [
        sum(m[i, j] * a[i].conj().T @ a[2 + j] for i in range(2) for j in range(2))
        / np.sqrt(2)
        for m in pauli
    ]
    scalar = transfer[0].conj().T @ transfer[0]
    vector = sum(b.conj().T @ b for b in transfer[1:])
    np.testing.assert_allclose(scalar, nl / 2 - nh @ nl / 4 - dot, atol=1e-14)
    np.testing.assert_allclose(vector, 3 * nl / 2 - 3 * nh @ nl / 4 + dot, atol=1e-14)
    rng = np.random.default_rng(55)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    mean = lambda op: float(np.vdot(state, op @ state).real)
    measured = strengths(mean(nh), mean(nl), mean(nh @ nl), mean(dot))
    np.testing.assert_allclose(
        [measured["W0"], measured["W1"]], [mean(scalar), mean(vector)]
    )


def test_junction_restart_identity_changes_with_model(monkeypatch):
    from examples.parameters import parameters
    from examples.junction import model

    p = parameters("junction")
    monkeypatch.setattr(model, "parameters", lambda case: p)
    assert model.input_fingerprint() == p["archived_input_sha256"]
    p["model"]["one_body_matrix"][0][0] += 0.1
    assert model.input_fingerprint() != p["archived_input_sha256"]


def test_siam_reuse_rejects_changed_parameters(tmp_path, monkeypatch):
    import pytest
    from examples.siam_modes import example04

    example04.check_parameters(tmp_path)
    changed = dict(example04.PARAMETERS, beta=2.0)
    monkeypatch.setattr(example04, "PARAMETERS", changed)
    with pytest.raises(ValueError, match="fresh --output"):
        example04.check_parameters(tmp_path)

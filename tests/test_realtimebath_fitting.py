"""RealTimeBath fitting and physical correlation conventions."""

import sys

import numpy as np
import pytest

from c3l import fitting
from c3l.pseudomode import realization_bcf


def test_one_mode_realization_has_the_analytic_branch_correlations():
    from c3l.pseudomode import PseudomodeRealization

    times = np.asarray([0.0, 0.3, 1.0])
    realization = PseudomodeRealization(
        H=np.asarray([[0.7]]), damping=np.asarray([[0.2]]), g=np.asarray([0.5j])
    )
    assert realization.size == 1
    np.testing.assert_allclose(
        realization_bcf(times, realization, "lesser"),
        0.25 * np.exp((-0.2 + 0.7j) * times),
    )
    np.testing.assert_allclose(
        realization_bcf(times, realization, "greater"),
        0.25 * np.exp((-0.2 - 0.7j) * times),
    )








def test_missing_realtimebath_has_installation_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "realtimebath", None)
    times = np.linspace(0, 1, 9)
    with pytest.raises(ImportError, match=r"c3l\[fitting\]"):
        fitting.fit_fermionic_bath(times, np.exp(-times), np.exp(-times), n_modes=1)


@pytest.mark.parametrize("times,values,modes", [([0, 1, 0.5], [1, 1, 1], 1), ([0, 0.5, 1], [1, np.nan, 1], 1), ([0, 0.5, 1], [[1], [1], [1]], 1), ([0, 0.5, 1], [1, 1, 1], True)])
def test_invalid_fit_input_is_rejected_before_backend(times, values, modes, monkeypatch):
    monkeypatch.setitem(sys.modules, "realtimebath", None)
    with pytest.raises(ValueError):
        fitting.fit_fermionic_bath(times, values, values, n_modes=modes)


@pytest.mark.parametrize("options", [{}, {"optimize": False}])
def test_real_backend_preserves_branch_correlations(options):
    pytest.importorskip("realtimebath")
    times = np.linspace(0, 4, 41)
    lesser = 0.3 * np.exp((-0.2 - 0.4j) * times)
    greater = 0.7 * np.exp((-0.35 + 0.25j) * times)
    result = fitting.fit_fermionic_bath(times, 1j * lesser, -1j * greater, n_modes=1, convention="negf", max_nfev=100, **options)
    np.testing.assert_allclose(realization_bcf(times, result.lesser, "lesser"), lesser, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(realization_bcf(times, result.greater, "greater"), greater, atol=2e-6, rtol=2e-6)
    assert result.backend == "realtimebath"
    assert set(result.diagnostics) == {"lesser", "greater"}

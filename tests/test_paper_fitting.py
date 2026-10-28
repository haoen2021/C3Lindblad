"""Paper fit entry points preserve both physical correlation branches."""

import numpy as np
import pytest

from c3l.pseudomode import realization_bcf


@pytest.mark.parametrize('case', ['kanamori', 'inchmoss'])
def test_paper_fit_recovers_one_mode_branches(case, monkeypatch):
    pytest.importorskip('realtimebath')
    if case == 'kanamori':
        from examples.kanamori import run
        arguments = (4.0, 1)
    else:
        from examples.inchmoss import run
        arguments = (1,)

    def correlation(time, branch, **kwargs):
        exponent = -0.2 + 0.4j if branch == 'lesser' else -0.35 - 0.25j
        weight = 0.3 if branch == 'lesser' else 0.7
        return weight * np.exp(exponent * time)

    monkeypatch.setattr(run, 'scalar_bcf', correlation)
    result = run.fit_scalar_bath(*arguments, training_points=41, validation_points=51, max_nfev=100)
    time = np.linspace(0, result.fit_end, 51)
    for branch in ('lesser', 'greater'):
        np.testing.assert_allclose(realization_bcf(time, getattr(result, branch), branch), correlation(time, branch), atol=2e-6, rtol=2e-6)
    assert result.relative_l2_error < 2e-6
    assert result.equal_time_relative_error < 2e-6

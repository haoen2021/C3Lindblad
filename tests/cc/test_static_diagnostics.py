"""Optional rank-six diagnostics must not add work to ordinary CCSD solves."""

import numpy as np

from c3l import FermionicLindbladModel
from c3l.cc import component_solver
from c3l.cc.component_tensor import ComponentStaticProblem
from c3l.cc.factorized import factorized_ccsd_problem


def test_default_solve_skips_rank_six(monkeypatch):
    model = FermionicLindbladModel(
        np.array([[0.0, 0.2], [0.2, 0.1]]),
        np.diag([0.0, 0.3]),
        np.diag([0.0, 0.2]),
        U=0.0,
    )
    problem = ComponentStaticProblem.compile(factorized_ccsd_problem(model))

    def forbidden(*args, **kwargs):
        raise AssertionError("An ordinary static solve must not evaluate rank six")

    monkeypatch.setattr(
        component_solver, "component_static_rank6_diagnostic", forbidden
    )
    solution = component_solver.solve_component_ccsd(problem)
    assert solution.converged
    assert solution.rank6 is None


def test_explicit_shell_request_requires_coordinates():
    import pytest
    from c3l.cc.factorized import factorized_ccsd_residual, factorized_ccsd_jvp

    model = FermionicLindbladModel(
        np.array([[0.0, 0.2], [0.2, 0.1]]),
        np.diag([0.0, 0.3]), np.diag([0.0, 0.2]), U=0.0,
    )
    problem = factorized_ccsd_problem(model)
    amplitudes = np.zeros(problem.num_amplitudes, dtype=complex)
    with pytest.raises(ValueError, match="build_next_rank_masks=True"):
        factorized_ccsd_residual(problem, amplitudes, include_next_rank=True)
    with pytest.raises(ValueError, match="build_next_rank_masks=True"):
        factorized_ccsd_jvp(problem, amplitudes, amplitudes, include_next_rank=True)

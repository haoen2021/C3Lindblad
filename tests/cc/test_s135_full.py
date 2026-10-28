"""Exact full-rank-five response action without selected-space machinery."""


import numpy as np

from c3l.cc.component_tensor import (
    ComponentEOMProblem,
    ComponentStaticProblem,
    component_block_action,
    component_eom_context,
)
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import factorized_cc_problem
from c3l.cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator
from tests.s135_fixture import s135_validation_model






def test_full_s135_right_action_matches_explicit_response_matrix() -> None:
    """Every retained/rank-five block agrees with the small dense oracle."""

    cc = factorized_cc_problem(s135_validation_model(), "CCSD", m_up=0.3)
    static = ComponentStaticProblem.compile(cc)
    coordinate = np.arange(cc.num_variables, dtype=np.float64)
    T = cc.tilde.unpack(6.0e-4 * np.cos(coordinate + 0.19))
    eom = eom_ccsd_problem(cc, response_spin=0, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, 1)
    context = component_eom_context(problem, T)
    operator = FullS135ResponseOperator(
        CompiledFullS135Action(context, materialization_batch_size=4),
        s5_tile_size=7,
    )
    exact = component_block_action(
        context,
        operator.coordinates,
        operator.coordinates,
        batch_size=11,
    )
    rng = np.random.default_rng(20260722)
    x = rng.normal(size=operator.dimension) + 1j * rng.normal(
        size=operator.dimension
    )

    np.testing.assert_allclose(operator.matvec(x), exact @ x, rtol=3e-11, atol=3e-11)

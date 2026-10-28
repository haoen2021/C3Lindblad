"""Small charge sectors can saturate without changing the CC truncation."""

import numpy as np

from c3l import FermionicLindbladModel
from c3l.cc.component_response import ComponentSigmaOperator, component_retarded_matrix_green
from c3l.cc.component_solver import solve_component_ccsd
from c3l.cc.component_tensor import (
    ComponentEOMProblem, ComponentStaticProblem, component_eom_context, component_eom_sources,
)
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import factorized_cc_problem
from c3l.cc.observables import factorized_ccsd_covariance, low_rank_cc_coefficients
from tests.cc.test_charge_s13 import charge_model


def test_single_mode_ccsd_and_s13_match_exact_quadratic_result():
    model = FermionicLindbladModel([[0.3]], [[0.8]], [[0.2]], U=0.0, symmetry="charge")
    oracle = factorized_cc_problem(model, "CCSD")
    assert oracle.num_amplitudes == 1
    assert oracle.retained_creator_rank == 4
    assert oracle.next_rank_creator_rank == 6
    assert oracle.next_rank_dimension == 0
    static = ComponentStaticProblem.compile(oracle)
    solution = solve_component_ccsd(static)
    assert solution.converged
    np.testing.assert_allclose(
        factorized_ccsd_covariance(oracle.reference, oracle.cluster_basis, solution.amplitudes),
        [[0.2]], atol=1e-12,
    )
    response = eom_ccsd_problem(oracle)
    context = component_eom_context(ComponentEOMProblem.compile(response, static, charge=1), solution.amplitudes)
    frequencies = np.array([-0.4, 0.1, 0.7])
    observed = component_retarded_matrix_green(
        ComponentSigmaOperator(context), [component_eom_sources(context, site=0)],
        frequencies, 0.1, rtol=1e-12, restart=2, maxiter=10,
    )
    np.testing.assert_allclose(observed.retarded[:, 0, 0], 1 / (frequencies - 0.3 + 1.1j), atol=1e-12)


def test_complete_small_sectors_are_reported_truthfully():
    oracle = factorized_cc_problem(charge_model(), "CCSD")
    record = oracle.construction_record()
    assert record["maximum_basis_dimension_constructed"] == record["complete_charge_zero_sector_dimension"] == 6
    assert record["complete_sector_basis_constructed"] is True
    assert record["complete_sector_state_vector_constructed"] is False
    response_record = eom_ccsd_problem(oracle).construction_record()
    assert response_record["maximum_basis_dimension_constructed"] == response_record["complete_odd_union_dimension"] == 8
    assert response_record["complete_odd_sector_constructed"] is True
    state = low_rank_cc_coefficients(oracle.reference, oracle.cluster_basis, np.zeros(oracle.num_amplitudes, complex))
    assert state.dimension == 6
    assert state.to_dict()["complete_sector_constructed"] is True

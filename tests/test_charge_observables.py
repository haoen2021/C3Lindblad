"""Physical covariances from the rank-limited coupled-cluster state."""

import numpy as np
import pytest

from c3l.cc.observables import factorized_ccsd_covariance, low_rank_cc_coefficients
from c3l.cc.basis import ccsd_basis
from c3l.ci.reference import full_hartree_fock_reference
from c3l.ci.sectors import mode_layout_from_references
from c3l.models import FermionicLindbladModel
from c3l.quadratic_ness import finite_pm_covariance
from tests.cc.test_charge_s13 import exact_lindblad, solved_charge


@pytest.fixture(scope="module", params=[0.0, 0.4])
def charge_solution(request):
    return solved_charge(request.param)


def test_charge_covariance_and_filling_match_physical_ed(charge_solution):
    model, static, solution, _, _ = charge_solution
    density, _, _, annihilation = exact_lindblad(model)
    expected = np.asarray([
        [np.trace(creator.conj().T @ annihilator @ density) for creator in annihilation]
        for annihilator in annihilation
    ])
    observed = factorized_ccsd_covariance(
        static.oracle.reference, static.oracle.cluster_basis, solution.amplitudes
    )
    assert observed.shape == (2, 2)
    assert abs(expected[0, 1].imag) > 1e-4
    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.trace(observed), np.trace(expected), atol=1e-10)


def _quadratic_reference(symmetry, num_sites):
    energy = np.diag(np.linspace(-0.2, 0.3, num_sites)).astype(complex)
    if num_sites > 1:
        energy[0, 1], energy[1, 0] = 0.1j, -0.1j
    model = FermionicLindbladModel(
        energy, np.diag(np.linspace(0.8, 0.6, num_sites)),
        np.diag(np.linspace(0.2, 0.4, num_sites)), U=0.0, symmetry=symmetry,
    )
    gamma, _ = finite_pm_covariance(
        model.physical_E, model.physical_gamma_loss, model.physical_gamma_gain
    )
    reference = full_hartree_fock_reference(model, model.physical_E, gamma)
    layout = mode_layout_from_references(*reference.mode_references)
    cluster = ccsd_basis(layout.mode_charges)
    return reference, cluster, np.zeros(cluster.dimension, dtype=complex)


@pytest.mark.parametrize("symmetry,num_sites", [("charge", 1), ("charge", 3), ("spin", 1), ("spin", 2)])
def test_generic_covariance_uses_actual_modes_without_requiring_a_bath(symmetry, num_sites):
    reference, cluster, amplitudes = _quadratic_reference(symmetry, num_sites)
    state = low_rank_cc_coefficients(reference, cluster, amplitudes)
    assert state.basis.layout.num_modes == reference.num_ph_modes
    observed = factorized_ccsd_covariance(reference, cluster, amplitudes)
    assert observed.shape == (reference.model.n_spin_orbitals,) * 2
    np.testing.assert_allclose(observed, reference.gamma, atol=1e-12)

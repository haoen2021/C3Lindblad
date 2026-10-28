"""Charge models reject spin-only reference and response requests."""

import numpy as np
import pytest

from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import factorized_cc_problem
from c3l.ci.matrix_free import physical_mode_operator
from c3l.ci.reference import full_hartree_fock_reference
from c3l.models import FermionicLindbladModel


@pytest.fixture
def charge_model():
    return FermionicLindbladModel(
        np.diag([0.13, -0.21]), 0.7 * np.eye(2), 0.3 * np.eye(2),
        U=0.0, symmetry="charge",
    )


def test_charge_factorized_problem_rejects_legacy_spin_reference(charge_model):
    with pytest.raises(ValueError, match="charge"):
        factorized_cc_problem(charge_model, "CCSD", reference_mode="legacy_half_filled")


def test_charge_eom_rejects_second_spin(charge_model):
    problem = factorized_cc_problem(charge_model, "CCSD")
    with pytest.raises(ValueError, match="response_spin.*0"):
        eom_ccsd_problem(problem, response_spin=1)


@pytest.mark.parametrize("channel", ["c", "c_dagger", "tilde_c", "tilde_c_dagger"])
def test_charge_physical_operator_rejects_second_spin(charge_model, channel):
    reference = full_hartree_fock_reference(
        charge_model, charge_model.E, 0.3 * np.eye(2)
    )
    with pytest.raises(ValueError, match="charge.*spin=0"):
        physical_mode_operator(reference, site=0, spin=1, channel=channel)

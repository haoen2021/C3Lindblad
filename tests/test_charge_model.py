"""Physical charge-only models and their full Gaussian references."""

from dataclasses import replace

import numpy as np
import pytest

from c3l.ci.reference import (
    full_hartree_fock_reference,
    half_filled_hartree_fock_reference,
)
from tests.reference.hf_operators import (
    global_gaussian_vacuum,
    global_particle_hole_operators,
    reference_decomposition_paper_K,
)
from c3l.gaussian_mean_field import solve_ghf
from c3l.interactions import SparseTwoBodyInteraction
from c3l.models import FermionicLindbladModel, QuadraticOpenSystem, homotopy_model
from tests.reference.exact_generator import (
    auxiliary_exact_inputs,
    physical_generator_indexed,
    physical_generator_jump_factorized,
)
from c3l.quadratic_ness import finite_pm_covariance
from tests.reference.generator import printed_quadratic_paper_K, superfermion_paper_K
from tests.reference.mapping import fermionic_mapping


def test_charge_mode_preserves_physical_orbital_space():
    energy = np.array([[0.1, 0.2j], [-0.2j, -0.3]])
    loss = np.diag([0.7, 0.4])
    gain = np.diag([0.2, 0.5])
    model = FermionicLindbladModel(
        energy, loss, gain, U=0.0, symmetry="charge"
    )

    assert model.symmetry == "charge"
    assert model.n_sites == model.n_spin_orbitals == 2
    np.testing.assert_array_equal(model.physical_E, energy)
    np.testing.assert_array_equal(model.physical_gamma_loss, loss)
    np.testing.assert_array_equal(model.physical_gamma_gain, gain)


def _charge_model(num_modes=2, interaction=0.0):
    energy = np.diag(np.linspace(-0.25, 0.4, num_modes)).astype(complex)
    energy[0, 1] = 0.13 + 0.19j
    energy[1, 0] = energy[0, 1].conjugate()
    loss = np.diag(np.linspace(0.7, 0.4, num_modes)).astype(complex)
    gain = np.diag(np.linspace(0.2, 0.5, num_modes)).astype(complex)
    loss[0, 1], loss[1, 0] = 0.04j, -0.04j
    gain[0, 1], gain[1, 0] = 0.03, 0.03
    return FermionicLindbladModel(
        energy, loss, gain, U=0.0, symmetry="charge",
        two_body_interaction=(
            SparseTwoBodyInteraction(num_modes, ((0, 1, 0, 1, interaction),))
            if interaction else None
        ),
    )


@pytest.mark.parametrize("num_modes", [2, 3])
def test_charge_matrices_indices_and_serialization_keep_physical_space(num_modes):
    model = _charge_model(num_modes, interaction=0.2)
    assert model.n_sites == model.n_spin_orbitals == num_modes
    assert model.interaction_sites == (0, 1)
    assert model.interaction.n_spin_orbitals == num_modes
    for orbital in range(num_modes):
        assert model.orbital_index(orbital, 0) == orbital
        assert model.orbital_site_spin(orbital) == (orbital, 0)
    with pytest.raises(ValueError):
        model.orbital_index(0, 1)
    with pytest.raises(ValueError):
        model.orbital_site_spin(num_modes)
    for name in ("E", "gamma_loss", "gamma_gain"):
        np.testing.assert_array_equal(getattr(model, "physical_" + name), getattr(model, name))
    restored = FermionicLindbladModel.from_dict(model.to_dict())
    assert restored.symmetry == "charge"
    assert restored.interaction == model.interaction
    np.testing.assert_array_equal(restored.physical_E, model.E)


def test_spin_default_and_legacy_serialization_keep_spin_lifting():
    model = FermionicLindbladModel([[0.0]], [[0.7]], [[0.3]], U=0.2)
    assert model.symmetry == "spin"
    assert model.n_spin_orbitals == 2
    assert model.orbital_site_spin(1) == (0, 1)
    for name in ("E", "gamma_loss", "gamma_gain"):
        np.testing.assert_array_equal(
            getattr(model, "physical_" + name), np.kron(getattr(model, name), np.eye(2))
        )
    legacy = model.to_dict()
    legacy.pop("symmetry", None)
    assert FermionicLindbladModel.from_dict(legacy).symmetry == "spin"


def test_charge_rejects_legacy_interactions_and_mismatched_support():
    model = _charge_model(3)
    with pytest.raises(ValueError, match="symmetry"):
        replace(model, symmetry="invalid")
    with pytest.raises(ValueError, match="charge"):
        replace(model, U=0.2)
    with pytest.raises(ValueError, match="charge"):
        replace(model, density_interaction=[[0.2]])
    with pytest.raises(ValueError, match="space"):
        replace(model, two_body_interaction=SparseTwoBodyInteraction(6, ()))
    with pytest.raises(ValueError, match="interaction_sites"):
        replace(model, two_body_interaction=SparseTwoBodyInteraction(3, ((0, 2, 0, 2, 0.2),)))
    with pytest.raises(ValueError, match="charge"):
        half_filled_hartree_fock_reference(model, 0.5)


@pytest.mark.parametrize("interaction", [0.0, 0.2])
def test_charge_homotopy_preserves_physical_layout(interaction):
    model = _charge_model(3, interaction)
    energy = model.E + 0.1 * np.eye(3)
    intermediate = homotopy_model(model, energy, 0.4)
    assert intermediate.symmetry == "charge"
    assert intermediate.n_spin_orbitals == 3
    np.testing.assert_allclose(intermediate.E, energy + 0.4 * (model.E - energy))
    assert intermediate.interaction.coefficient(0, 1, 0, 1) == pytest.approx(0.4 * interaction)


@pytest.mark.parametrize("num_modes", [2, 3])
def test_charge_physical_oracles_preserve_dimensions_and_generator(num_modes):
    model = _charge_model(num_modes)
    hamiltonian, annihilation, loss, gain = auxiliary_exact_inputs(model)
    assert hamiltonian.shape == (2**num_modes,) * 2
    assert len(annihilation) == num_modes
    np.testing.assert_array_equal(loss, model.gamma_loss)
    np.testing.assert_array_equal(gain, model.gamma_gain)
    quadratic = QuadraticOpenSystem(model.E, loss, gain)
    generator = physical_generator_indexed(model)
    np.testing.assert_allclose(generator, physical_generator_indexed(quadratic), atol=1e-14)
    np.testing.assert_allclose(generator, physical_generator_jump_factorized(model), atol=1e-14)
    mapping = fermionic_mapping(num_modes)
    np.testing.assert_allclose(superfermion_paper_K(model), mapping @ (1j * generator) @ mapping.conj().T, atol=1e-13)


def test_charge_interacting_hamiltonian_and_superfermion_oracle():
    model = _charge_model(interaction=0.2)
    hamiltonian, _, _, _ = auxiliary_exact_inputs(model)
    quadratic_hamiltonian, _, _, _ = auxiliary_exact_inputs(_charge_model())
    np.testing.assert_allclose(hamiltonian - quadratic_hamiltonian, np.diag([0, 0, 0, 0.2]), atol=1e-14)
    mapping = fermionic_mapping(2)
    np.testing.assert_allclose(superfermion_paper_K(model), mapping @ (1j * physical_generator_indexed(model)) @ mapping.conj().T, atol=1e-13)


@pytest.mark.parametrize("num_modes,interaction", [(2, 0.0), (3, 0.0), (2, 0.2)])
def test_charge_reference_retains_coherence_and_dense_vacuum(num_modes, interaction):
    model = _charge_model(num_modes, interaction)
    solved = solve_ghf(model.E, model.interaction, model.gamma_loss, model.gamma_gain, tolerance=1e-12)
    assert solved.converged
    reference = full_hartree_fock_reference(model, solved.fock, solved.gamma)
    assert reference.down is None and reference.E_down is None
    assert reference.mode_references == (reference.up,)
    assert len(reference.one_body_blocks) == 1
    assert reference.num_ph_modes == 2 * num_modes
    np.testing.assert_allclose(reference.one_body_blocks[0], solved.fock)
    assert abs(reference.gamma[0, 1]) > 1e-3
    annihilators, creators = global_particle_hole_operators(reference)
    assert len(annihilators) == len(creators) == 2 * num_modes
    assert annihilators[0].shape == (4**num_modes,) * 2
    vacuum = global_gaussian_vacuum(reference)
    mapping = fermionic_mapping(num_modes)
    density = (mapping.conj().T @ vacuum.state).reshape((2**num_modes,) * 2, order="F")
    _, physical_annihilators, _, _ = auxiliary_exact_inputs(model)
    covariance = np.asarray([
        [np.trace(physical_annihilators[col].conj().T @ physical_annihilators[row] @ density) for col in range(num_modes)]
        for row in range(num_modes)
    ])
    np.testing.assert_allclose(covariance, solved.gamma, atol=1e-10)
    np.testing.assert_allclose(reference_decomposition_paper_K(reference), printed_quadratic_paper_K(model), atol=1e-10)
    with pytest.raises(ValueError):
        replace(reference, E_down=reference.E_up)
    with pytest.raises(ValueError):
        replace(reference, fock=reference.fock + np.eye(num_modes) * 0.1)


def test_charge_reference_uses_full_reference_only_dissipation():
    model = _charge_model(3)
    loss = model.gamma_loss + 0.1 * np.eye(3)
    gain = model.gamma_gain + 0.2 * np.eye(3)
    gamma, _ = finite_pm_covariance(model.E, loss, gain)
    reference = full_hartree_fock_reference(model, model.E, gamma, reference_gamma_loss=loss, reference_gamma_gain=gain)
    np.testing.assert_allclose(reference.reference_gamma_loss, loss)
    np.testing.assert_allclose(reference.gamma, gamma)
    np.testing.assert_allclose(reference_decomposition_paper_K(reference), printed_quadratic_paper_K(model), atol=1e-10)


def test_one_mode_charge_model_has_empty_interaction_and_exact_reference():
    interaction = SparseTwoBodyInteraction(1, ())
    assert interaction.coefficient(0, 0, 0, 0) == 0.0
    model = FermionicLindbladModel([[0.3]], [[0.8]], [[0.2]], U=0.0, symmetry="charge")
    assert model.interaction == interaction
    assert not model.has_interactions
    restored = FermionicLindbladModel.from_dict(replace(model, two_body_interaction=interaction).to_dict())
    assert restored.interaction == interaction
    hamiltonian, annihilation, _, _ = auxiliary_exact_inputs(model)
    assert len(annihilation) == 1
    np.testing.assert_allclose(hamiltonian, np.diag([0.0, 0.3]), atol=1e-14)
    reference = full_hartree_fock_reference(model, model.E, [[0.2]])
    vacuum = global_gaussian_vacuum(reference)
    density = (fermionic_mapping(1).conj().T @ vacuum.state).reshape((2, 2), order="F")
    np.testing.assert_allclose(density, np.diag([0.8, 0.2]), atol=1e-12)
    np.testing.assert_allclose(physical_generator_indexed(model) @ density.reshape(-1, order="F"), 0, atol=1e-12)
    with pytest.raises(ValueError, match="distinct"):
        SparseTwoBodyInteraction(1, ((0, 0, 0, 0, 0.2),))


@pytest.mark.parametrize("num_modes", [0, -1, True, 1.5])
def test_sparse_interaction_rejects_invalid_mode_count(num_modes):
    with pytest.raises(ValueError, match="n_spin_orbitals"):
        SparseTwoBodyInteraction(num_modes, ())

"""Small Fock-space oracles for the canonical sparse interaction."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from c3l.gaussian_mean_field import (
    quadratic_frequency_response,
    quadratic_time_response,
    solve_ghf,
    wick_fock,
)
from c3l.ci.reference import (
    full_hartree_fock_reference,
    half_filled_hartree_fock_reference,
)
from tests.reference.hf_operators import global_particle_hole_operators
from c3l.cc.factorized import factorized_ccsd_problem
from c3l.models import FermionicLindbladModel
from c3l.models import homotopy_model
from tests.reference.physical_fock import (
    annihilation_operators,
    auxiliary_hamiltonian,
    expand_spin_diagonal,
)
from tests.reference.exact_generator import physical_generator_indexed
from tests.reference.generator import superfermion_paper_K
from tests.reference.mapping import fermionic_mapping
from c3l.quadratic_ness import finite_pm_covariance
from c3l.pseudomode import PseudomodeRealization, realization_bcf


def test_sparse_interaction_preserves_operator_order_and_fermionic_signs() -> None:
    """Swapping either antisymmetric index pair must flip the coefficient."""

    from c3l.interactions import (
        SparseTwoBodyInteraction,
    )

    value = 0.7 - 0.2j
    interaction = SparseTwoBodyInteraction(
        n_spin_orbitals=4,
        terms=(
            (1, 0, 2, 3, -value),
            (3, 2, 0, 1, -value.conjugate()),
        ),
    )
    annihilation = annihilation_operators(4)
    from tests.reference.physical_fock import interaction_hamiltonian
    observed = interaction_hamiltonian(interaction, annihilation)
    expected = (
        value
        * annihilation[0].conj().T
        @ annihilation[1].conj().T
        @ annihilation[3]
        @ annihilation[2]
        + value.conjugate()
        * annihilation[2].conj().T
        @ annihilation[3].conj().T
        @ annihilation[1]
        @ annihilation[0]
    )

    np.testing.assert_allclose(observed, expected, atol=2.0e-15, rtol=0.0)
    np.testing.assert_allclose(observed, observed.conj().T, atol=2.0e-15, rtol=0.0)
    assert interaction.coefficient(0, 1, 2, 3) == value
    assert interaction.coefficient(1, 0, 2, 3) == -value
    assert interaction.coefficient(0, 1, 3, 2) == -value


def test_sparse_wick_fock_uses_v_prqs_gamma_sr_index_order() -> None:
    """The general contraction must follow the declared spin-orbital formula."""

    from c3l.interactions import SparseTwoBodyInteraction

    value = 0.4 + 0.3j
    interaction = SparseTwoBodyInteraction(
        4,
        (
            (0, 1, 2, 3, value),
            (2, 3, 0, 1, value.conjugate()),
        ),
    )
    h = np.asarray(
        [
            [-0.2, 0.1j, 0.03, -0.04j],
            [-0.1j, 0.3, 0.02j, 0.05],
            [0.03, -0.02j, -0.1, 0.07j],
            [0.04j, 0.05, -0.07j, 0.15],
        ],
        dtype=np.complex128,
    )
    gamma = np.asarray(
        [
            [0.61, 0.02 + 0.03j, -0.04j, 0.01],
            [0.02 - 0.03j, 0.47, 0.05, 0.03j],
            [0.04j, 0.05, 0.38, -0.02 + 0.01j],
            [0.01, -0.03j, -0.02 - 0.01j, 0.54],
        ],
        dtype=np.complex128,
    )
    tensor = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    tensor[0, 1, 2, 3] = value
    tensor[1, 0, 2, 3] = -value
    tensor[0, 1, 3, 2] = -value
    tensor[1, 0, 3, 2] = value
    tensor[2, 3, 0, 1] = value.conjugate()
    tensor[3, 2, 0, 1] = -value.conjugate()
    tensor[2, 3, 1, 0] = -value.conjugate()
    tensor[3, 2, 1, 0] = value.conjugate()
    expected = np.array(h, copy=True)
    for p in range(4):
        for q in range(4):
            for r in range(4):
                for s in range(4):
                    expected[p, q] += tensor[p, r, q, s] * gamma[s, r]

    np.testing.assert_allclose(
        wick_fock(h, interaction, gamma), expected, atol=2.0e-15, rtol=0.0
    )


def test_density_model_adapts_to_the_same_sparse_interaction() -> None:
    """Legacy onsite and intersite densities must use the canonical object."""

    from tests.reference.physical_fock import interaction_hamiltonian

    zero = np.zeros((2, 2), dtype=np.complex128)
    model = FermionicLindbladModel(
        E=zero,
        gamma_loss=zero,
        gamma_gain=zero,
        U=0.0,
        interaction_sites=(0, 1),
        density_interaction=np.asarray(((1.2, 0.3), (0.3, 0.8))),
    )
    annihilation = annihilation_operators(4)
    density = tuple(
        operator.conj().T @ operator for operator in annihilation
    )
    expected = (
        1.2 * density[0] @ density[1]
        + 0.8 * density[2] @ density[3]
        + 0.3
        * (density[0] + density[1])
        @ (density[2] + density[3])
    )

    np.testing.assert_allclose(
        interaction_hamiltonian(model.interaction, annihilation),
        expected,
        atol=2.0e-15,
        rtol=0.0,
    )


def test_auxiliary_hamiltonian_accepts_general_sparse_interaction() -> None:
    """The model and dense oracle must consume the same general interaction."""

    from c3l.interactions import SparseTwoBodyInteraction

    pair_hopping = 0.35
    interaction = SparseTwoBodyInteraction(
        4,
        (
            (0, 1, 2, 3, pair_hopping),
            (2, 3, 0, 1, pair_hopping),
        ),
    )
    zero = np.zeros((2, 2), dtype=np.complex128)
    model = FermionicLindbladModel(
        E=zero,
        gamma_loss=zero,
        gamma_gain=zero,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )
    annihilation = annihilation_operators(4)
    expected = (
        pair_hopping
        * annihilation[0].conj().T
        @ annihilation[1].conj().T
        @ annihilation[3]
        @ annihilation[2]
        + pair_hopping
        * annihilation[2].conj().T
        @ annihilation[3].conj().T
        @ annihilation[1]
        @ annihilation[0]
    )

    assert model.interaction is interaction
    assert model.has_interactions
    np.testing.assert_allclose(
        auxiliary_hamiltonian(model, annihilation),
        expected,
        atol=2.0e-15,
        rtol=0.0,
    )


def test_general_interaction_matches_independent_superfermion_map() -> None:
    """Physical and tilde interaction signs must reproduce the indexed Liouvillian."""

    from c3l.interactions import SparseTwoBodyInteraction

    exchange = 0.2 + 0.07j
    interaction = SparseTwoBodyInteraction(
        4,
        (
            (0, 3, 1, 2, exchange),
            (1, 2, 0, 3, exchange.conjugate()),
        ),
    )
    E = np.asarray(((0.0, -0.15), (-0.15, 0.0)), dtype=np.complex128)
    loss = np.diag((0.3, 0.25)).astype(np.complex128)
    gain = np.diag((0.1, 0.12)).astype(np.complex128)
    model = FermionicLindbladModel(
        E,
        loss,
        gain,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )
    mapping = fermionic_mapping(4)
    expected = mapping @ (1j * physical_generator_indexed(model)) @ mapping.conj().T

    np.testing.assert_allclose(
        superfermion_paper_K(model), expected, atol=8.0e-14, rtol=8.0e-14
    )


def test_density_adapter_matches_independent_superfermion_map() -> None:
    """Legacy density input must share the canonical physical/tilde signs."""

    E = np.asarray(((0.0, -0.12), (-0.12, 0.0)), dtype=np.complex128)
    loss = np.diag((0.3, 0.25)).astype(np.complex128)
    gain = np.diag((0.1, 0.12)).astype(np.complex128)
    model = FermionicLindbladModel(
        E,
        loss,
        gain,
        U=0.0,
        interaction_sites=(0, 1),
        density_interaction=np.asarray(((0.3, 0.08), (0.08, 0.25))),
    )
    mapping = fermionic_mapping(4)
    expected = mapping @ (1j * physical_generator_indexed(model)) @ mapping.conj().T

    np.testing.assert_allclose(
        superfermion_paper_K(model), expected, atol=8.0e-14, rtol=8.0e-14
    )


def test_general_factorized_products_reconstruct_the_exact_generator() -> None:
    """Production physical/tilde products must retain complex coefficients."""

    from c3l.interactions import SparseTwoBodyInteraction

    value = -0.18 + 0.06j
    interaction = SparseTwoBodyInteraction(
        4,
        (
            (0, 3, 1, 2, value),
            (1, 2, 0, 3, value.conjugate()),
        ),
    )
    E = np.zeros((2, 2), dtype=np.complex128)
    damping = np.diag((0.3, 0.25)).astype(np.complex128)
    model = FermionicLindbladModel(
        E,
        damping,
        damping,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )
    reference = half_filled_hartree_fock_reference(model, 0.5)
    problem = factorized_ccsd_problem(
        model,
        reference=reference,
        build_next_rank_masks=False,
    )
    annihilation, creation = global_particle_hole_operators(reference)
    quadratic = FermionicLindbladModel(
        E,
        damping,
        damping,
        U=0.0,
        interaction_sites=(0, 1),
    )
    observed = np.zeros_like(superfermion_paper_K(model))

    def linear(operator):
        return sum(
            coefficient
            * (creation[mode] if kind == "create" else annihilation[mode])
            for kind, mode, coefficient in operator.components
        )

    for product in problem.products:
        term = np.eye(observed.shape[0], dtype=np.complex128)
        for factor in product.factors:
            term = term @ linear(factor)
        observed += product.coefficient * term

    expected = superfermion_paper_K(model) - superfermion_paper_K(quadratic)
    np.testing.assert_allclose(observed, expected, atol=8.0e-14, rtol=8.0e-14)


def test_density_factorized_products_reconstruct_the_same_exact_generator() -> None:
    """The legacy density adapter and general sparse path have one algebra."""

    from c3l.models import QuadraticOpenSystem

    E = np.zeros((2, 2), dtype=np.complex128)
    damping = np.diag((0.3, 0.25)).astype(np.complex128)
    model = FermionicLindbladModel(
        E,
        damping,
        damping,
        U=0.0,
        interaction_sites=(0, 1),
        density_interaction=np.asarray(((0.4, 0.1), (0.1, 0.3))),
    )
    reference = half_filled_hartree_fock_reference(model, 0.5)
    problem = factorized_ccsd_problem(
        model,
        reference=reference,
        build_next_rank_masks=False,
    )
    annihilation, creation = global_particle_hole_operators(reference)
    observed = np.zeros_like(superfermion_paper_K(model))

    def linear(operator):
        return sum(
            coefficient
            * (creation[mode] if kind == "create" else annihilation[mode])
            for kind, mode, coefficient in operator.components
        )

    for product in problem.products:
        term = np.eye(observed.shape[0], dtype=np.complex128)
        for factor in product.factors:
            term = term @ linear(factor)
        observed += product.coefficient * term

    full_fock = np.zeros((4, 4), dtype=np.complex128)
    full_fock[0::2, 0::2] = reference.E_up
    full_fock[1::2, 1::2] = reference.E_down
    full_damping = np.zeros((4, 4), dtype=np.complex128)
    full_damping[0::2, 0::2] = full_damping[1::2, 1::2] = damping
    quadratic = QuadraticOpenSystem(full_fock, full_damping, full_damping)
    expected = superfermion_paper_K(model) - superfermion_paper_K(quadratic)
    np.testing.assert_allclose(observed, expected, atol=8.0e-14, rtol=8.0e-14)


def test_general_interaction_uses_the_existing_model_serialization() -> None:
    """A model round trip must retain every sparse coefficient and index."""

    from c3l.interactions import SparseTwoBodyInteraction

    value = 0.12 - 0.04j
    interaction = SparseTwoBodyInteraction(
        4,
        (
            (0, 1, 2, 3, value),
            (2, 3, 0, 1, value.conjugate()),
        ),
    )
    zero = np.zeros((2, 2), dtype=np.complex128)
    model = FermionicLindbladModel(
        zero,
        zero,
        zero,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )

    restored = FermionicLindbladModel.from_dict(model.to_dict())
    assert restored.interaction.terms == interaction.terms


def test_general_full_hf_uses_the_actual_complex_covariance() -> None:
    """The fixed point must reconstruct F[gamma] without diagonal projection."""

    from c3l.interactions import density_density_interaction

    spatial = np.asarray(((-0.2, -0.11j), (0.11j, 0.25)))
    h = np.zeros((4, 4), dtype=np.complex128)
    h[0::2, 0::2] = spatial
    h[1::2, 1::2] = spatial
    loss = np.diag((0.32, 0.32, 0.27, 0.27)).astype(np.complex128)
    gain = np.diag((0.13, 0.13, 0.09, 0.09)).astype(np.complex128)
    interaction = density_density_interaction(
        2,
        (0, 1),
        np.asarray(((0.35, 0.10), (0.10, 0.25))),
    )

    result = solve_ghf(
        h,
        interaction,
        loss,
        gain,
        tolerance=2.0e-11,
        max_iterations=100,
        damping=0.7,
    )
    target, lyapunov = finite_pm_covariance(result.fock, loss, gain)

    assert result.converged
    assert result.self_consistency_residual < 2.0e-11
    assert result.lyapunov_residual < 2.0e-11
    assert lyapunov < 2.0e-12
    np.testing.assert_allclose(result.fock, wick_fock(h, interaction, result.gamma))
    np.testing.assert_allclose(result.gamma, target, atol=2.0e-11, rtol=2.0e-11)
    assert np.linalg.norm(result.gamma - np.diag(np.diag(result.gamma))) > 1.0e-3
    assert result.physicality_defect < 2.0e-12


def test_full_hf_reference_keeps_the_solved_fock_and_covariance() -> None:
    """CC normal modes must be built from the actual self-consistent solution."""

    from c3l.ci.reference import full_hartree_fock_reference
    from c3l.interactions import density_density_interaction

    spatial = np.asarray(((-0.3, -0.08), (-0.08, 0.2)), dtype=np.complex128)
    h = np.zeros((4, 4), dtype=np.complex128)
    h[0::2, 0::2] = spatial
    h[1::2, 1::2] = spatial
    loss_spatial = np.diag((0.31, 0.26)).astype(np.complex128)
    gain_spatial = np.diag((0.12, 0.10)).astype(np.complex128)
    loss = np.zeros((4, 4), dtype=np.complex128)
    gain = np.zeros((4, 4), dtype=np.complex128)
    loss[0::2, 0::2] = loss[1::2, 1::2] = loss_spatial
    gain[0::2, 0::2] = gain[1::2, 1::2] = gain_spatial
    interaction = density_density_interaction(
        2,
        (0, 1),
        np.asarray(((0.28, 0.07), (0.07, 0.22))),
    )
    model = FermionicLindbladModel(
        spatial,
        loss_spatial,
        gain_spatial,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )
    hf = solve_ghf(
        h,
        interaction,
        loss,
        gain,
        tolerance=2.0e-11,
        max_iterations=100,
        damping=0.7,
    )
    reference = full_hartree_fock_reference(model, hf.fock, hf.gamma)

    np.testing.assert_allclose(reference.gamma, hf.gamma)
    np.testing.assert_allclose(reference.fock, hf.fock)
    np.testing.assert_allclose(reference.E_up, hf.fock[0::2, 0::2])
    np.testing.assert_allclose(reference.E_down, hf.fock[1::2, 1::2])
    assert reference.m_up != 0.5
    assert reference.m_down != 0.5


def test_full_hf_reference_rejects_a_quadratic_but_non_hf_fixed_point() -> None:
    """A covariance of arbitrary F is not enough: F must equal h plus vbar gamma."""

    from c3l.interactions import density_density_interaction

    zero = np.zeros((2, 2), dtype=np.complex128)
    damping = np.diag((0.3, 0.25)).astype(np.complex128)
    interaction = density_density_interaction(
        2,
        (0, 1),
        np.asarray(((0.2, 0.0), (0.0, 0.2))),
    )
    model = FermionicLindbladModel(
        zero,
        damping,
        damping,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )
    wrong_fock = np.zeros((4, 4), dtype=np.complex128)
    gamma = 0.5 * np.eye(4, dtype=np.complex128)

    with pytest.raises(ValueError, match="Fock reconstruction"):
        full_hartree_fock_reference(model, wrong_fock, gamma)


def test_full_hf_reference_counterterm_recovers_the_physical_generator() -> None:
    """F plus the existing products must equal the requested E plus H2."""

    from c3l.interactions import density_density_interaction
    from c3l.models import QuadraticOpenSystem

    interaction = density_density_interaction(
        2,
        (0, 1),
        np.asarray(((0.25, 0.08), (0.08, 0.20))),
    )
    E = np.asarray(((0.18, -0.07j), (0.07j, -0.12)), dtype=np.complex128)
    damping = np.diag((0.3, 0.24)).astype(np.complex128)
    model = FermionicLindbladModel(
        E,
        damping,
        damping,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )
    h = np.zeros((4, 4), dtype=np.complex128)
    h[0::2, 0::2] = h[1::2, 1::2] = E
    full_damping = np.zeros((4, 4), dtype=np.complex128)
    full_damping[0::2, 0::2] = full_damping[1::2, 1::2] = damping
    hf = solve_ghf(
        h,
        interaction,
        full_damping,
        full_damping,
        tolerance=2.0e-11,
        max_iterations=100,
        damping=0.7,
    )
    reference = full_hartree_fock_reference(model, hf.fock, hf.gamma)
    problem = factorized_ccsd_problem(
        model,
        reference=reference,
        build_next_rank_masks=False,
    )
    annihilation, creation = global_particle_hole_operators(reference)
    observed = np.zeros_like(superfermion_paper_K(model))

    def linear(operator):
        return sum(
            coefficient
            * (creation[mode] if kind == "create" else annihilation[mode])
            for kind, mode, coefficient in operator.components
        )

    for product in problem.products:
        term = np.eye(observed.shape[0], dtype=np.complex128)
        for factor in product.factors:
            term = term @ linear(factor)
        observed += product.coefficient * term

    quadratic = QuadraticOpenSystem(hf.fock, full_damping, full_damping)
    expected = superfermion_paper_K(model) - superfermion_paper_K(quadratic)
    np.testing.assert_allclose(observed, expected, atol=8.0e-14, rtol=8.0e-14)


def test_reference_only_dissipation_is_exactly_removed_from_the_generator() -> None:
    """A dark-mode Gaussian regulator must not change the physical Liouvillian."""

    from c3l.interactions import density_density_interaction
    from c3l.models import QuadraticOpenSystem

    bright = 0.5 * np.asarray(((1.0, 1.0), (1.0, 1.0)))
    dark = 0.5 * np.asarray(((1.0, -1.0), (-1.0, 1.0)))
    physical_loss = 0.24 * bright
    physical_gain = 0.16 * bright
    reference_loss = physical_loss + 0.07 * dark
    reference_gain = physical_gain + 0.07 * dark
    E = np.asarray(((0.12, -0.03), (-0.03, -0.08)), dtype=np.complex128)
    interaction = density_density_interaction(
        2,
        (0, 1),
        np.asarray(((0.25, 0.04), (0.04, 0.20))),
    )
    model = FermionicLindbladModel(
        E,
        physical_loss,
        physical_gain,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )
    h = np.zeros((4, 4), dtype=np.complex128)
    h[0::2, 0::2] = h[1::2, 1::2] = E
    loss = np.zeros_like(h)
    gain = np.zeros_like(h)
    loss[0::2, 0::2] = loss[1::2, 1::2] = reference_loss
    gain[0::2, 0::2] = gain[1::2, 1::2] = reference_gain
    hf = solve_ghf(
        h,
        interaction,
        loss,
        gain,
        tolerance=2.0e-11,
        max_iterations=100,
        damping=0.7,
    )
    reference = full_hartree_fock_reference(
        model,
        hf.fock,
        hf.gamma,
        reference_gamma_loss=reference_loss,
        reference_gamma_gain=reference_gain,
    )
    problem = factorized_ccsd_problem(
        model,
        reference=reference,
        build_next_rank_masks=False,
    )
    annihilation, creation = global_particle_hole_operators(reference)
    observed = np.zeros_like(superfermion_paper_K(model))

    def linear(operator):
        return sum(
            coefficient
            * (creation[mode] if kind == "create" else annihilation[mode])
            for kind, mode, coefficient in operator.components
        )

    for product in problem.products:
        term = np.eye(observed.shape[0], dtype=np.complex128)
        for factor in product.factors:
            term = term @ linear(factor)
        observed += product.coefficient * term

    quadratic = QuadraticOpenSystem(hf.fock, loss, gain)
    expected = superfermion_paper_K(model) - superfermion_paper_K(quadratic)
    from tests.reference.generator import parity_projectors

    even, _ = parity_projectors(model.n_spin_orbitals)
    np.testing.assert_allclose(
        even @ observed @ even,
        even @ expected @ even,
        atol=1.2e-13,
        rtol=1.2e-13,
    )


def test_reference_only_dissipation_preserves_the_u0_retarded_response() -> None:
    """The odd retained EOM sector must recover the unregulated resolvent."""

    from c3l.cc.component_response import (
        ComponentSigmaOperator,
        component_retarded_matrix_green,
    )
    from c3l.cc.component_tensor import (
        ComponentEOMProblem,
        ComponentStaticProblem,
        component_eom_context,
        component_eom_sources,
        component_static_residual,
    )
    from c3l.cc.eom import eom_ccsd_problem
    from c3l.interactions import SparseTwoBodyInteraction

    bright = 0.5 * np.asarray(((1.0, 1.0), (1.0, 1.0)))
    dark = 0.5 * np.asarray(((1.0, -1.0), (-1.0, 1.0)))
    physical_loss = 0.24 * bright
    physical_gain = 0.16 * bright
    reference_loss = physical_loss + 0.07 * dark
    reference_gain = physical_gain + 0.07 * dark
    E = np.asarray(((0.12, -0.03), (-0.03, 0.12)), dtype=np.complex128)
    model = FermionicLindbladModel(
        E,
        physical_loss,
        physical_gain,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=SparseTwoBodyInteraction(4, ()),
    )
    fock = expand_spin_diagonal(E)
    loss = expand_spin_diagonal(reference_loss)
    gain = expand_spin_diagonal(reference_gain)
    gamma, _ = finite_pm_covariance(fock, loss, gain)
    reference = full_hartree_fock_reference(
        model,
        fock,
        gamma,
        reference_gamma_loss=reference_loss,
        reference_gamma_gain=reference_gain,
    )
    cc = factorized_ccsd_problem(
        model,
        reference=reference,
        build_next_rank_masks=False,
    )
    static = ComponentStaticProblem.compile(cc)
    amplitudes = np.zeros(cc.num_amplitudes, dtype=np.complex128)
    residual = component_static_residual(static, amplitudes)
    assert np.linalg.norm(residual.retained) < 2.0e-13
    eom = eom_ccsd_problem(cc, response_spin=0, build_next_rank_masks=False)
    response_problem = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(response_problem, amplitudes)
    sources = tuple(component_eom_sources(context, site=site) for site in (0, 1))
    omega = np.asarray((-0.5, 0.0, 0.4))
    eta = 0.08
    observed = component_retarded_matrix_green(
        ComponentSigmaOperator(context),
        sources,
        omega,
        eta,
        rtol=1.0e-12,
        restart=20,
        maxiter=100,
    )
    expected = quadratic_frequency_response(
        E,
        physical_loss,
        physical_gain,
        gamma[0::2, 0::2],
        omega + 1j * eta,
    )
    np.testing.assert_allclose(
        observed.retarded,
        expected.retarded,
        atol=3.0e-11,
        rtol=3.0e-11,
    )


def test_general_factorized_problem_solves_full_hf_by_default() -> None:
    """A general model must not enter the legacy half-filled reference path."""

    from c3l.interactions import density_density_interaction

    E = np.asarray(((-0.22, -0.06), (-0.06, 0.17)), dtype=np.complex128)
    loss = np.diag((0.30, 0.26)).astype(np.complex128)
    gain = np.diag((0.12, 0.09)).astype(np.complex128)
    interaction = density_density_interaction(
        2,
        (0, 1),
        np.asarray(((0.24, 0.06), (0.06, 0.20))),
    )
    model = FermionicLindbladModel(
        E,
        loss,
        gain,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )

    problem = factorized_ccsd_problem(model, build_next_rank_masks=False)
    reference = problem.reference
    h = np.zeros((4, 4), dtype=np.complex128)
    h[0::2, 0::2] = h[1::2, 1::2] = E

    assert reference.gamma is not None
    assert reference.fock is not None
    assert problem.construction_record()["reference_quadratic_separated"]
    np.testing.assert_allclose(
        reference.fock,
        wick_fock(h, model.interaction, reference.gamma),
        atol=2.0e-11,
        rtol=2.0e-11,
    )


def test_general_factorized_problem_rejects_legacy_reference_E_shortcut() -> None:
    """General interactions require the actual covariance, not a design Fock matrix."""

    from c3l.interactions import density_density_interaction

    E = np.asarray(((-0.2, 0.0), (0.0, 0.1)), dtype=np.complex128)
    damping = np.diag((0.3, 0.25)).astype(np.complex128)
    model = FermionicLindbladModel(
        E,
        damping,
        damping,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=density_density_interaction(
            2,
            (0, 1),
            np.asarray(((0.2, 0.05), (0.05, 0.18))),
        ),
    )

    with pytest.raises(ValueError, match="actual full-HF reference"):
        factorized_ccsd_problem(
            model,
            reference_E=np.zeros_like(E),
            build_next_rank_masks=False,
        )


def test_legacy_density_uses_full_hf_by_default_and_explicit_compatibility() -> None:
    """Legacy fields are input adapters, not an implicit reference selection."""

    E = np.asarray(((-0.2, -0.04), (-0.04, 0.12)), dtype=np.complex128)
    damping = np.diag((0.3, 0.25)).astype(np.complex128)
    model = FermionicLindbladModel(
        E,
        damping,
        damping,
        U=0.0,
        interaction_sites=(0, 1),
        density_interaction=np.asarray(((0.22, 0.05), (0.05, 0.18))),
    )

    default = factorized_ccsd_problem(model, build_next_rank_masks=False)
    compatibility = factorized_ccsd_problem(
        model,
        build_next_rank_masks=False,
        reference_mode="legacy_half_filled",
    )

    assert default.reference.gamma is not None
    assert default.reference.fock is not None
    assert compatibility.reference.gamma is None
    assert compatibility.reference.fock is None


def test_component_path_rejects_interactions_that_change_spin_projection() -> None:
    """The established four coordinate blocks conserve up and down separately."""

    from c3l.interactions import SparseTwoBodyInteraction

    interaction = SparseTwoBodyInteraction(
        4,
        (
            (0, 2, 0, 3, 0.1),
            (0, 3, 0, 2, 0.1),
        ),
    )
    zero = np.zeros((2, 2), dtype=np.complex128)
    damping = np.diag((0.3, 0.25)).astype(np.complex128)
    model = FermionicLindbladModel(
        zero,
        damping,
        damping,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )
    explicit_reference = half_filled_hartree_fock_reference(model, 0.5)

    with pytest.raises(ValueError, match="separate up/down conservation"):
        factorized_ccsd_problem(model, build_next_rank_masks=False)
    with pytest.raises(ValueError, match="separate up/down conservation"):
        factorized_ccsd_problem(
            model,
            reference=explicit_reference,
            build_next_rank_masks=False,
        )


def test_general_interaction_uses_the_existing_physical_homotopy() -> None:
    """At s=0 only Kref remains and at s=1 the exact model is recovered."""

    from c3l.interactions import SparseTwoBodyInteraction

    value = 0.21
    interaction = SparseTwoBodyInteraction(
        4,
        ((0, 1, 2, 3, value), (2, 3, 0, 1, value)),
    )
    E = np.asarray(((0.2, -0.1), (-0.1, -0.3)), dtype=np.complex128)
    K_ref = np.asarray(((0.05, -0.03), (-0.03, 0.04)), dtype=np.complex128)
    damping = np.diag((0.25, 0.3)).astype(np.complex128)
    physical = FermionicLindbladModel(
        E,
        damping,
        damping,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )

    start = homotopy_model(physical, K_ref, 0.0)
    middle = homotopy_model(physical, K_ref, 0.4)
    endpoint = homotopy_model(physical, K_ref, 1.0)

    np.testing.assert_allclose(start.E, K_ref)
    assert not start.has_interactions
    np.testing.assert_allclose(middle.E, K_ref + 0.4 * (E - K_ref))
    assert middle.interaction.coefficient(0, 1, 2, 3) == 0.4 * value
    np.testing.assert_allclose(endpoint.E, physical.E)
    assert endpoint.interaction.terms == physical.interaction.terms


def test_matrix_bath_factorization_reconstructs_M_times_scalar_shape() -> None:
    """Eigenchannels must reproduce the requested matrix without rotating observables."""

    from c3l.interactions import SparseTwoBodyInteraction
    from c3l.pseudomode import matrix_pseudomode_model

    lesser = PseudomodeRealization(
        H=np.asarray(((-0.6, 0.12), (0.12, -1.1)), dtype=np.complex128),
        damping=np.diag((0.25, 0.4)).astype(np.complex128),
        g=np.asarray((0.7, -0.2j), dtype=np.complex128),
    )
    greater = PseudomodeRealization(
        H=-lesser.H.conj(),
        damping=lesser.damping.conj(),
        g=lesser.g.conj(),
    )
    M = np.asarray(((1.0, 0.35j), (-0.35j, 0.7)), dtype=np.complex128)
    interaction = SparseTwoBodyInteraction(
        4,
        ((0, 1, 0, 1, 0.2), (2, 3, 2, 3, 0.15)),
    )
    model = matrix_pseudomode_model(
        np.asarray(((-0.1, 0.04), (0.04, 0.2)), dtype=np.complex128),
        M,
        lesser,
        greater,
        interaction,
    )
    impurity = np.arange(2)
    bath = np.arange(2, model.n_sites)
    coupling = model.E[np.ix_(impurity, bath)]
    bath_E = model.E[np.ix_(bath, bath)]
    damping = (model.gamma_loss + model.gamma_gain)[np.ix_(bath, bath)]
    omega = np.asarray((-0.8, 0.0, 0.9))
    observed = np.empty((omega.size, 2, 2), dtype=np.complex128)
    for index, value in enumerate(omega):
        bath_green = np.linalg.inv(
            value * np.eye(bath.size) - bath_E + 1j * damping
        )
        observed[index] = coupling @ bath_green @ coupling.conj().T

    def scalar(realization, value):
        return realization.g.conj() @ np.linalg.solve(
            value * np.eye(realization.size)
            - realization.H
            + 1j * realization.damping,
            realization.g,
        )

    expected = np.asarray(
        [M * (scalar(lesser, value) + scalar(greater, value)) for value in omega]
    )
    np.testing.assert_allclose(observed, expected, atol=3.0e-14, rtol=3.0e-14)
    for value in observed:
        broadening = 0.5j * (value - value.conj().T)
        assert np.linalg.eigvalsh(broadening)[0] > -2.0e-14

    times = np.asarray((0.0, 0.3, 0.8))
    lesser_matrix = np.zeros((times.size, 2, 2), dtype=np.complex128)
    greater_matrix = np.zeros_like(lesser_matrix)
    channel_size = lesser.size + greater.size
    for channel in range(np.linalg.matrix_rank(M)):
        start = 2 + channel * channel_size
        lesser_block = np.arange(start, start + lesser.size)
        greater_block = np.arange(
            start + lesser.size,
            start + lesser.size + greater.size,
        )
        for output, block, sign in (
            (lesser_matrix, lesser_block, 1.0),
            (greater_matrix, greater_block, -1.0),
        ):
            coupling_block = model.E[np.ix_(impurity, block)]
            generator = (
                sign * 1j * model.E[np.ix_(block, block)]
                - damping[np.ix_(block - 2, block - 2)]
            )
            for index, time in enumerate(times):
                output[index] += (
                    coupling_block
                    @ scipy.linalg.expm(generator * time)
                    @ coupling_block.conj().T
                )
    np.testing.assert_allclose(
        lesser_matrix,
        np.asarray([M * value for value in realization_bcf(times, lesser, "lesser")]),
        atol=3.0e-14,
        rtol=3.0e-14,
    )
    np.testing.assert_allclose(
        greater_matrix,
        np.asarray([M * value for value in realization_bcf(times, greater, "greater")]),
        atol=3.0e-14,
        rtol=3.0e-14,
    )
    assert model.interaction.coefficient(0, 1, 0, 1) == 0.2
    assert model.interaction.coefficient(2, 3, 2, 3) == 0.15


def test_general_interaction_and_reference_are_public_cclens_symbols() -> None:
    """Benchmark builders should not import private implementation modules."""

    from c3l import SparseTwoBodyInteraction
    from c3l.ci import full_hartree_fock_reference

    assert SparseTwoBodyInteraction.__name__ == "SparseTwoBodyInteraction"
    assert full_hartree_fock_reference.__name__ == "full_hartree_fock_reference"


def test_matrix_bath_u0_covariance_and_response_are_exact() -> None:
    """The assembled quadratic model must match its direct Lyapunov/resolvent algebra."""

    from c3l.interactions import SparseTwoBodyInteraction
    from c3l.pseudomode import matrix_pseudomode_model

    lesser = PseudomodeRealization(
        H=np.asarray(((-0.5,),), dtype=np.complex128),
        damping=np.asarray(((0.3,),), dtype=np.complex128),
        g=np.asarray((0.7,), dtype=np.complex128),
    )
    greater = PseudomodeRealization(
        H=np.asarray(((0.5,),), dtype=np.complex128),
        damping=np.asarray(((0.3,),), dtype=np.complex128),
        g=np.asarray((0.7,), dtype=np.complex128),
    )
    model = matrix_pseudomode_model(
        np.asarray(((-0.15, 0.06), (0.06, 0.2)), dtype=np.complex128),
        np.ones((2, 2), dtype=np.complex128),
        lesser,
        greater,
        SparseTwoBodyInteraction(4, ()),
    )
    h = expand_spin_diagonal(model.E)
    loss = expand_spin_diagonal(model.gamma_loss)
    gain = expand_spin_diagonal(model.gamma_gain)
    gamma, residual = finite_pm_covariance(h, loss, gain)
    shifts = np.asarray((-0.4 + 0.08j, 0.3 + 0.08j))
    selected = np.asarray((0, 2), dtype=np.int64)
    response = quadratic_frequency_response(
        h,
        loss,
        gain,
        gamma,
        shifts,
        orbitals=selected,
    )
    direct = np.linalg.inv(
        shifts[:, None, None] * np.eye(h.shape[0])[None, :, :]
        - h[None, :, :]
        + 1j * (loss + gain)[None, :, :]
    )[:, selected][:, :, selected]
    time = quadratic_time_response(
        h,
        loss,
        gain,
        gamma,
        np.asarray((0.0, 0.2)),
        orbitals=selected,
    )

    assert residual < 2.0e-13
    assert np.linalg.eigvalsh(gamma)[0] > -2.0e-13
    assert np.linalg.eigvalsh(gamma)[-1] < 1.0 + 2.0e-13
    np.testing.assert_allclose(response.retarded, direct, atol=3.0e-14, rtol=3.0e-14)
    np.testing.assert_allclose(time.retarded[0], -1j * np.eye(2), atol=3.0e-15)

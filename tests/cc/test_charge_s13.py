"""Charge-only CCSD/S13 against an independent physical Lindblad oracle."""

import numpy as np
import pytest
import scipy.linalg

from c3l import FermionicLindbladModel, SparseTwoBodyInteraction
from c3l.cc.component_tensor import (
    ComponentEOMProblem, ComponentStaticProblem, component_eom_context,
    component_eom_sources, component_static_jvp, component_static_residual,
)
from c3l.cc.component_solver import solve_component_ccsd
from c3l.cc.component_response import ComponentSigmaOperator, component_retarded_matrix_green
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import factorized_cc_problem
from c3l.cc.newton_krylov import NewtonKrylovPolicy


def charge_model(interaction=0.4):
    return FermionicLindbladModel(
        np.array([[0.13, 0.09 + 0.07j], [0.09 - 0.07j, -0.21]]),
        np.array([[0.31, 0.025j], [-0.025j, 0.24]]),
        np.array([[0.17, 0.02], [0.02, 0.11]]),
        U=0.0, symmetry="charge", interaction_sites=(0, 1),
        two_body_interaction=SparseTwoBodyInteraction(2, ((0, 1, 0, 1, interaction),)),
    )


def solved_charge(interaction=0.4):
    model = charge_model(interaction)
    oracle = factorized_cc_problem(model, "CCSD", build_next_rank_masks=False)
    static = ComponentStaticProblem.compile(oracle)
    solution = solve_component_ccsd(
        static, policy=NewtonKrylovPolicy(tolerance=1e-12),
    )
    assert solution.converged
    eom = eom_ccsd_problem(oracle, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(problem, solution.amplitudes)
    sources = tuple(component_eom_sources(context, site=orbital) for orbital in range(2))
    return model, static, solution, context, sources


def exact_lindblad(model):
    from tests.reference.exact_generator import auxiliary_exact_inputs, physical_generator_indexed

    hamiltonian, annihilation, loss, gain = auxiliary_exact_inputs(model)
    generator = physical_generator_indexed(model)
    dimension = hamiltonian.shape[0]
    trace = np.eye(dimension).reshape(-1, order="F")
    system = generator.copy()
    right = np.zeros(dimension**2, dtype=complex)
    system[0] = trace
    right[0] = 1
    density = np.linalg.solve(system, right).reshape((dimension, dimension), order="F")
    response = generator.copy()
    for first in range(len(annihilation)):
        for second in range(len(annihilation)):
            response -= 4 * loss[first, second] * np.kron(
                annihilation[first].conj(), annihilation[second],
            )
            response -= 4 * gain[first, second] * np.kron(
                annihilation[second].T, annihilation[first].conj().T,
            )
    return density, generator, response, annihilation


def test_charge_ccsd_two_modes_is_exact():
    from tests.reference.hf_operators import global_gaussian_vacuum, global_particle_hole_operators
    from tests.reference.mapping import fermionic_mapping

    model, static, solution, _, _ = solved_charge()
    oracle = static.oracle
    assert oracle.cluster_basis.num_modes == 4
    assert oracle.num_amplitudes == 5
    vacuum = global_gaussian_vacuum(oracle.reference)
    _, creators = global_particle_hole_operators(oracle.reference)
    cluster = np.zeros((16, 16), dtype=complex)
    for amplitude, mask in zip(solution.amplitudes, oracle.cluster_basis.masks):
        product = np.eye(16, dtype=complex)
        for mode, creator in enumerate(creators):
            if mask & (1 << mode):
                product = product @ creator
        cluster += amplitude * product
    observed = np.linalg.solve(fermionic_mapping(2), scipy.linalg.expm(cluster) @ vacuum.state)
    expected, generator, _, annihilation = exact_lindblad(model)
    np.testing.assert_allclose(observed.reshape((4, 4), order="F"), expected, atol=1e-9)
    assert np.linalg.norm(generator @ observed) < 1e-9
    occupation = annihilation[0].conj().T @ annihilation[0] @ annihilation[1].conj().T @ annihilation[1]
    assert abs(np.trace(occupation @ expected) - np.trace(occupation @ observed.reshape((4, 4), order="F"))) < 1e-9


@pytest.mark.parametrize("interaction", [0.0, 0.4])
def test_charge_s13_full_matrix_matches_lindblad_ed(interaction):
    model, _, _, context, sources = solved_charge(interaction)
    assert context.problem.coordinates.dimension == 4
    frequencies = np.array([-0.6, 0.0, 0.7])
    broadening = 0.13
    result = component_retarded_matrix_green(
        ComponentSigmaOperator(context), sources, frequencies, broadening,
        rtol=1e-12, restart=10, maxiter=30,
    )
    expected = exact_retarded_green(model, frequencies, broadening)
    assert np.max(np.abs(expected[:, 0, 1])) > 1e-3
    assert np.max(np.abs(expected[:, 1, 0])) > 1e-3
    np.testing.assert_allclose(result.retarded, expected, atol=1e-9, rtol=1e-9)


def exact_retarded_green(model, frequencies, broadening):
    """Resolve both off-diagonal channels directly in physical Liouville space."""

    frequencies = np.asarray(frequencies)
    density, _, generator, annihilation = exact_lindblad(model)
    expected = np.empty((frequencies.size, 2, 2), dtype=complex)
    for position, frequency in enumerate(frequencies):
        resolvent = (frequency + 1j * broadening) * np.eye(16) - 1j * generator
        for source, creator in enumerate(annihilation):
            right = (creator.conj().T @ density + density @ creator.conj().T).reshape(-1, order="F")
            response = np.linalg.solve(resolvent, right).reshape((4, 4), order="F")
            for sink, operator in enumerate(annihilation):
                expected[position, sink, source] = np.trace(operator @ response)
    return expected


def test_charge_jvp_matches_finite_difference():
    oracle = factorized_cc_problem(charge_model(), "CCSD", build_next_rank_masks=False)
    problem = ComponentStaticProblem.compile(oracle)
    amplitudes = oracle.tilde.unpack(np.linspace(-0.01, 0.01, oracle.num_variables))
    direction = oracle.tilde.unpack(np.linspace(0.03, -0.02, oracle.num_variables))
    step = 1e-5
    difference = (
        component_static_residual(problem, amplitudes + step * direction).retained
        - component_static_residual(problem, amplitudes - step * direction).retained
    ) / (2 * step)
    np.testing.assert_allclose(component_static_jvp(problem, amplitudes, direction).retained, difference, atol=1e-10)


def test_charge_mode_preserves_spin_conserving_ccsd_and_s13_results():
    from c3l.cc.observables import factorized_ccsd_covariance

    spin_model = FermionicLindbladModel(
        [[-0.2, 0.17], [0.17, 0.11]], [[0.21, 0], [0, 0.32]],
        [[0.13, 0], [0, 0.18]], U=0.4,
    )
    full_model = FermionicLindbladModel(
        spin_model.physical_E, spin_model.physical_gamma_loss, spin_model.physical_gamma_gain,
        U=0.0, symmetry="charge", interaction_sites=(0, 1),
        two_body_interaction=spin_model.interaction,
    )
    covariances = []
    greens = []
    for model in (spin_model, full_model):
        oracle = factorized_cc_problem(model, "CCSD", build_next_rank_masks=False)
        static = ComponentStaticProblem.compile(oracle)
        solution = solve_component_ccsd(static, policy=NewtonKrylovPolicy(tolerance=1e-12))
        assert solution.converged
        covariances.append(factorized_ccsd_covariance(oracle.reference, oracle.cluster_basis, solution.amplitudes))
        eom = eom_ccsd_problem(oracle, build_next_rank_masks=False)
        context = component_eom_context(ComponentEOMProblem.compile(eom, static, charge=1), solution.amplitudes)
        sites = (0, 1) if model.symmetry == "spin" else (0, 2)
        sources = [component_eom_sources(context, site=site) for site in sites]
        result = component_retarded_matrix_green(
            ComponentSigmaOperator(context), sources, [-0.6, 0.2, 0.7], 0.13,
            rtol=1e-12, restart=30, maxiter=50,
        )
        greens.append(result.retarded)
    np.testing.assert_allclose(covariances[0], covariances[1], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(greens[0], greens[1], atol=1e-9, rtol=1e-9)


def test_charge_jax_actions_and_runtime_reuse():
    pytest.importorskip("jax")
    from c3l.cc.component_jax import JaxStaticCCSDOperator, JaxRetainedEOMOperator

    engines = []
    cache_sizes = []
    for strength in (0.0, 0.4):
        _, static, solution, context, _ = solved_charge(strength)
        operator = JaxStaticCCSDOperator.from_cpu(static, platform="cpu", production=False)
        direction = static.oracle.tilde.unpack(np.full(static.oracle.num_variables, 0.02))
        np.testing.assert_allclose(operator.residual(solution.amplitudes), component_static_residual(static, solution.amplitudes).retained, atol=1e-10)
        np.testing.assert_allclose(operator.jvp(solution.amplitudes, direction), component_static_jvp(static, solution.amplitudes, direction).retained, atol=1e-10)
        engines.append(operator.engine)
        cache_sizes.append((operator.engine.residual._cache_size(), operator.engine.jvp._cache_size()))
        response = JaxRetainedEOMOperator.from_cpu(context, platform="cpu", production=False)
        vector = np.arange(response.dimension, dtype=complex) + 0.2j
        np.testing.assert_allclose(response.matvec(vector), ComponentSigmaOperator(context).matvec(vector), atol=1e-10)
    assert engines[0] is engines[1]
    assert cache_sizes[0] == cache_sizes[1]


def test_general_spin_changing_interaction_matches_dense_similarity_transform():
    from tests.reference.hf_operators import creation_matrix
    from tests.reference.hf_operators import global_gaussian_vacuum, global_particle_hole_operators
    from tests.reference.generator import printed_quadratic_paper_K

    coupling = 0.06 + 0.03j
    model = FermionicLindbladModel(
        np.diag([-0.2, 0.1, 0.15, -0.05]), np.eye(4) * 0.31,
        np.diag([0.12, 0.18, 0.15, 0.11]), U=0.0, symmetry="charge",
        interaction_sites=(0, 1, 2, 3),
        two_body_interaction=SparseTwoBodyInteraction(4, (
            (0, 2, 0, 1, coupling), (0, 1, 0, 2, coupling.conjugate()),
            (0, 3, 0, 3, 0.13),
        )),
    )
    oracle = factorized_cc_problem(model, "CCSD", build_next_rank_masks=False)
    static = ComponentStaticProblem.compile(oracle)
    amplitudes = oracle.tilde.unpack(np.linspace(-0.001, 0.002, oracle.num_variables))
    vacuum = global_gaussian_vacuum(oracle.reference)
    _, creators = global_particle_hole_operators(oracle.reference)
    dimension = vacuum.state.size
    determinants = np.empty((dimension, dimension), dtype=complex)
    determinants[:, 0] = vacuum.state
    for mask in range(1, dimension):
        lowest = mask & -mask
        determinants[:, mask] = creators[lowest.bit_length() - 1] @ determinants[:, mask ^ lowest]
    generator = np.linalg.solve(determinants, printed_quadratic_paper_K(model) @ determinants)
    cluster = sum(amplitude * creation_matrix(mask, oracle.cluster_basis.num_modes) for amplitude, mask in zip(amplitudes, oracle.cluster_basis.masks))
    transformed = scipy.linalg.expm(-cluster) @ generator @ scipy.linalg.expm(cluster)
    expected = -1j * transformed[np.array(oracle.cluster_basis.masks), 0]
    np.testing.assert_allclose(component_static_residual(static, amplitudes).retained, expected, atol=1e-10)
    eom = eom_ccsd_problem(oracle, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(problem, amplitudes)
    masks = [eom.basis.masks[position] for position in problem.positions]
    expected_action = np.empty((len(masks), len(masks)), complex)
    for column, mask in enumerate(masks):
        creator = creation_matrix(mask, oracle.cluster_basis.num_modes)
        expected_action[:, column] = (transformed[:, mask] - creator @ transformed[:, 0])[masks]
    vector = np.linspace(0.1, 0.3, len(masks)) + 0.02j
    np.testing.assert_allclose(ComponentSigmaOperator(context).matvec(vector), expected_action @ vector, atol=1e-10)


def test_charge_green_transforms_under_complex_orbital_rotation():
    model, _, _, context, sources = solved_charge()
    rotation = scipy.linalg.expm(1j * np.array([[0.1, 0.3j], [-0.3j, -0.2]]))
    rotated = FermionicLindbladModel(
        rotation.conj().T @ model.E @ rotation,
        rotation.conj().T @ model.gamma_loss @ rotation,
        rotation.conj().T @ model.gamma_gain @ rotation,
        U=0.0, symmetry="charge", interaction_sites=(0, 1),
        two_body_interaction=model.interaction,
    )
    oracle = factorized_cc_problem(rotated, "CCSD", build_next_rank_masks=False)
    static = ComponentStaticProblem.compile(oracle)
    solution = solve_component_ccsd(static, policy=NewtonKrylovPolicy(tolerance=1e-12))
    assert solution.converged
    eom = eom_ccsd_problem(oracle, build_next_rank_masks=False)
    rotated_context = component_eom_context(ComponentEOMProblem.compile(eom, static, charge=1), solution.amplitudes)
    rotated_sources = tuple(component_eom_sources(rotated_context, site=orbital) for orbital in range(2))
    options = dict(rtol=1e-12, restart=10, maxiter=30)
    original = component_retarded_matrix_green(ComponentSigmaOperator(context), sources, [0.0, 0.5], 0.1, **options).retarded
    observed = component_retarded_matrix_green(ComponentSigmaOperator(rotated_context), rotated_sources, [0.0, 0.5], 0.1, **options).retarded
    np.testing.assert_allclose(observed, rotation.conj().T @ original @ rotation, atol=1e-9)

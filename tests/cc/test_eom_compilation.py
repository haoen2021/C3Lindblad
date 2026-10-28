"""Production EOM preparation, rolled projections and shifted-solver gates."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from functools import cache
from itertools import combinations
import sys

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, gmres

from c3l import FermionicLindbladModel, SparseTwoBodyInteraction
from c3l.cc import component_tensor as component
from c3l.cc import s135_full
from c3l.cc.component_response import ComponentSigmaOperator, solve_component_shifted
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import factorized_cc_problem


SECTORS = ((0, 1), (0, -1), (1, 1), (1, -1))


def _production_api(name):
    pytest.importorskip("jax")
    from c3l.cc.component_jax import JaxRetainedEOMOperator
    from c3l.cc.s135_jax import JaxFullS135ResponseOperator

    return {
        "JaxRetainedEOMOperator": JaxRetainedEOMOperator,
        "JaxFullS135ResponseOperator": JaxFullS135ResponseOperator,
    }[name]


@cache
def _static_fixture():
    model = FermionicLindbladModel(
        np.asarray(((-0.2, 0.18j), (-0.18j, 0.25))),
        np.diag((0.0, 0.32)),
        np.diag((0.0, 0.11)),
        U=0.4,
    )
    problem = factorized_cc_problem(
        model,
        "CCSD",
        m_up=0.3,
        build_next_rank_masks=False,
        reference_mode="legacy_half_filled",
    )
    static = component.ComponentStaticProblem.compile(problem)
    indices = np.arange(problem.num_variables, dtype=np.float64)
    amplitudes = problem.tilde.unpack(5.0e-4 * np.cos(indices + 0.17))
    return problem, static, amplitudes


@cache
def _context(spin, charge):
    problem, static, amplitudes = _static_fixture()
    response = eom_ccsd_problem(
        problem, response_spin=spin, build_next_rank_masks=False
    )
    compiled = component.ComponentEOMProblem.compile(response, static, charge)
    return component.component_eom_context(compiled, amplitudes)


@cache
def _full_reference(spin, charge):
    context = _context(spin, charge)
    operator = s135_full.FullS135ResponseOperator(
        s135_full.CompiledFullS135Action(context, materialization_batch_size=4),
        s5_tile_size=7,
    )
    matrix = component.component_block_action(
        context, operator.coordinates, operator.coordinates, batch_size=11
    )
    return operator, matrix


@cache
def _operator(space, spin, charge, tile_size):
    name = "JaxRetainedEOMOperator" if space == "retained" else "JaxFullS135ResponseOperator"
    implementation = _production_api(name)
    pytest.importorskip("jax")
    if space == "retained":
        return implementation.from_cpu(
            _context(spin, charge),
            platform="cpu",
            production=False,
            projection_tile_size=tile_size,
        )
    reference, _ = _full_reference(spin, charge)
    cpu = s135_full.FullS135ResponseOperator(reference.action, s5_tile_size=tile_size)
    return implementation.from_cpu(cpu, platform="cpu", production=False)


def _vector(dimension):
    indices = np.arange(dimension, dtype=np.float64)
    return np.cos(indices + 0.31) + 0.7j * np.sin(0.6 * indices + 0.23)


@contextmanager
def _compilation_spy():
    """Observe real source calls without replacing any source implementation."""

    observed = {"materializations": [], "input_five_calls": 0}
    previous = sys.getprofile()

    def profile(frame, event, argument):
        if event == "call":
            if frame.f_code is component._materialize_terms_raw.__code__:
                observed["materializations"].append(
                    (frozenset(frame.f_locals["degrees"]), frame.f_locals["batch_size"])
                )
            elif frame.f_code is s135_full._input_five_plans.__code__:
                observed["input_five_calls"] += 1
        if previous is not None:
            previous(frame, event, argument)

    sys.setprofile(profile)
    try:
        yield observed
    finally:
        sys.setprofile(previous)


def _assert_restricted_materialization(observed):
    assert observed["input_five_calls"] == 0
    assert observed["materializations"], "the spy must observe real materialization"
    assert all(
        degrees and degrees <= {0, 1, 2, 3}
        for degrees, _ in observed["materializations"]
    )


@pytest.mark.parametrize("spin,charge", SECTORS)
def test_retained_jax_preparation_and_action_are_restricted(spin, charge):
    implementation = _production_api("JaxRetainedEOMOperator")
    context = _context(spin, charge)
    with _compilation_spy() as observed:
        operator = implementation.from_cpu(
            context, platform="cpu", production=False, projection_tile_size=3
        )
    _assert_restricted_materialization(observed)
    assert operator.plan.s5_dimension == 0
    assert operator.dimension == context.problem.coordinates.dimension
    assert all(prefix.input_degree in (1, 3) for prefix in operator.plan.prefixes)
    assert all(bucket.output_degree in (1, 3) for bucket in operator.plan.projections)
    vector = _vector(operator.dimension)
    expected = component.component_eom_sigma(context, vector).retained
    actual = operator.matvec(vector)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-11, atol=3.0e-11)
    assert actual.flags.writeable
    device_vector = operator.jax.device_put(vector, operator.device)
    np.testing.assert_allclose(
        np.asarray(operator.device_matvec(device_vector)), expected, rtol=3.0e-11, atol=3.0e-11
    )


@pytest.mark.parametrize("spin,charge", SECTORS)
@pytest.mark.parametrize("target_delta", (-1, 0, 1), ids=("T-1", "T", "T+1"))
def test_full_q5_tile_tails_match_dense_cpu_blocks(spin, charge, target_delta):
    _production_api("JaxFullS135ResponseOperator")
    reference, matrix = _full_reference(spin, charge)
    target_size = max(block.indices.shape[0] for block in reference.S5.blocks)
    tile_size = target_size - target_delta
    assert target_size == tile_size + target_delta
    operator = _operator("full", spin, charge, tile_size)
    vector = _vector(operator.dimension)
    expected = matrix @ vector
    actual = operator.matvec(vector)
    assert actual.shape == (reference.dimension,)
    assert actual.dtype == np.complex128
    assert actual.flags.writeable
    np.testing.assert_allclose(actual, expected, rtol=3.0e-11, atol=3.0e-11)
    split = reference.p_dimension
    blocks = operator.block_action(vector)
    for actual_block, expected_block in (
        (blocks.APP, matrix[:split, :split] @ vector[:split]),
        (blocks.AP5, matrix[:split, split:] @ vector[split:]),
        (blocks.A5P, matrix[split:, :split] @ vector[:split]),
        (blocks.A55, matrix[split:, split:] @ vector[split:]),
    ):
        np.testing.assert_allclose(actual_block, expected_block, rtol=3.0e-11, atol=3.0e-11)
    if target_delta == 1:
        profiled, _ = operator.profile_stages(vector)
        np.testing.assert_allclose(profiled, expected, rtol=3.0e-11, atol=3.0e-11)


def _equations(value):
    # Some JAX versions expose a Jaxpr's jaxpr attribute as itself.
    if hasattr(value, "eqns"):
        for equation in value.eqns:
            yield equation
            yield from _equations(equation.params)
    elif hasattr(value, "jaxpr"):
        yield from _equations(value.jaxpr)
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _equations(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            yield from _equations(nested)


@pytest.mark.parametrize("space", ("retained", "full"))
def test_projection_executables_keep_tiles_rolled(space):
    operator = _operator(space, 0, 1, 3)
    banks = tuple(
        operator.jax.ShapeDtypeStruct(
            (
                sum(
                    operator.plan.prefix_buckets[index].prefix_indices.size
                    for index in bank.source_buckets
                ),
            ) + (operator.plan.n,) * sum(bank.key),
            np.complex128,
        )
        for bank in operator.plan.acted_banks
    )
    checked = 0
    for kernel, data, target in zip(
        operator._projection_kernels,
        operator._projection_data,
        operator._coordinates,
        strict=True,
    ):
        if target.indices.shape[0] <= operator.projection_tile_size:
            continue
        graph = operator.jax.make_jaxpr(kernel)(target.indices, data[0], data[1], banks)
        loops = [
            equation for equation in _equations(graph)
            if equation.primitive.name in ("scan", "while")
        ]
        assert len(loops) == 1, "projection tiles must use one rolled loop"
        if loops[0].primitive.name == "scan":
            assert loops[0].params["unroll"] == 1
            assert loops[0].params["unroll"] is not True
            assert loops[0].params["length"] == (
                target.indices.shape[0] + operator.projection_tile_size - 1
            ) // operator.projection_tile_size
        checked += 1
    assert checked > 0, "the fixture must exercise multiple projection tiles"


@pytest.mark.parametrize("spin,charge", SECTORS)
@pytest.mark.parametrize("space", ("retained", "full"))
def test_shifted_solution_and_physical_response_match_cpu(space, spin, charge):
    operator = _operator(space, spin, charge, 3)
    context = _context(spin, charge)
    sources = component.component_eom_sources(context, site=0)
    source = np.zeros(operator.dimension, dtype=np.complex128)
    source[:sources.addition.size] = sources.addition + sources.removal
    shift = 0.37 + 0.41j
    if space == "retained":
        cpu = ComponentSigmaOperator(context)
        expected = solve_component_shifted(
            cpu, shift, source, rtol=1.0e-11, restart=30, maxiter=100
        )
        actual = operator.solve_shifted(
            shift, source, rtol=1.0e-11, restart=30, maxiter=100
        )
        assert expected.converged and actual.converged
        assert actual.relative_residual < 1.0e-10
        solution = actual.solution
        expected_solution = expected.solution
        residual = shift * solution - cpu.matvec(solution) - source
    else:
        _, matrix = _full_reference(spin, charge)
        shifted = shift * np.eye(operator.dimension) - matrix
        expected_solution = np.linalg.solve(shifted, source)
        action = LinearOperator(
            (operator.dimension, operator.dimension),
            matvec=lambda vector: shift * vector - operator.matvec(vector),
            dtype=np.complex128,
        )
        solution, info = gmres(
            action, source, rtol=1.0e-11, atol=0.0, restart=30, maxiter=100
        )
        assert info == 0
        residual = shifted @ solution - source
    assert np.linalg.norm(source) > 0.0
    assert np.linalg.norm(residual) / np.linalg.norm(source) < 1.0e-10
    np.testing.assert_allclose(solution, expected_solution, rtol=2.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(
        sources.sink @ solution[:sources.sink.size],
        sources.sink @ expected_solution[:sources.sink.size],
        rtol=2.0e-9,
        atol=2.0e-9,
    )


def test_general_interaction_retained_and_full_actions_match_cpu():
    """Preserve general complex interaction terms beyond the Hubbard fixture."""

    retained_class = _production_api("JaxRetainedEOMOperator")
    full_class = _production_api("JaxFullS135ResponseOperator")
    pytest.importorskip("jax")
    generator = np.random.default_rng(105)
    pairs = tuple(combinations(range(4), 2))
    terms = []
    for up_count in (0, 1, 2):
        sector = tuple(pair for pair in pairs if sum(orbital % 2 == 0 for orbital in pair) == up_count)
        entries = generator.normal(size=(len(sector), len(sector))) + 1j * generator.normal(size=(len(sector), len(sector)))
        entries = 0.12 * (entries + entries.conj().T) / max(2 * len(sector), 1)
        for row, creation in enumerate(sector):
            for column, annihilation in enumerate(sector):
                terms.append((*creation, *annihilation, complex(entries[row, column])))
    model = FermionicLindbladModel(
        np.asarray(((-0.2, 0.07 + 0.014j), (0.07 - 0.014j, 0.25))),
        np.diag((0.3, 0.32)), np.diag((0.17, 0.18)), U=0.0,
        interaction_sites=(0, 1), two_body_interaction=SparseTwoBodyInteraction(4, tuple(terms)),
    )
    static = component.ComponentStaticProblem.compile(
        factorized_cc_problem(model, "CCSD", build_next_rank_masks=False), fuse_products=True,
    )
    amplitudes = static.oracle.tilde.unpack(0.001 * np.cos(np.arange(static.oracle.num_variables) + 0.13))
    assert np.linalg.norm(amplitudes) > 0.0
    response = eom_ccsd_problem(
        static.oracle, response_spin=0, build_next_rank_masks=False
    )
    problem = component.ComponentEOMProblem.compile(response, static, charge=1)
    context = component.component_eom_context(problem, amplitudes)
    retained = retained_class.from_cpu(
        context, platform="cpu", production=False, projection_tile_size=3
    )
    retained_vector = _vector(retained.dimension)
    np.testing.assert_allclose(
        retained.matvec(retained_vector),
        component.component_eom_sigma(context, retained_vector).retained,
        rtol=3.0e-11,
        atol=3.0e-11,
    )
    cpu_full = s135_full.FullS135ResponseOperator(
        s135_full.CompiledFullS135Action(context, materialization_batch_size=4),
        s5_tile_size=3,
    )
    matrix = component.component_block_action(
        context, cpu_full.coordinates, cpu_full.coordinates, batch_size=11
    )
    full = full_class.from_cpu(cpu_full, platform="cpu", production=False)
    full_vector = _vector(full.dimension)
    np.testing.assert_allclose(
        full.matvec(full_vector), matrix @ full_vector, rtol=3.0e-11, atol=3.0e-11
    )


@pytest.mark.parametrize("spin,charge", SECTORS)
def test_retained_linear_transpose_matches_cpu(spin, charge):
    """Expose valid linear scan inputs to the unchanged JAX shifted solver."""

    operator = _operator("retained", spin, charge, 3)
    context = _context(spin, charge)
    coordinates = context.problem.coordinates
    matrix = component.component_block_action(
        context, coordinates, coordinates, batch_size=11
    )
    vector = _vector(operator.dimension)
    transpose = operator.jax.linear_transpose(
        operator._device_action,
        operator.jax.ShapeDtypeStruct(vector.shape, np.complex128),
    )
    transposed = operator.jax.jit(transpose)(
        operator.jax.device_put(vector, operator.device)
    )[0]
    np.testing.assert_allclose(
        np.asarray(transposed), matrix.T @ vector, rtol=3.0e-11, atol=3.0e-11
    )


def test_retained_preparation_ignores_injected_rank_five_input():
    implementation = _production_api("JaxRetainedEOMOperator")
    original = _context(0, 1)
    dimension = original.problem.coordinates.n
    forbidden = component._ResponsePlanTerm(
        input_degree=5, coefficient=0.17 + 0.09j, creation_factors=(),
        contractions=({0: np.ones(dimension)}, {2: np.ones(dimension)}),
    )
    context = replace(original, response_plan=original.response_plan + (forbidden,))
    with _compilation_spy() as observed:
        operator = implementation.from_cpu(context, platform="cpu", production=False, projection_tile_size=3)
    _assert_restricted_materialization(observed)
    vector = _vector(operator.dimension)
    np.testing.assert_allclose(operator.matvec(vector), component.component_eom_sigma(original, vector).retained, atol=3e-11, rtol=3e-11)


def test_empty_projection_target_returns_empty_complex_array():
    implementation = _production_api("JaxFullS135ResponseOperator")
    original = _operator("full", 0, 1, 3)
    coordinate = original.plan.coordinates[0]
    empty = replace(coordinate, indices=np.empty((0, coordinate.indices.shape[1]), dtype=coordinate.indices.dtype), vector_positions=np.empty(0, dtype=np.int64))
    plan = replace(original.plan, p_dimension=0, s5_dimension=0, coordinates=(empty,), prefixes=(), projections=(), prefix_buckets=(), acted_banks=(), projection_kernels=(), rapidity_diagonal=np.empty(0, dtype=np.complex128))
    operator = implementation(plan, jax=original.jax, jnp=original.jnp, device=original.device, runtime=original.runtime, projection_tile_size=3)
    actual = operator._projection_kernels[0](operator._coordinates[0].indices, (), (), ())
    assert actual.shape == (0,)
    assert actual.dtype == np.complex128

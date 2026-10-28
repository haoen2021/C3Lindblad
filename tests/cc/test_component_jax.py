"""CPU/JAX parity for the existing static and retained component actions."""

from __future__ import annotations

import numpy as np
import pytest

from c3l import FermionicLindbladModel, SparseTwoBodyInteraction
from c3l.cc.component_tensor import (
    ComponentEOMProblem,
    ComponentStaticProblem,
    component_eom_context,
    component_eom_sigma,
    component_eom_sources,
    component_static_jvp,
    component_static_residual,
)
from c3l.cc.component_response import (
    ComponentSigmaOperator,
    component_retarded_matrix_green,
    component_retarded_matrix_green_arnoldi,
    component_retarded_matrix_time,
    solve_component_shifted,
)
from c3l.cc.component_solver import solve_component_ccsd
from c3l.cc.newton_krylov import NewtonKrylovPolicy
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import factorized_cc_problem
from tests.s135_fixture import s135_validation_model


def _fixture():
    cc = factorized_cc_problem(
        s135_validation_model(),
        "CCSD",
        m_up=0.3,
        build_next_rank_masks=False,
        reference_mode="legacy_half_filled",
    )
    static = ComponentStaticProblem.compile(cc)
    index = np.arange(cc.num_variables, dtype=np.float64)
    amplitudes = cc.tilde.unpack(7.0e-4 * np.cos(index + 0.13))
    direction = cc.tilde.unpack(0.2 * np.sin(index + 0.31))
    return cc, static, amplitudes, direction


def _two_source_fixture():
    U = 0.4
    E = np.asarray(((-0.5 * U, 0.18j), (-0.18j, 0.25)))
    model = FermionicLindbladModel(
        E,
        np.diag((0.0, 0.32)),
        np.diag((0.0, 0.11)),
        U=U,
    )
    cc = factorized_cc_problem(
        model,
        "CCSD",
        m_up=0.3,
        build_next_rank_masks=False,
        reference_mode="legacy_half_filled",
    )
    static = ComponentStaticProblem.compile(cc)
    index = np.arange(cc.num_variables, dtype=np.float64)
    amplitudes = cc.tilde.unpack(5.0e-4 * np.cos(index + 0.17))
    eom = eom_ccsd_problem(cc, response_spin=0, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(problem, amplitudes)
    sources = tuple(component_eom_sources(context, site=site) for site in (0, 1))
    return context, sources


def test_jax_static_residual_and_jvp_match_existing_component_actions() -> None:
    """The GPU lowering must preserve the nonlinear residual and analytic JVP."""

    from c3l.cc.component_jax import JaxStaticCCSDOperator

    _, static, amplitudes, direction = _fixture()
    operator = JaxStaticCCSDOperator.from_cpu(
        static,
        platform="cpu",
        production=False,
    )
    expected_residual = component_static_residual(static, amplitudes).retained
    expected_jvp = component_static_jvp(
        static,
        amplitudes,
        direction,
    ).retained

    np.testing.assert_allclose(
        operator.residual(amplitudes),
        expected_residual,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    np.testing.assert_allclose(
        operator.jvp(amplitudes, direction),
        expected_jvp,
        atol=3.0e-11,
        rtol=3.0e-11,
    )

    profile = operator.profile(amplitudes, direction, warm_repeats=2)
    assert profile.first_residual_seconds > 0.0
    assert profile.first_jvp_seconds > 0.0
    assert len(profile.warm_residual_seconds) == 2
    assert len(profile.warm_jvp_seconds) == 2


def test_jax_retained_eom_matches_existing_component_action() -> None:
    """The retained device action must keep the canonical Q1+Q3 ordering."""

    from c3l.cc.component_jax import JaxRetainedEOMOperator

    cc, static, amplitudes, _ = _fixture()
    eom = eom_ccsd_problem(cc, response_spin=0, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(problem, amplitudes)
    operator = JaxRetainedEOMOperator.from_cpu(
        context,
        platform="cpu",
        production=False,
    )
    rng = np.random.default_rng(20260823)
    vector = rng.normal(size=problem.coordinates.dimension) + 1j * rng.normal(
        size=problem.coordinates.dimension
    )
    expected = component_eom_sigma(context, vector).retained

    np.testing.assert_allclose(
        operator.matvec(vector), expected, atol=3.0e-11, rtol=3.0e-11
    )


def test_jax_retained_shifted_solve_matches_cpu_and_true_residual() -> None:
    """Shifted Krylov vectors stay on device while the physical solution agrees."""

    from c3l.cc.component_jax import JaxRetainedEOMOperator

    cc, static, amplitudes, _ = _fixture()
    eom = eom_ccsd_problem(cc, response_spin=0, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(problem, amplitudes)
    cpu = ComponentSigmaOperator(context)
    gpu = JaxRetainedEOMOperator.from_cpu(
        context,
        platform="cpu",
        production=False,
    )
    rng = np.random.default_rng(20260824)
    source = rng.normal(size=cpu.dimension) + 1j * rng.normal(size=cpu.dimension)
    shift = 0.37 + 0.12j
    expected = solve_component_shifted(
        cpu,
        shift,
        source,
        rtol=1.0e-11,
        restart=30,
        maxiter=100,
    )
    observed = gpu.solve_shifted(
        shift,
        source,
        rtol=1.0e-11,
        restart=30,
        maxiter=100,
    )

    assert expected.converged
    assert observed.converged
    assert observed.relative_residual < 1.0e-10
    np.testing.assert_allclose(
        observed.solution, expected.solution, atol=2.0e-9, rtol=2.0e-9
    )


def test_jax_newton_gmres_reaches_the_same_small_ccsd_root() -> None:
    """Static Krylov vectors and JVPs stay on device through the Newton solve."""

    from c3l.cc.component_jax import solve_component_ccsd_jax

    _, static, _, _ = _fixture()
    policy = NewtonKrylovPolicy(
        tolerance=1.0e-9,
        maximum_newton_iterations=12,
        gmres_restart=30,
        gmres_maximum_restart_cycles=8,
        minimum_gmres_relative_tolerance=1.0e-9,
        maximum_gmres_relative_tolerance=1.0e-3,
    )
    expected = solve_component_ccsd(
        static,
        policy=policy,
        compute_rank6=False,
    )
    accepted: list[tuple[int, float]] = []
    observed = solve_component_ccsd_jax(
        static,
        policy=policy,
        platform="cpu",
        production=False,
        accepted_step_callback=lambda iteration, amplitudes, residual: accepted.append(
            (iteration, residual)
        ),
    )

    assert expected.converged
    assert observed.converged
    assert observed.packed_residual_norm < 1.0e-9
    assert observed.jvp_evaluations > 0
    assert len(accepted) == observed.iterations
    np.testing.assert_allclose(
        observed.amplitudes,
        expected.amplitudes,
        atol=2.0e-7,
        rtol=2.0e-7,
    )


def test_jax_retained_matrix_response_keeps_sources_and_sinks_on_device() -> None:
    """All frequencies retain sink-row/source-column ordering and true residuals."""

    from c3l.cc.component_jax import component_retarded_matrix_green_jax

    context, sources = _two_source_fixture()
    omega = np.asarray((-0.35, 0.15))
    expected = component_retarded_matrix_green(
        ComponentSigmaOperator(context),
        sources,
        omega,
        0.12,
        rtol=1.0e-11,
        restart=30,
        maxiter=100,
    )
    observed = component_retarded_matrix_green_jax(
        context,
        sources,
        omega,
        0.12,
        platform="cpu",
        production=False,
        rtol=1.0e-11,
        restart=30,
        maxiter=100,
    )

    assert observed.sites == (0, 1)
    assert np.max(observed.addition_relative_residuals) < 1.0e-10
    assert np.max(observed.removal_relative_residuals) < 1.0e-10
    np.testing.assert_allclose(
        observed.retarded, expected.retarded, atol=2.0e-9, rtol=2.0e-9
    )
    assert np.max(abs(observed.retarded[:, 0, 1] - observed.retarded[:, 1, 0])) > 1.0e-3


def test_jax_retained_arnoldi_response_matches_cpu_projection() -> None:
    """A single device basis must serve every retained frequency and sink."""

    from c3l.cc.component_jax import (
        component_retarded_matrix_green_arnoldi_jax,
    )

    context, sources = _two_source_fixture()
    omega = np.asarray((-0.35, 0.15))
    expected = component_retarded_matrix_green_arnoldi(
        ComponentSigmaOperator(context),
        sources,
        omega,
        0.12,
        maximum_krylov_dimension=60,
        residual_tolerance=1.0e-10,
        minimum_krylov_dimension=1,
        check_interval=1,
    )
    observed = component_retarded_matrix_green_arnoldi_jax(
        context,
        sources,
        omega,
        0.12,
        platform="cpu",
        production=False,
        maximum_krylov_dimension=60,
        residual_tolerance=1.0e-10,
    )

    assert observed.sites == (0, 1)
    assert len(observed.krylov_dimensions) == 4
    assert np.max(observed.addition_relative_residuals) < 1.0e-10
    assert np.max(observed.removal_relative_residuals) < 1.0e-10
    np.testing.assert_allclose(
        observed.retarded, expected.retarded, atol=2.0e-9, rtol=2.0e-9
    )


def test_jax_retained_time_response_uses_device_arnoldi() -> None:
    """Device Arnoldi must reproduce time propagation and shifted residual gates."""

    from c3l.cc.component_jax import component_retarded_matrix_time_jax

    context, sources = _two_source_fixture()
    time = np.asarray((0.0, 0.1, 0.2))
    shifts = np.asarray((-0.2 + 0.12j, 0.25 + 0.12j))
    expected = component_retarded_matrix_time(
        ComponentSigmaOperator(context),
        sources,
        time,
        shifts,
        maximum_krylov_dimension=60,
        residual_tolerance=1.0e-10,
        include_channels=False,
    )
    observed = component_retarded_matrix_time_jax(
        context,
        sources,
        time,
        shifts,
        platform="cpu",
        production=False,
        maximum_krylov_dimension=60,
        residual_tolerance=1.0e-10,
    )

    assert observed.projected_backend == "jax_uniform_step_expm"
    assert np.max(observed.relative_residuals) < 1.0e-10
    np.testing.assert_allclose(
        observed.retarded, expected.retarded, atol=2.0e-9, rtol=2.0e-9
    )
    np.testing.assert_allclose(observed.retarded[0], -1j * np.eye(2), atol=3.0e-13)


def test_jax_static_action_accepts_general_exchange_terms() -> None:
    """The device contraction graph must not fall back to density-only algebra."""

    interaction = SparseTwoBodyInteraction(
        4,
        (
            (0, 3, 1, 2, 0.08),
            (1, 2, 0, 3, 0.08),
        ),
    )
    E = np.asarray(((-0.2, -0.07), (-0.07, 0.15)), dtype=np.complex128)
    loss = np.diag((0.30, 0.26)).astype(np.complex128)
    gain = np.diag((0.12, 0.10)).astype(np.complex128)
    model = FermionicLindbladModel(
        E,
        loss,
        gain,
        U=0.0,
        interaction_sites=(0, 1),
        two_body_interaction=interaction,
    )
    cc = factorized_cc_problem(
        model,
        "CCSD",
        build_next_rank_masks=False,
    )
    static = ComponentStaticProblem.compile(cc)
    index = np.arange(cc.num_variables, dtype=np.float64)
    amplitudes = cc.tilde.unpack(4.0e-4 * np.cos(index + 0.2))
    direction = cc.tilde.unpack(0.1 * np.sin(index + 0.4))
    from c3l.cc.component_jax import JaxStaticCCSDOperator

    operator = JaxStaticCCSDOperator.from_cpu(
        static,
        platform="cpu",
        production=False,
    )
    expected_residual = component_static_residual(static, amplitudes).retained
    expected_jvp = component_static_jvp(static, amplitudes, direction).retained

    np.testing.assert_allclose(
        operator.residual(amplitudes), expected_residual, atol=3.0e-11, rtol=3.0e-11
    )
    np.testing.assert_allclose(
        operator.jvp(amplitudes, direction), expected_jvp, atol=3.0e-11, rtol=3.0e-11
    )


def test_component_jax_actions_match_on_available_gpu() -> None:
    """One optional device gate covers zero/nonzero static and retained actions."""

    import jax

    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("no JAX GPU is visible")
    if not devices:
        pytest.skip("no JAX GPU is visible")
    from c3l.cc.component_jax import (
        JaxRetainedEOMOperator,
        JaxStaticCCSDOperator,
    )

    cc, static, amplitudes, direction = _fixture()
    static_gpu = JaxStaticCCSDOperator.from_cpu(static, platform="gpu")
    for point in (np.zeros_like(amplitudes), amplitudes):
        np.testing.assert_allclose(
            static_gpu.residual(point),
            component_static_residual(static, point).retained,
            atol=3.0e-11,
            rtol=3.0e-11,
        )
    np.testing.assert_allclose(
        static_gpu.jvp(amplitudes, direction),
        component_static_jvp(static, amplitudes, direction).retained,
        atol=3.0e-11,
        rtol=3.0e-11,
    )

    eom = eom_ccsd_problem(cc, response_spin=0, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(problem, amplitudes)
    retained_gpu = JaxRetainedEOMOperator.from_cpu(context, platform="gpu")
    rng = np.random.default_rng(20260825)
    vector = rng.normal(size=retained_gpu.dimension) + 1j * rng.normal(
        size=retained_gpu.dimension
    )
    np.testing.assert_allclose(
        retained_gpu.matvec(vector),
        component_eom_sigma(context, vector).retained,
        atol=3.0e-11,
        rtol=3.0e-11,
    )

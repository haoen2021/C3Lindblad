import numpy as np
import pytest

from c3l import FermionicLindbladModel
from c3l.cc.component_tensor import ComponentStaticProblem, ComponentEOMProblem, component_eom_context, component_eom_sources
from c3l.cc.factorized import factorized_cc_problem
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.newton_krylov import NewtonKrylovPolicy


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
def test_same_jax_static_and_response_pipeline_matches_dense(platform):
    import jax
    from c3l.cc.component_jax import solve_component_ccsd_jax, JaxRetainedEOMOperator
    from c3l.cc.component_solver import solve_component_ccsd
    from c3l.cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator
    from c3l.cc.s135_jax import JaxFullS135ResponseOperator
    from c3l.green.resident_arnoldi import ResidentArnoldi
    from examples.shared.response import reduced_gmres

    jax.config.update("jax_enable_x64", True)
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"No {platform} device")
    model = FermionicLindbladModel(
        np.array([[-.2, .18j], [-.18j, .25]]),
        np.diag([0., .32]), np.diag([0., .11]), U=.4,
    )
    oracle = factorized_cc_problem(model, "CCSD", m_up=.3, build_next_rank_masks=False)
    static = ComponentStaticProblem.compile(oracle)
    policy = NewtonKrylovPolicy(tolerance=1e-10, maximum_newton_iterations=30, gmres_restart=64)
    expected = solve_component_ccsd(static, None, policy=policy)
    solution = solve_component_ccsd_jax(static, None, policy=policy, platform=platform, production=True)
    assert expected.converged and solution.converged
    np.testing.assert_allclose(solution.amplitudes, expected.amplitudes, atol=1e-9, rtol=1e-9)
    eom = eom_ccsd_problem(oracle, response_spin=0, build_next_rank_masks=False)
    context = component_eom_context(ComponentEOMProblem.compile(eom, static, charge=1), solution.amplitudes)
    source = component_eom_sources(context, site=0)
    cpu_full = FullS135ResponseOperator(CompiledFullS135Action(context))
    shifts = np.array([-.5 + .1j, .3 + .1j, 1j, 3j])
    for operator in (
        JaxRetainedEOMOperator.from_cpu(context, platform=platform, production=True),
        JaxFullS135ResponseOperator.from_cpu(cpu_full, platform=platform, production=True),
    ):
        assert operator.device == device
        dimension = operator.dimension
        rhs = np.zeros(dimension, complex)
        sink = np.zeros(dimension, complex)
        rhs[:len(source.addition)] = source.addition + source.removal
        sink[:len(source.sink)] = source.sink
        chain = ResidentArnoldi(operator.device_matvec, rhs, sink[None, :], dimension, device=device)
        chain.advance(dimension)
        coefficients, residuals = reduced_gmres(
            chain.hessenberg[:chain.current + 1, :chain.current], np.linalg.norm(rhs), shifts,
        )
        from c3l.cc.component_response import ComponentSigmaOperator

        reference = cpu_full if dimension == cpu_full.dimension else ComponentSigmaOperator(context)
        matrix = np.column_stack([reference.matvec(column) for column in np.eye(dimension)])
        observed = coefficients @ chain.projected[:, :chain.current].T
        dense = np.array([sink @ np.linalg.solve(shift * np.eye(dimension) - matrix, rhs) for shift in shifts])
        np.testing.assert_allclose(observed[:, 0], dense, atol=1e-9, rtol=1e-9)
        assert max(residuals) < 1e-9
        assert chain.basis.devices() == {device}


@pytest.mark.parametrize("kind", ["frequency", "time"])
def test_retained_helpers_keep_reduced_arrays_on_selected_device(monkeypatch, kind):
    import jax
    import jax.numpy as jnp
    from c3l.cc.component_jax import component_retarded_matrix_green_arnoldi_jax, component_retarded_matrix_time_jax
    from tests.cc.test_component_jax import _two_source_fixture
    from c3l.cc.component_response import ComponentSigmaOperator, component_retarded_matrix_green_arnoldi, component_retarded_matrix_time

    jax.config.update("jax_enable_x64", True)
    devices = jax.devices("cpu")
    if jax.default_backend() == "cpu" and len(devices) < 2:
        pytest.skip("Requires GPU default or two CPU devices to detect misplaced arrays")
    selected = devices[-1]
    original = jax.devices
    monkeypatch.setattr(jax, "devices", lambda backend=None: [selected] if backend == "cpu" else original(backend))
    allocations = []

    def track(factory):
        def allocate(*args, **kwargs):
            array = factory(*args, **kwargs)
            if isinstance(array, jax.Array) and not isinstance(array, jax.core.Tracer):
                allocations.append(array.devices())
            return array
        return allocate

    monkeypatch.setattr(jnp, "zeros", track(jnp.zeros))
    monkeypatch.setattr(jnp, "eye", track(jnp.eye))
    context, sources = _two_source_fixture()
    frequency = np.array([-.35, .15])
    options = dict(maximum_krylov_dimension=60, residual_tolerance=1e-10)
    if kind == "frequency":
        observed = component_retarded_matrix_green_arnoldi_jax(
            context, sources, frequency, .12, platform="cpu", production=True, **options,
        )
        expected = component_retarded_matrix_green_arnoldi(
            ComponentSigmaOperator(context), sources, frequency, .12, **options,
        )
    else:
        times = np.array([0., .1, .2])
        observed = component_retarded_matrix_time_jax(
            context, sources, times, frequency + .12j, platform="cpu", production=True, **options,
        )
        expected = component_retarded_matrix_time(
            ComponentSigmaOperator(context), sources, times, frequency + .12j, **options,
        )
    np.testing.assert_allclose(observed.retarded, expected.retarded, atol=2e-9, rtol=2e-9)
    assert allocations and all(devices == {selected} for devices in allocations)

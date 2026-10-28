from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@pytest.fixture
def problem(monkeypatch):
    from examples.shared import anderson_response, multiorbital_response

    def forbidden(*args, **kwargs):
        raise AssertionError("Resident examples must not use host matvecs or disk bases")

    monkeypatch.setattr(np.lib.format, "open_memmap", forbidden)
    monkeypatch.setattr(anderson_response, "PREFIXES", (3, 6, 12))
    monkeypatch.setattr(multiorbital_response, "PREFIXES", (3, 6, 12))
    rng = np.random.default_rng(712)
    matrix = np.diag(np.arange(1, 9) - 0.4j).astype(complex)
    matrix += np.triu(rng.normal(size=(8, 8)), 1) * 0.1j
    device_matrix = jnp.asarray(matrix)
    operator = SimpleNamespace(
        dimension=8, matvec=forbidden,
        device_matvec=jax.jit(lambda vector: device_matrix @ vector),
        jax=jax, jnp=jnp, device=jax.devices()[0],
    )
    source = SimpleNamespace(
        site=0, addition=rng.normal(size=8) + 1j * rng.normal(size=8),
        removal=rng.normal(size=8) + 1j * rng.normal(size=8),
    )
    sinks = tuple(SimpleNamespace(site=site, sink=rng.normal(size=8)) for site in (0, 1))
    spec = SimpleNamespace(omega=np.array([-1.0, 0.3, 1.0]), eta=0.2, name="small")
    return matrix, operator, source, sinks, spec


def test_combined_resident_ignores_legacy_basis_and_reuses_compact(problem, tmp_path):
    from examples.shared.multiorbital_response import _run_combined_source

    matrix, operator, source, sinks, spec = problem
    legacy = tmp_path / "checkpoint" / "basis.npy"
    legacy.parent.mkdir()
    legacy.write_bytes(b"retired checkpoint: must not read or overwrite")
    result = _run_combined_source(operator, source, sinks, spec, tmp_path, provenance="original")
    expected = np.array([
        np.vstack([item.sink for item in sinks])
        @ np.linalg.solve(shift * np.eye(8) - matrix, source.addition + source.removal)
        for shift in spec.omega + 1j * spec.eta
    ])
    np.testing.assert_allclose(result["G_full_m8"], expected, rtol=1e-11, atol=1e-11)
    assert str(result["orthogonalization_backend"]) == f"{operator.device.platform}_resident_cgs2"
    assert legacy.read_bytes() == b"retired checkpoint: must not read or overwrite"
    assert list(tmp_path.rglob("basis.npy")) == [legacy]
    operator.device_matvec = operator.matvec
    reused = _run_combined_source(operator, source, sinks, spec, tmp_path, provenance="original")
    np.testing.assert_array_equal(reused["G_full_m8"], result["G_full_m8"])
    with pytest.raises(RuntimeError, match="provenance"):
        _run_combined_source(operator, source, sinks, spec, tmp_path, provenance="changed")
    spec.eta = 0.7
    with pytest.raises(RuntimeError, match="eta"):
        _run_combined_source(operator, source, sinks, spec, tmp_path, provenance="original")


def test_partial_compact_restarts_fresh(problem, tmp_path):
    from examples.shared.multiorbital_response import _run_combined_source

    _, operator, source, sinks, spec = problem
    result = _run_combined_source(operator, source, sinks, spec, tmp_path, provenance="original")
    partial = {name: value for name, value in result.items() if not name.endswith("_m8")}
    partial.update(final_m=6, prefixes=np.array([3, 6]),
                   observable_change_relative_l2=np.array([np.nan, 1.0]),
                   converged_observable=False)
    partial["action_wall_seconds"] = np.array([-1000.0])
    np.savez_compressed(tmp_path / "compact.npz", **partial)
    resumed = _run_combined_source(operator, source, sinks, spec, tmp_path, provenance="original")
    np.testing.assert_allclose(resumed["G_full_m8"], result["G_full_m8"], atol=1e-12)
    assert np.all(resumed["action_wall_seconds"] >= 0)
    np.testing.assert_array_equal(resumed["prefixes"], [3, 6, 8])


@pytest.mark.parametrize("kind", ["frequency", "time"])
@pytest.mark.parametrize("short_removal", [False, True])
def test_separate_resident_channels_match_dense(problem, tmp_path, kind, short_removal):
    from examples.shared.anderson_response import _run_source

    matrix, operator, source, sinks, spec = problem
    if short_removal:
        source.removal = np.eye(8)[0].astype(complex)
    grid = spec.omega if kind == "frequency" else np.array([0.0, 0.2, 0.7])
    eta = spec.eta if kind == "frequency" else None
    result = _run_source(operator, source, sinks, grid, kind, tmp_path,
                         eta=eta, provenance="test")
    for branch in ("addition", "removal"):
        rhs = getattr(source, branch)
        expected = np.array([
            np.vstack([item.sink for item in sinks]) @ (
                np.linalg.solve((point + 1j * eta) * np.eye(8) - matrix, rhs)
                if kind == "frequency" else -1j * expm(-1j * matrix * point) @ rhs
            )
            for point in grid
        ])
        np.testing.assert_allclose(result[f"G_{branch}_m8"], expected, rtol=1e-11, atol=1e-11)
    assert not list(tmp_path.rglob("basis.npy"))


def test_zero_removal_and_early_closure(problem, tmp_path):
    from examples.shared.anderson_response import _run_source, _source_metadata

    _, operator, source, sinks, spec = problem
    source.addition = np.eye(8)[0].astype(complex)
    source.removal = np.zeros(8, complex)
    result = _run_source(operator, source, sinks, spec.omega, "frequency", tmp_path,
                         eta=spec.eta, provenance="test")
    assert int(result["final_m"]) == 1
    assert bool(result["converged_observable"])
    np.testing.assert_array_equal(result["G_removal_m1"], 0)
    reused = _run_source(operator, source, sinks, spec.omega, "frequency", tmp_path,
                         eta=spec.eta, provenance="test")
    np.testing.assert_array_equal(reused["G_full_m1"], result["G_full_m1"])
    metadata = _source_metadata(result)
    assert metadata["cold_action_wall_seconds_removal"] == 0.0
    assert metadata["cold_action_wall_seconds_addition"] >= 0.0


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("runner_name", ["combined", "separate"])
def test_resident_response_uses_selected_device(problem, tmp_path, platform, runner_name):
    from examples.shared import anderson_response, multiorbital_response

    try:
        selected = jax.devices(platform)[0]
    except RuntimeError:
        if platform == "gpu":
            pytest.skip("GPU unavailable")
        raise
    matrix, operator, source, sinks, spec = problem
    operator.device = selected
    device_matrix = jax.device_put(matrix, selected)
    action = jax.jit(lambda vector: device_matrix @ vector)
    observed = []

    def selected_action(vector):
        observed.append(vector.device)
        assert vector.device == selected
        return action(vector)

    operator.device_matvec = selected_action
    if runner_name == "combined":
        result = multiorbital_response._run_combined_source(
            operator, source, sinks, spec, tmp_path, provenance="selected-device"
        )
    else:
        result = anderson_response._run_source(
            operator, source, sinks, spec.omega, "frequency", tmp_path,
            eta=spec.eta, provenance="selected-device",
        )
    assert observed
    assert str(result["orthogonalization_backend"]) == f"{platform}_resident_cgs2"
    expected = np.array([
        np.vstack([item.sink for item in sinks])
        @ np.linalg.solve(shift * np.eye(8) - matrix, source.addition + source.removal)
        for shift in spec.omega + 1j * spec.eta
    ])
    np.testing.assert_allclose(result["G_full_m8"], expected, rtol=1e-11, atol=1e-11)


def test_junction_resident_gmres_checks_true_residual(problem, tmp_path):
    from examples.junction.cc import solve_source

    matrix, operator, source, sinks, spec = problem
    shifts = spec.omega + 1j * spec.eta
    rows = np.vstack([item.sink for item in sinks])
    column, record = solve_source(operator, source.addition, rows, shifts, tmp_path, 12)
    expected = np.array([
        rows @ np.linalg.solve(shift * np.eye(8) - matrix, source.addition)
        for shift in shifts
    ])
    np.testing.assert_allclose(column, expected, atol=1e-11)
    assert record["converged"]
    assert record["history"][-1]["true_residual"] < 1e-11
    assert not list(tmp_path.rglob("basis.npy"))
    empty, record = solve_source(operator, np.zeros(8), rows, shifts, tmp_path, 12)
    np.testing.assert_array_equal(empty, 0)
    assert record["converged"]


def test_junction_rejects_nonfinite_full_residual(problem, tmp_path):
    from examples.junction.cc import solve_source

    _, operator, source, sinks, spec = problem
    action = operator.device_matvec
    calls = 0

    def corrupted(vector):
        nonlocal calls
        calls += 1
        return action(vector) if calls <= 8 else jnp.full_like(vector, jnp.nan)

    operator.device_matvec = corrupted
    with pytest.raises(FloatingPointError, match="full residual"):
        solve_source(operator, source.addition, np.vstack([item.sink for item in sinks]),
                     spec.omega + 1j * spec.eta, tmp_path, 8)


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("rank", ["S13", "S135", "anderson_S135"])
def test_interacting_eom_resident_matches_cpu_dense(tmp_path, monkeypatch, rank, platform):
    from c3l.cc.factorized import factorized_cc_problem
    from c3l.cc.component_tensor import (
        ComponentStaticProblem, ComponentEOMProblem,
        component_eom_context, component_eom_sources,
    )
    from c3l.cc.eom import eom_ccsd_problem
    from c3l.cc.component_response import ComponentSigmaOperator
    from c3l.cc.component_jax import JaxRetainedEOMOperator
    from c3l.cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator
    from examples.shared import multiorbital_response as runner
    from tests.s135_fixture import s135_validation_model

    try:
        jax.devices(platform)
    except RuntimeError:
        if platform == "gpu":
            pytest.skip("GPU unavailable")
        raise
    oracle = factorized_cc_problem(s135_validation_model(), "CCSD", m_up=0.3)
    static = ComponentStaticProblem.compile(oracle)
    amplitudes = oracle.tilde.unpack(6e-4 * np.cos(np.arange(oracle.num_variables) + 0.19))
    response = ComponentEOMProblem.compile(
        eom_ccsd_problem(oracle, response_spin=0, build_next_rank_masks=False), static, 1
    )
    context = component_eom_context(response, amplitudes)
    sources = (component_eom_sources(context, site=0),)
    if rank == "S13":
        cpu = ComponentSigmaOperator(context)
        operator = JaxRetainedEOMOperator.from_cpu(context, platform=platform, production=True)
    else:
        cpu = FullS135ResponseOperator(CompiledFullS135Action(context))
        if rank == "S135":
            operator, _, _ = runner._build_full_operator(context, device=platform)
        else:
            from examples.shared.anderson_response import _build_operator

            operator, sources, _, _ = _build_operator(static, amplitudes, (0,), device=platform)
    dimension = operator.dimension
    identity = np.eye(dimension, dtype=complex)
    matrix = np.column_stack([cpu.matvec(vector) for vector in identity])
    rhs = np.zeros(dimension, complex)
    sink = np.zeros(dimension, complex)
    rhs[:sources[0].addition.size] = sources[0].addition + sources[0].removal
    sink[:sources[0].sink.size] = sources[0].sink
    monkeypatch.setattr(runner, "PREFIXES", (4, dimension))
    spec = SimpleNamespace(omega=np.array([-1.0, 0.0, 1.0]), eta=0.2, name=rank)
    result = runner._run_combined_source(operator, sources[0], sources, spec, tmp_path, provenance=rank)
    expected = np.array([
        sink @ np.linalg.solve(shift * identity - matrix, rhs)
        for shift in spec.omega + 1j * spec.eta
    ])
    np.testing.assert_allclose(result[f"G_full_m{int(result['final_m'])}"][:, 0],
                               expected, rtol=1e-10, atol=1e-10)
    assert operator.device.platform == platform

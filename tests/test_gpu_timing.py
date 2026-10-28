import importlib
from types import SimpleNamespace

import numpy as np
import pytest


MODULES = [importlib.import_module(f'examples.{model}.gpu_timing')
           for model in ('inchmoss', 'junction')]


@pytest.fixture(params=MODULES, ids=['inchmoss', 'junction'])
def timing(request):
    return request.param


@pytest.mark.parametrize('device', [None, 'cpu', 'gpu'])
def test_every_static_run_starts_from_zero(timing, device):
    calls = []

    def solver(problem, initial, **kwargs):
        assert kwargs['platform'] == (device or 'gpu')
        assert kwargs['production'] is True
        calls.append(None if initial is None else initial.copy())
        return SimpleNamespace(amplitudes=np.array([problem]), converged=True,
                               packed_residual_norm=1e-12, iterations=1)

    for _ in range(2):
        timing.solve_static(((.5, .5), (1., 1.)), None, solver=solver,
                            **({} if device is None else {'device': device}))
    assert calls[0] is None and calls[2] is None
    np.testing.assert_array_equal(calls[1], [.5])
    np.testing.assert_array_equal(calls[3], [.5])


@pytest.mark.parametrize('residual', [np.nan, np.inf, 1e-6])
def test_invalid_static_residual_is_rejected(timing, residual):
    def solver(problem, initial, **kwargs):
        return SimpleNamespace(amplitudes=np.ones(1), converged=True,
                               packed_residual_norm=residual, iterations=1)

    with pytest.raises(RuntimeError, match='Static solve failed'):
        timing.solve_static(((1., object()),), None, solver=solver)


def test_joint_response_matches_dense_solve(timing, tmp_path):
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    rng = np.random.default_rng(91)
    dimension = 8
    matrix = .1 * (rng.normal(size=(dimension, dimension)) + 1j*rng.normal(size=(dimension, dimension))) - 1j*np.eye(dimension)
    rhs = rng.normal(size=dimension) + 1j*rng.normal(size=dimension)
    sinks = rng.normal(size=(2, dimension)) + 1j*rng.normal(size=(2, dimension))
    device_matrix = jnp.asarray(matrix)
    operator = SimpleNamespace(device_matvec=lambda vector: device_matrix @ vector)
    shifts = np.array([-.4+.05j, .6+.05j, .7j, 1.3j])
    observed, metrics = timing.source_response(operator, rhs, sinks, shifts, 2, dimension, tmp_path / 'source.npz')
    expected = np.array([sinks @ np.linalg.solve(shift*np.eye(dimension)-matrix, rhs) for shift in shifts])
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)
    assert metrics['dimension'] == dimension
    assert metrics['converged']
    assert not list(tmp_path.glob('*.npy'))


def test_nonfinite_response_residual_is_rejected(timing, tmp_path):
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    dimension = 4
    matrix = jnp.diag(jnp.linspace(-1, 1, dimension) - 1j)
    calls = 0

    def action(vector):
        nonlocal calls
        calls += 1
        return matrix @ vector if calls <= dimension else jnp.full_like(vector, np.nan)

    with pytest.raises(FloatingPointError, match='Nonfinite response residual'):
        timing.source_response(SimpleNamespace(device_matvec=action), np.ones(dimension),
                               np.ones((1, dimension)), np.array([.1+.2j, 1j]), 1,
                               dimension, tmp_path / 'invalid.npz')


@pytest.mark.parametrize('changed', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_cold_warm_orchestration_reuses_only_matching_operators(timing, tmp_path, monkeypatch, changed, device):
    jax = pytest.importorskip('jax')
    preparations, solves, operators, responses = [], [], [], []
    policy = SimpleNamespace(to_dict=lambda: {})
    problem = object()
    amplitudes = np.array([.1])
    reference = tmp_path / 'reference.npz'
    np.savez(reference, amplitudes=amplitudes)
    selected_device = SimpleNamespace(platform=device, device_kind=f'test {device}',
                                      memory_stats=lambda: None if device == 'cpu' else {'bytes_in_use': 10})

    def devices(platform=None):
        assert platform == device
        return [selected_device]

    monkeypatch.setattr(jax, 'devices', devices)

    def prepare(args):
        preparations.append(args)
        return ((1., problem),), policy, 10.

    def solve_static(problems, policy, *, device='gpu'):
        assert device == selected_device.platform
        solves.append(problems)
        return amplitudes + (1e-12 if changed and len(solves) == 2 else 0), [dict(residual=1e-12)]

    def build_response(problem, values, rank, *, device='gpu'):
        assert device == selected_device.platform
        operators.append(values.copy())
        source = SimpleNamespace(addition=np.ones(1), removal=np.zeros(1), sink=np.ones(1))
        return SimpleNamespace(device=selected_device), (source, source)

    def source_response(operator, rhs, sinks, shifts, real_count, cap, output):
        responses.append((operator, output))
        return np.zeros((len(shifts), len(sinks)), complex), dict(converged=True)

    monkeypatch.setattr(timing, 'prepare', prepare)
    monkeypatch.setattr(timing, 'solve_static', solve_static)
    monkeypatch.setattr(timing, 'build_response', build_response)
    monkeypatch.setattr(timing, 'source_response', source_response)
    args = SimpleNamespace(rank='S13', bath=tmp_path / 'bath.npz', reference=reference,
                           output=tmp_path, device=device)
    record = timing.benchmark(args)
    assert record['device'] == selected_device.device_kind
    assert record['basis_storage'] == f'{device}_only'
    assert all(run['device_memory'] == selected_device.memory_stats() for run in record['runs'])
    assert len(preparations) == 1
    assert len(solves) == 2
    assert len(operators) == (2 if changed else 1)
    assert len(responses) == 4
    assert record['runs'][1]['operator_reused'] is not changed
    assert record['runs'][1]['label'] == 'warm'
    assert (tmp_path / 'warm/response.npz').exists()
    assert record['cold_total_seconds'] >= record['model_setup_seconds']
    assert record['warm_total_seconds'] == record['runs'][1]['wall_seconds']

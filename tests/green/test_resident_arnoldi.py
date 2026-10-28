import numpy as np
import pytest

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from c3l.green.resident_arnoldi import ResidentArnoldi, resident_kernels


def test_resident_recurrence_and_reconstruction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No disk-backed basis')

    monkeypatch.setattr(np.lib.format, 'open_memmap', forbidden)
    rng = np.random.default_rng(82)
    matrix = rng.normal(size=(24, 24)) + 1j * rng.normal(size=(24, 24))
    rhs = rng.normal(size=24) + 1j * rng.normal(size=24)
    sinks = rng.normal(size=(3, 24)) + 1j * rng.normal(size=(3, 24))
    device_matrix = jnp.asarray(matrix)
    basis = ResidentArnoldi(lambda vector: device_matrix @ vector, rhs, sinks, 16, panel=4)
    basis.advance(7)
    basis.advance(16)
    vectors = np.asarray(basis.basis[:17]).T
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(17), atol=1e-12)
    np.testing.assert_allclose(matrix @ vectors[:, :16], vectors @ basis.hessenberg[:17, :16], atol=1e-12)
    np.testing.assert_allclose(basis.projected[:, :17], sinks @ vectors, atol=1e-12)
    weights = rng.normal(size=16) + 1j * rng.normal(size=16)
    np.testing.assert_allclose(basis.reconstruct(weights), vectors[:, :16] @ weights, atol=1e-12)
    assert isinstance(basis.basis, jax.Array)


def test_invalid_basis_inputs():
    with pytest.raises(ValueError):
        ResidentArnoldi(lambda vector: vector, np.zeros(4), np.ones((1, 4)), 3)
    with pytest.raises(ValueError):
        ResidentArnoldi(lambda vector: vector, np.ones(4), np.ones((1, 3)), 3)


def test_insufficient_memory_fails_without_streaming(monkeypatch):
    from types import SimpleNamespace

    device = SimpleNamespace(memory_stats=lambda: {"bytes_limit": 1024, "bytes_in_use": 0})
    monkeypatch.setattr(jax, "devices", lambda: [device])
    with pytest.raises(MemoryError, match="No streaming fallback"):
        ResidentArnoldi(lambda vector: vector, np.ones(4), np.ones((1, 4)), 3)


def test_gpu_reconstruction_scratch_is_bounded():
    if jax.default_backend() != 'gpu':
        pytest.skip('GPU compiler memory regression')
    dimension, rows = 2470420, 1568
    reconstruct = resident_kernels(dimension, rows, 32)[2]
    executable = reconstruct.lower(
        jax.ShapeDtypeStruct((rows, dimension), jnp.complex128),
        jax.ShapeDtypeStruct((rows,), jnp.complex128),
        jax.ShapeDtypeStruct((), jnp.int32),
    ).compile()
    assert executable.memory_analysis().temp_size_in_bytes < 8 * 2**30


def test_public_s135_device_action_never_converts_through_numpy(monkeypatch):
    from types import SimpleNamespace
    from c3l.cc import s135_jax

    operator = object.__new__(s135_jax.JaxFullS135ResponseOperator)
    operator.jax = jax
    operator.jnp = jnp
    operator.device = jax.devices()[0]
    operator.dimension = 4
    operator._device_action = lambda vector: 3 * vector
    vector = jnp.arange(4, dtype=jnp.complex128)

    def forbidden(*args, **kwargs):
        raise AssertionError('Device action must not use numpy.asarray')

    monkeypatch.setattr(s135_jax, 'np', SimpleNamespace(asarray=forbidden))
    np.testing.assert_allclose(operator.device_matvec(vector), 3 * np.arange(4))

from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from c3l.green.jax_orthogonalization import (  # noqa: E402
    JaxPanelArnoldiOrthogonalizer,
)


def test_panel_cgs2_matches_literal_complex128_reference() -> None:
    coordinate = np.arange(55, dtype=np.float64).reshape(11, 5)
    basis = np.asfortranarray(
        np.sin(0.17 * coordinate + 0.3)
        + 1j * np.cos(0.11 * coordinate - 0.2),
        dtype=np.complex128,
    )
    axis = np.arange(11, dtype=np.float64)
    vector = np.asarray(
        np.cos(0.31 * axis + 0.1) + 1j * np.sin(0.23 * axis - 0.4),
        dtype=np.complex128,
    )
    expected = vector.copy()
    correction = np.zeros(basis.shape[1], dtype=np.complex128)
    for _ in range(2):
        delta = basis.conj().T @ expected
        correction += delta
        expected -= basis @ delta

    orthogonalizer = JaxPanelArnoldiOrthogonalizer(
        jax=jax,
        jnp=jnp,
        device=jax.devices("cpu")[0],
        vector_dimension=11,
        panel_columns=2,
        cache_columns=4,
    )
    observed = orthogonalizer.orthogonalize(basis, vector)

    np.testing.assert_allclose(
        observed.correction, correction, atol=2.0e-12, rtol=2.0e-12
    )
    np.testing.assert_allclose(
        observed.remainder, expected, atol=2.0e-12, rtol=2.0e-12
    )
    assert observed.remainder.flags.writeable
    assert observed.profile["panel_dispatches"] == 6.0


@pytest.mark.parametrize("width", [4, 5])
def test_complete_cached_panels_are_not_reuploaded(width):
    generator = np.random.default_rng(73)
    raw = generator.normal(size=(11, 6)) + 1j * generator.normal(size=(11, 6))
    basis = np.asfortranarray(np.linalg.qr(raw)[0][:, :width])
    vector = np.asarray(raw[:, 0] + 0.3j * raw[:, 1], dtype=np.complex128)
    orthogonalizer = JaxPanelArnoldiOrthogonalizer(
        jax=jax, jnp=jnp, device=jax.devices("cpu")[0],
        vector_dimension=11, panel_columns=2, cache_columns=6,
    )
    first = orthogonalizer.orthogonalize(basis, vector)
    second = orthogonalizer.orthogonalize(basis, vector)
    np.testing.assert_allclose(second.correction, first.correction, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(second.remainder, first.remainder, atol=2e-12, rtol=2e-12)
    assert second.profile["cache_fill_bytes"] == 0
    expected_tail = 11 * 2 * 16 if width % 2 else 0
    assert second.profile["stream_upload_bytes"] == expected_tail
    assert second.profile["transient_panel_reuse_bytes"] == expected_tail


def test_streaming_warning_once_per_basis(monkeypatch, caplog):
    basis = np.asfortranarray(np.eye(11, 6, dtype=np.complex128))
    vector = np.ones(11, dtype=np.complex128)
    orthogonalizer = JaxPanelArnoldiOrthogonalizer(
        jax=jax, jnp=jnp, device=jax.devices("cpu")[0],
        vector_dimension=11, panel_columns=2, cache_columns=2,
    )
    monkeypatch.setattr(orthogonalizer, "_platform", "gpu", raising=False)
    orthogonalizer.orthogonalize(basis[:, :3], vector)
    assert not caplog.records
    orthogonalizer.orthogonalize(basis, vector)
    orthogonalizer.orthogonalize(basis, vector)
    assert len(caplog.records) == 1
    assert "CPU memory and GPU memory" in caplog.text
    assert "cache_columns=2" in caplog.text
    orthogonalizer.orthogonalize(basis.copy(order="F"), vector)
    assert len(caplog.records) == 2


def test_explicit_budget_selects_cache_and_metadata():
    orthogonalizer = JaxPanelArnoldiOrthogonalizer(
        jax=jax, jnp=jnp, device=jax.devices("cpu")[0],
        vector_dimension=11, panel_columns=2, cache_columns=None,
        maximum_columns=5, basis_budget_bytes=11 * 4 * 16,
    )
    assert orthogonalizer.cache_columns == 4
    assert orthogonalizer.memory_metadata["mode"] == "streaming"
    assert orthogonalizer.memory_metadata["reason"] == "explicit_basis_budget"
    assert orthogonalizer.memory_metadata["required_bytes"] == 11 * 6 * 16


def test_cpu_execution_does_not_warn(caplog):
    orthogonalizer = JaxPanelArnoldiOrthogonalizer(
        jax=jax, jnp=jnp, device=jax.devices("cpu")[0],
        vector_dimension=11, panel_columns=2, cache_columns=0,
    )
    orthogonalizer.orthogonalize(np.eye(11, 4, dtype=np.complex128), np.ones(11, dtype=np.complex128))
    assert not caplog.records


def test_shifted_views_do_not_share_cached_columns():
    basis = np.asfortranarray(np.eye(11, 8, dtype=np.complex128))
    vector = np.arange(11, dtype=np.float64).astype(np.complex128)
    operator = JaxPanelArnoldiOrthogonalizer(jax=jax, jnp=jnp, device=jax.devices("cpu")[0], vector_dimension=11, panel_columns=2, cache_columns=4)
    operator.orthogonalize(basis[:, :4], vector)
    result = operator.orthogonalize(basis[:, 4:], vector)
    np.testing.assert_allclose(result.correction, basis[:, 4:].conj().T @ vector, atol=2e-12)


def test_reset_invalidates_same_buffer_and_reopened_memmap(tmp_path):
    path = tmp_path / "basis.npy"
    basis = np.lib.format.open_memmap(path, mode="w+", dtype=np.complex128, shape=(11, 4), fortran_order=True)
    basis[:] = np.eye(11, 4)
    vector = np.arange(11, dtype=np.float64).astype(np.complex128)
    operator = JaxPanelArnoldiOrthogonalizer(jax=jax, jnp=jnp, device=jax.devices("cpu")[0], vector_dimension=11, panel_columns=2, cache_columns=4)
    operator.orthogonalize(basis, vector)
    basis[:] *= 1j
    basis.flush()
    operator.reset()
    result = operator.orthogonalize(basis, vector)
    np.testing.assert_allclose(result.correction, basis.conj().T @ vector, atol=2e-12)
    basis[:] *= 1j
    basis.flush()
    reopened = np.load(path, mmap_mode="r")
    result = operator.orthogonalize(reopened, vector)
    np.testing.assert_allclose(result.correction, reopened.conj().T @ vector, atol=2e-12)


def test_budget_rejects_basis_exceeding_declared_maximum():
    operator = JaxPanelArnoldiOrthogonalizer(jax=jax, jnp=jnp, device=jax.devices("cpu")[0], vector_dimension=11, panel_columns=2, cache_columns=None, maximum_columns=2, basis_budget_bytes=11 * 2 * 16)
    with pytest.raises(ValueError, match="maximum_columns"):
        operator.orthogonalize(np.eye(11, 4, dtype=np.complex128), np.ones(11, dtype=np.complex128))


@pytest.mark.parametrize("explicit_reset", [False, True])
def test_shorter_partial_tail_does_not_retain_old_columns(explicit_reset):
    basis = np.asfortranarray(np.eye(8, 4, dtype=np.complex128))
    vector = np.arange(1, 9, dtype=np.float64).astype(np.complex128)
    operator = JaxPanelArnoldiOrthogonalizer(jax=jax, jnp=jnp, device=jax.devices("cpu")[0], vector_dimension=8, panel_columns=4, cache_columns=4)
    operator.orthogonalize(basis[:, :3], vector)
    if explicit_reset:
        operator.reset()
    result = operator.orthogonalize(basis[:, :1], vector)
    expected = vector.copy()
    expected[0] = 0
    np.testing.assert_allclose(result.remainder, expected, atol=2e-12)

"""Bounded diagnostics for large disk-backed Arnoldi bases."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def _sampled_columns(total: int, count: int) -> np.ndarray:
    keep = min(total, count)
    selected = np.unique(
        np.rint(np.linspace(0, total - 1, keep)).astype(np.int64)
    )
    if selected.size < keep:
        missing = np.setdiff1d(
            np.arange(total, dtype=np.int64), selected, assume_unique=True
        )[: keep - selected.size]
        selected = np.sort(np.concatenate((selected, missing)))
    return selected


def orthogonality_diagnostics(
    basis: ArrayLike,
    *,
    dimension: int,
    exact_limit: int = 256,
    sample_columns: int = 128,
    row_block_size: int = 65536,
    accelerator: Any | None = None,
) -> dict[str, Any]:
    """Stream an exact or deterministic sampled complex Gram matrix."""

    values = np.asanyarray(basis)
    if values.ndim != 2 or values.dtype != np.complex128:
        raise ValueError("Arnoldi basis must be a two-dimensional complex128 array")
    if not 0 <= dimension < values.shape[1]:
        raise ValueError("Arnoldi diagnostic dimension lies outside the basis")
    if exact_limit <= 0 or sample_columns <= 0 or row_block_size <= 0:
        raise ValueError("orthogonality diagnostic bounds must be positive")
    total = dimension + 1
    exact = dimension <= exact_limit
    columns = (
        np.arange(total, dtype=np.int64)
        if exact
        else _sampled_columns(total, sample_columns)
    )
    started = perf_counter()

    if accelerator is None:
        gram = np.zeros((columns.size, columns.size), dtype=np.complex128)
        for start in range(0, values.shape[0], row_block_size):
            stop = min(start + row_block_size, values.shape[0])
            block = np.asarray(values[start:stop, columns], dtype=np.complex128)
            gram += block.conj().T @ block
    else:
        jax = accelerator.jax
        jnp = accelerator.jnp
        device = accelerator.device
        gram_device = jax.device_put(
            np.zeros((columns.size, columns.size), dtype=np.complex128), device
        )
        for start in range(0, values.shape[0], row_block_size):
            stop = min(start + row_block_size, values.shape[0])
            host = np.ascontiguousarray(values[start:stop, columns])
            block = jax.device_put(host, device)
            gram_device = gram_device + jnp.conj(block).T @ block
        gram_device.block_until_ready()
        gram = np.asarray(gram_device, dtype=np.complex128)

    defect = gram - np.eye(columns.size, dtype=np.complex128)
    diagonal = np.real(np.diag(gram))
    off_diagonal = defect.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    return {
        "mode": "exact" if exact else "sampled",
        "dimension": int(dimension),
        "column_count": int(columns.size),
        "columns": [int(value) for value in columns],
        "frobenius_defect": float(np.linalg.norm(defect)),
        "maximum_absolute_defect": float(np.max(np.abs(defect))),
        "maximum_absolute_off_diagonal": float(np.max(np.abs(off_diagonal))),
        "minimum_vector_norm": float(np.sqrt(max(float(np.min(diagonal)), 0.0))),
        "maximum_vector_norm": float(np.sqrt(max(float(np.max(diagonal)), 0.0))),
        "wall_seconds": float(perf_counter() - started),
        "row_block_size": int(row_block_size),
        "accelerated": accelerator is not None,
    }


__all__ = ["orthogonality_diagnostics"]

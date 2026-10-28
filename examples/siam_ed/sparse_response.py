"""Explicit sparse sentinel oracle for the printed odd-sector generator.

This module materializes the projected coefficient matrix ``K = i L_resp``
only after a deterministic storage preflight.  Sparse LU is intended for a
small set of sentinel frequencies.  Every reported solve residual is
recomputed from ``(z I - K) x - b`` and divided by ``||b||_2``.
"""

from __future__ import annotations

from dataclasses import dataclass
import resource
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu


ComplexArray = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
RealArray = NDArray[np.float64]


class ColumnResponseAction(Protocol):
    """Projected odd-sector action with deterministic coefficient columns."""

    @property
    def dimension(self) -> int: ...

    def column_entries(self, column: int) -> tuple[tuple[int, complex], ...]: ...

    def matvec(self, vector: ArrayLike) -> ComplexVector: ...


class ResponseAction(Protocol):
    @property
    def dimension(self) -> int: ...

    def matvec(self, vector: ArrayLike) -> ComplexVector: ...


def _storage_bytes(matrix: csc_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _source_matrix(value: ArrayLike, dimension: int) -> ComplexArray:
    source = np.asarray(value, dtype=np.complex128)
    if source.ndim == 1:
        source = source[:, None]
    if (
        source.ndim != 2
        or source.shape[0] != dimension
        or source.shape[1] == 0
        or not np.all(np.isfinite(source))
    ):
        raise ValueError("rhs must contain finite columns matching the operator")
    norms = np.linalg.norm(source, axis=0)
    if np.any(norms == 0.0):
        raise ValueError("every rhs column must be nonzero")
    return source


@dataclass(frozen=True, slots=True)
class SparseResponsePreflight:
    dimension: int
    sampled_columns: tuple[int, ...]
    sampled_mean_nnz: np.float64
    sampled_maximum_nnz: int
    estimated_nnz: int
    estimated_csc_bytes: int
    maximum_bytes: int
    allowed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "sampled_columns": list(self.sampled_columns),
            "sampled_mean_nnz": float(self.sampled_mean_nnz),
            "sampled_maximum_nnz": self.sampled_maximum_nnz,
            "estimated_nnz": self.estimated_nnz,
            "estimated_csc_bytes": self.estimated_csc_bytes,
            "maximum_bytes": self.maximum_bytes,
            "allowed": self.allowed,
        }


@dataclass(frozen=True, slots=True)
class ExplicitSparseResponseOperator:
    r"""CSC representation of $K=i\mathcal L_{\mathrm{resp}}$."""

    matrix: csc_matrix
    preflight: SparseResponsePreflight
    nnz: int
    storage_bytes: int
    assembly_wall_seconds: np.float64
    action_parity_defect: np.float64
    peak_rss_kib: int

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, csc_matrix):
            raise TypeError("response matrix must use CSC storage")
        if self.matrix.shape != (self.preflight.dimension,) * 2:
            raise ValueError("sparse response matrix and preflight disagree")
        if self.nnz != self.matrix.nnz or self.storage_bytes != _storage_bytes(
            self.matrix
        ):
            raise ValueError("sparse response storage metadata is inconsistent")

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    def matvec(self, vector: ArrayLike) -> ComplexVector:
        x = np.asarray(vector, dtype=np.complex128)
        if x.shape != (self.dimension,) or not np.all(np.isfinite(x)):
            raise ValueError("input must be a finite odd-sector vector")
        return np.asarray(self.matrix @ x, dtype=np.complex128)

    def rmatvec(self, vector: ArrayLike) -> ComplexVector:
        x = np.asarray(vector, dtype=np.complex128)
        if x.shape != (self.dimension,) or not np.all(np.isfinite(x)):
            raise ValueError("adjoint input must be a finite odd-sector vector")
        return np.asarray(self.matrix.conj().T @ x, dtype=np.complex128)

    def column_entries(self, column: int) -> tuple[tuple[int, complex], ...]:
        if isinstance(column, bool) or not isinstance(column, int):
            raise TypeError("column must be an integer")
        if not 0 <= column < self.dimension:
            raise ValueError("column is outside the response space")
        start = self.matrix.indptr[column]
        stop = self.matrix.indptr[column + 1]
        return tuple(
            (int(row), complex(value))
            for row, value in zip(
                self.matrix.indices[start:stop],
                self.matrix.data[start:stop],
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "generator_convention": "K=i*L_response",
            "dimension": self.dimension,
            "nnz": self.nnz,
            "storage_bytes": self.storage_bytes,
            "assembly_wall_seconds": float(self.assembly_wall_seconds),
            "action_parity_defect": float(self.action_parity_defect),
            "peak_rss_kib": self.peak_rss_kib,
            "preflight": self.preflight.to_dict(),
        }


def sparse_response_preflight(
    operator: ColumnResponseAction,
    *,
    maximum_bytes: int,
    sample_count: int = 32,
    safety_factor: float = 2.0,
) -> SparseResponsePreflight:
    """Estimate CSC storage from deterministic columns before assembly."""

    if not all(hasattr(operator, name) for name in ("dimension", "column_entries")):
        raise TypeError("operator must expose dimension and column_entries")
    dimension = int(operator.dimension)
    if dimension <= 0:
        raise ValueError("operator dimension must be positive")
    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes <= 0
    ):
        raise ValueError("maximum_bytes must be a positive integer")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise ValueError("sample_count must be a positive integer")
    safety = float(safety_factor)
    if not np.isfinite(safety) or safety < 1.0:
        raise ValueError("safety_factor must be finite and at least one")
    columns = tuple(
        int(value)
        for value in np.linspace(
            0,
            dimension - 1,
            min(sample_count, dimension),
            dtype=np.int64,
        )
    )
    counts = np.asarray(
        [len(operator.column_entries(column)) for column in columns],
        dtype=np.float64,
    )
    estimated_nnz = min(
        dimension * dimension,
        int(np.ceil(dimension * float(np.mean(counts)) * safety)),
    )
    # SciPy uses complex128 values and int32 row indices for these dimensions.
    estimated_bytes = (
        estimated_nnz
        * (np.dtype(np.complex128).itemsize + np.dtype(np.int32).itemsize)
        + (dimension + 1) * np.dtype(np.int64).itemsize
    )
    return SparseResponsePreflight(
        dimension=dimension,
        sampled_columns=columns,
        sampled_mean_nnz=np.float64(np.mean(counts)),
        sampled_maximum_nnz=int(np.max(counts)),
        estimated_nnz=estimated_nnz,
        estimated_csc_bytes=estimated_bytes,
        maximum_bytes=maximum_bytes,
        allowed=bool(estimated_bytes <= maximum_bytes),
    )


def compile_sparse_response_operator(
    operator: ColumnResponseAction,
    *,
    maximum_bytes: int | None = None,
    sample_count: int = 32,
    safety_factor: float = 2.0,
) -> ExplicitSparseResponseOperator:
    """Assemble deterministic CSC columns in two bounded passes."""

    if not all(
        hasattr(operator, name)
        for name in ("dimension", "column_entries", "matvec")
    ):
        raise TypeError("operator must expose dimension, column_entries, and matvec")
    default_limit = getattr(
        getattr(operator, "policy", None),
        "dense_memory_limit_bytes",
        1 << 30,
    )
    limit = int(default_limit if maximum_bytes is None else maximum_bytes)
    preflight = sparse_response_preflight(
        operator,
        maximum_bytes=limit,
        sample_count=sample_count,
        safety_factor=safety_factor,
    )
    if not preflight.allowed:
        raise MemoryError(
            "sampled sparse response estimate requires "
            f"{preflight.estimated_csc_bytes} bytes, above the declared "
            f"{preflight.maximum_bytes}-byte limit"
        )

    started = perf_counter()
    dimension = int(operator.dimension)
    indptr = np.zeros(dimension + 1, dtype=np.int64)
    for column in range(dimension):
        entries = operator.column_entries(column)
        indptr[column + 1] = indptr[column] + len(entries)
        compact_bytes = (
            int(indptr[column + 1])
            * (np.dtype(np.complex128).itemsize + np.dtype(np.int32).itemsize)
            + indptr.nbytes
        )
        if compact_bytes > limit:
            raise MemoryError(
                "exact sparse response column count exceeds the declared "
                f"{limit}-byte limit"
            )

    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.complex128)
    for column in range(dimension):
        entries = operator.column_entries(column)
        start = int(indptr[column])
        stop = int(indptr[column + 1])
        if len(entries) != stop - start:
            raise RuntimeError("projected response columns are not deterministic")
        previous = -1
        for offset, (row, value) in enumerate(entries, start=start):
            if not 0 <= row < dimension or row <= previous:
                raise ValueError("column entries must have unique sorted row indices")
            if not np.isfinite(value.real) or not np.isfinite(value.imag):
                raise ValueError("column entries must be finite")
            indices[offset] = row
            data[offset] = value
            previous = row
    matrix = csc_matrix((data, indices, indptr), shape=(dimension, dimension))
    matrix.eliminate_zeros()
    matrix.sort_indices()

    # One deterministic dense probe audits destination/source orientation.
    phase = np.arange(1, dimension + 1, dtype=np.float64)
    probe = np.cos(phase) + 1j * np.sin(phase)
    expected = np.asarray(operator.matvec(probe), dtype=np.complex128)
    observed = np.asarray(matrix @ probe, dtype=np.complex128)
    scale = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    parity = np.float64(np.linalg.norm(observed - expected) / scale)
    return ExplicitSparseResponseOperator(
        matrix=matrix,
        preflight=preflight,
        nnz=int(matrix.nnz),
        storage_bytes=_storage_bytes(matrix),
        assembly_wall_seconds=np.float64(perf_counter() - started),
        action_parity_defect=parity,
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


@dataclass(frozen=True, slots=True)
class SparseDirectResolvent:
    r"""Sparse-LU oracle for a bounded set of $(zI-K)x=b$ systems."""

    shifts: ComplexVector
    rhs: ComplexArray
    solutions: ComplexArray
    relative_residuals: RealArray
    factorization_wall_seconds: RealArray
    solve_wall_seconds: RealArray
    factor_nnz: tuple[int, ...]
    peak_rss_kib: int

    @property
    def source_count(self) -> int:
        return int(self.rhs.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.rhs.shape[0])

    @property
    def maximum_relative_residual(self) -> np.float64:
        return np.float64(np.max(self.relative_residuals))

    def transfer(self, sinks: ArrayLike) -> ComplexArray:
        """Contract algebraic sink rows with every source solution."""

        rows = np.asarray(sinks, dtype=np.complex128)
        if rows.ndim == 1:
            rows = rows[None, :]
        if (
            rows.ndim != 2
            or rows.shape[1] != self.dimension
            or not np.all(np.isfinite(rows))
        ):
            raise ValueError("sinks must contain finite algebraic rows")
        return np.asarray(
            np.einsum("ai,zib->zab", rows, self.solutions),
            dtype=np.complex128,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "sparse_direct_resolvent_sentinel_oracle",
            "generator_convention": "K=i*L_response",
            "dimension": self.dimension,
            "source_count": self.source_count,
            "shifts": {
                "real": self.shifts.real.tolist(),
                "imag": self.shifts.imag.tolist(),
            },
            "relative_residuals": self.relative_residuals.tolist(),
            "maximum_relative_residual": float(self.maximum_relative_residual),
            "factorization_wall_seconds": [
                float(x) for x in self.factorization_wall_seconds
            ],
            "solve_wall_seconds": [float(x) for x in self.solve_wall_seconds],
            "factor_nnz": list(self.factor_nnz),
            "peak_rss_kib": self.peak_rss_kib,
            "residual_definition": "norm((z*I-K)*x-b)_2/norm(b)_2",
            "sink_contraction": "algebraic_row_without_conjugation",
        }


def sparse_direct_resolvent(
    operator: ExplicitSparseResponseOperator | ResponseAction,
    rhs: ArrayLike,
    shifts: ArrayLike,
    *,
    verification_operator: ResponseAction | None = None,
    permutation: str = "COLAMD",
) -> SparseDirectResolvent:
    r"""Factor and solve a small sentinel set, then recompute true residuals."""

    if not all(hasattr(operator, name) for name in ("matrix", "dimension", "matvec")):
        raise TypeError("operator must expose a CSC matrix, dimension, and matvec")
    if not isinstance(operator.matrix, csc_matrix):
        raise TypeError("operator.matrix must use CSC storage")
    if permutation not in ("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"):
        raise ValueError("unknown SuperLU permutation")
    z = np.asarray(shifts, dtype=np.complex128)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("shifts must be a finite nonempty vector")
    B = _source_matrix(rhs, operator.dimension)
    verifier = operator if verification_operator is None else verification_operator
    if int(verifier.dimension) != operator.dimension:
        raise ValueError("verification operator dimension differs from sparse K")

    solutions = np.empty(
        (z.size, operator.dimension, B.shape[1]), dtype=np.complex128
    )
    residuals = np.empty((z.size, B.shape[1]), dtype=np.float64)
    factor_seconds = np.empty(z.size, dtype=np.float64)
    solve_seconds = np.empty(z.size, dtype=np.float64)
    factor_nnz: list[int] = []
    identity = eye(operator.dimension, dtype=np.complex128, format="csc")
    source_norms = np.linalg.norm(B, axis=0)
    for row, shift in enumerate(z):
        shifted = shift * identity - operator.matrix
        factor_started = perf_counter()
        factor = splu(shifted, permc_spec=permutation)
        factor_seconds[row] = perf_counter() - factor_started
        solve_started = perf_counter()
        solutions[row] = factor.solve(B)
        solve_seconds[row] = perf_counter() - solve_started
        factor_nnz.append(int(factor.L.nnz + factor.U.nnz))
        for source in range(B.shape[1]):
            x = solutions[row, :, source]
            residual = shift * x - verifier.matvec(x) - B[:, source]
            residuals[row, source] = np.linalg.norm(residual) / source_norms[source]
    return SparseDirectResolvent(
        shifts=z.copy(),
        rhs=B.copy(),
        solutions=solutions,
        relative_residuals=residuals,
        factorization_wall_seconds=factor_seconds,
        solve_wall_seconds=solve_seconds,
        factor_nnz=tuple(factor_nnz),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


__all__ = [
    "ColumnResponseAction",
    "ExplicitSparseResponseOperator",
    "ResponseAction",
    "SparseDirectResolvent",
    "SparseResponsePreflight",
    "compile_sparse_response_operator",
    "sparse_direct_resolvent",
    "sparse_response_preflight",
]

"""Sparse physical-basis oracle in conserved matrix-unit sectors.

The complete physical Liouville space has dimension ``4**(2*n)`` for an
``n``-site spinful auxiliary model.  Neither the static finite-system oracle
nor fermionic regression needs that full matrix.  The indexed-Gamma generator
preserves the two matrix-unit charges

``Q_sigma = N_sigma(row) - N_sigma(column)``.

This module therefore works directly with deterministic column-major matrix
units ``|row><column|``.  Static states use ``(Q_up,Q_down)=(0,0)``.  A fixed
spin response uses one of ``(+/-1,0)`` or ``(0,+/-1)``.  In an odd response
sector every Lindblad recycling term changes sign; Hamiltonian and
anticommutator terms do not.  The response action exposed here is the printed
coefficient generator ``K = i L_response``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
import resource
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from .sparse_response import (
    SparseResponsePreflight,
    sparse_response_preflight,
)
from c3l.models import FermionicLindbladModel


ComplexVector = NDArray[np.complex128]
IntVector = NDArray[np.int64]


def _immutable(a: ArrayLike, dtype: np.dtype) -> NDArray:
    value = np.ascontiguousarray(a, dtype=dtype)
    out = np.frombuffer(value.tobytes(), dtype=dtype).reshape(value.shape)
    out.flags.writeable = False
    return out


def _charge_counts(n: int) -> tuple[IntVector, IntVector]:
    d = 1 << (2 * n)
    up_bits = sum(1 << (2 * i) for i in range(n))
    down_bits = sum(1 << (2 * i + 1) for i in range(n))
    up = np.fromiter(
        ((mask & up_bits).bit_count() for mask in range(d)),
        dtype=np.int64,
        count=d,
    )
    down = np.fromiter(
        ((mask & down_bits).bit_count() for mask in range(d)),
        dtype=np.int64,
        count=d,
    )
    return up, down


@dataclass(frozen=True, slots=True)
class PhysicalMatrixUnitSector:
    """Column-major ``|row><column|`` basis at fixed spin charges."""

    n_sites: int
    q_up: int
    q_down: int
    rows: IntVector
    columns: IntVector
    global_indices: IntVector
    index_of_global: IntVector

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_sites, bool)
            or not isinstance(self.n_sites, int)
            or self.n_sites < 1
        ):
            raise ValueError("n_sites must be a positive integer")
        for q in (self.q_up, self.q_down):
            if isinstance(q, bool) or not isinstance(q, int) or abs(q) > self.n_sites:
                raise ValueError("spin charges must be integers in [-n_sites,n_sites]")
        rows = np.asarray(self.rows)
        columns = np.asarray(self.columns)
        indices = np.asarray(self.global_indices)
        inverse = np.asarray(self.index_of_global)
        d = 1 << (2 * self.n_sites)
        expected = comb(2 * self.n_sites, self.n_sites + self.q_up) * comb(
            2 * self.n_sites, self.n_sites + self.q_down
        )
        if rows.shape != (expected,) or columns.shape != (expected,):
            raise ValueError("matrix-unit arrays have the wrong sector dimension")
        if indices.shape != (expected,) or inverse.shape != (d * d,):
            raise ValueError("matrix-unit index arrays have incompatible shapes")
        if not np.array_equal(indices, rows + d * columns):
            raise ValueError("global indices do not use column-major vectorization")
        if expected and not np.array_equal(
            inverse[indices], np.arange(expected, dtype=np.int64)
        ):
            raise ValueError("global-to-sector map is inconsistent")
        object.__setattr__(self, "rows", _immutable(rows, np.dtype(np.int64)))
        object.__setattr__(
            self, "columns", _immutable(columns, np.dtype(np.int64))
        )
        object.__setattr__(
            self, "global_indices", _immutable(indices, np.dtype(np.int64))
        )
        object.__setattr__(
            self, "index_of_global", _immutable(inverse, np.dtype(np.int64))
        )

    @property
    def dimension(self) -> int:
        return int(self.rows.size)

    @property
    def fock_dimension(self) -> int:
        return 1 << (2 * self.n_sites)

    def index(self, row: int, column: int) -> int:
        if not 0 <= row < self.fock_dimension or not 0 <= column < self.fock_dimension:
            raise ValueError("matrix-unit mask is outside the physical Fock space")
        return int(self.index_of_global[row + self.fock_dimension * column])


def physical_matrix_unit_sector(
    n_sites: int, q_up: int = 0, q_down: int = 0
) -> PhysicalMatrixUnitSector:
    """Construct one complete fixed-charge sector without a full supermatrix."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, int) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    for q in (q_up, q_down):
        if isinstance(q, bool) or not isinstance(q, int) or abs(q) > n_sites:
            raise ValueError("spin charges must be integers in [-n_sites,n_sites]")
    d = 1 << (2 * n_sites)
    n_up, n_down = _charge_counts(n_sites)
    row_parts: list[NDArray[np.int64]] = []
    column_parts: list[NDArray[np.int64]] = []
    for column in range(d):
        rows = np.flatnonzero(
            (n_up - n_up[column] == q_up)
            & (n_down - n_down[column] == q_down)
        ).astype(np.int64, copy=False)
        if rows.size:
            row_parts.append(rows)
            column_parts.append(np.full(rows.size, column, dtype=np.int64))
    row_masks = np.concatenate(row_parts)
    column_masks = np.concatenate(column_parts)
    global_indices = row_masks + d * column_masks
    index_of_global = np.full(d * d, -1, dtype=np.int64)
    index_of_global[global_indices] = np.arange(global_indices.size, dtype=np.int64)
    return PhysicalMatrixUnitSector(
        n_sites=n_sites,
        q_up=q_up,
        q_down=q_down,
        rows=row_masks,
        columns=column_masks,
        global_indices=global_indices,
        index_of_global=index_of_global,
    )


def _annihilate(mask: int, p: int) -> tuple[int, int] | None:
    bit = 1 << p
    if not mask & bit:
        return None
    sign = -1 if (mask & (bit - 1)).bit_count() % 2 else 1
    return mask ^ bit, sign


def _create(mask: int, p: int) -> tuple[int, int] | None:
    bit = 1 << p
    if mask & bit:
        return None
    sign = -1 if (mask & (bit - 1)).bit_count() % 2 else 1
    return mask | bit, sign


def _spin_entries(a: NDArray[np.complex128]) -> tuple[tuple[int, int, complex], ...]:
    entries: list[tuple[int, int, complex]] = []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] != 0.0:
                for spin in (0, 1):
                    entries.append((2 * i + spin, 2 * j + spin, complex(a[i, j])))
    return tuple(entries)


def _one_body_actions(
    n_sites: int,
    entries: tuple[tuple[int, int, complex], ...],
    *,
    constant: complex = 0.0,
    impurity: int | None = None,
    U: float = 0.0,
) -> tuple[tuple[tuple[int, complex], ...], ...]:
    """Tabulate ``constant + sum_ij a_ij c_i^dagger c_j`` on Fock kets."""

    d = 1 << (2 * n_sites)
    actions: list[tuple[tuple[int, complex], ...]] = []
    for mask in range(d):
        out: dict[int, complex] = {}
        if constant != 0.0:
            out[mask] = complex(constant)
        if impurity is not None:
            up = 1 << (2 * impurity)
            down = 1 << (2 * impurity + 1)
            if mask & up and mask & down:
                out[mask] = out.get(mask, 0.0j) + U
        for i, j, value in entries:
            first = _annihilate(mask, j)
            if first is None:
                continue
            middle, sign_j = first
            second = _create(middle, i)
            if second is None:
                continue
            target, sign_i = second
            out[target] = out.get(target, 0.0j) + value * sign_i * sign_j
        actions.append(
            tuple(sorted((target, value) for target, value in out.items() if value))
        )
    return tuple(actions)


class IndexedGammaSectorAction:
    """Matrix-free indexed-Gamma action in one physical matrix-unit sector."""

    def __init__(
        self,
        model: FermionicLindbladModel,
        sector: PhysicalMatrixUnitSector,
        *,
        odd_response: bool,
    ) -> None:
        if not isinstance(model, FermionicLindbladModel):
            raise TypeError("model must be a FermionicLindbladModel")
        if sector.n_sites != model.n_sites:
            raise ValueError("sector and model have different site counts")
        q = (sector.q_up, sector.q_down)
        if odd_response:
            if q not in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                raise ValueError("odd response requires one fixed-spin charge +/-1")
        elif q != (0, 0):
            raise ValueError("the density generator requires the Q=(0,0) sector")
        self.model = model
        self.sector = sector
        self.odd_response = bool(odd_response)
        self.dimension = sector.dimension
        self.generator_convention = (
            "K=i*L_response" if odd_response else "L_density"
        )

        E = np.asarray(model.E, dtype=np.complex128)
        loss = np.asarray(model.gamma_loss, dtype=np.complex128)
        gain = np.asarray(model.gamma_gain, dtype=np.complex128)
        self._H = _one_body_actions(
            model.n_sites,
            _spin_entries(E),
            impurity=model.impurity_site,
            U=float(model.U),
        )
        self._loss_number = _one_body_actions(
            model.n_sites, _spin_entries(loss)
        )
        self._gain_number = _one_body_actions(
            model.n_sites,
            tuple((i, j, -value) for i, j, value in _spin_entries(gain)),
            constant=2.0 * np.trace(gain),
        )
        self._loss_recycling = _spin_entries(loss)
        self._gain_recycling = _spin_entries(gain)

    def _add(
        self,
        out: dict[int, complex],
        row: int,
        column: int,
        value: complex,
    ) -> None:
        if value == 0.0:
            return
        target = self.sector.index(row, column)
        if target < 0:
            raise RuntimeError("indexed-Gamma action left its conserved sector")
        out[target] = out.get(target, 0.0j) + value

    def column_entries(self, column: int) -> tuple[tuple[int, complex], ...]:
        if isinstance(column, bool) or not isinstance(column, int):
            raise TypeError("column must be an integer")
        if not 0 <= column < self.dimension:
            raise ValueError("column is outside the physical sector")
        row_mask = int(self.sector.rows[column])
        column_mask = int(self.sector.columns[column])
        out: dict[int, complex] = {}

        # -i[H,|r><c|].  The right action is the conjugate ket action because
        # H is Hermitian.
        for target, value in self._H[row_mask]:
            self._add(out, target, column_mask, -1j * value)
        for target, value in self._H[column_mask]:
            self._add(out, row_mask, target, 1j * value.conjugate())

        # -{M_loss+M_gain,|r><c|} with Hermitian number operators.
        for actions in (self._loss_number, self._gain_number):
            for target, value in actions[row_mask]:
                self._add(out, target, column_mask, -value)
            for target, value in actions[column_mask]:
                self._add(out, row_mask, target, -value.conjugate())

        # Recycling acts on both matrix-unit indices.  Odd fermionic
        # regression changes the sign of every recycling contribution.
        recycling_sign = -2.0 if self.odd_response else 2.0
        for i, j, value in self._loss_recycling:
            left = _annihilate(row_mask, j)
            right = _annihilate(column_mask, i)
            if left is not None and right is not None:
                target_row, sign_row = left
                target_column, sign_column = right
                self._add(
                    out,
                    target_row,
                    target_column,
                    recycling_sign * value * sign_row * sign_column,
                )
        for i, j, value in self._gain_recycling:
            left = _create(row_mask, i)
            right = _create(column_mask, j)
            if left is not None and right is not None:
                target_row, sign_row = left
                target_column, sign_column = right
                self._add(
                    out,
                    target_row,
                    target_column,
                    recycling_sign * value * sign_row * sign_column,
                )

        multiplier = 1j if self.odd_response else 1.0
        return tuple(
            sorted(
                (row, multiplier * value)
                for row, value in out.items()
                if value != 0.0
            )
        )

    def matvec(self, vector: ArrayLike) -> ComplexVector:
        x = np.asarray(vector, dtype=np.complex128)
        if x.shape != (self.dimension,) or not np.all(np.isfinite(x)):
            raise ValueError("input must be a finite sector vector")
        y = np.zeros(self.dimension, dtype=np.complex128)
        for column in np.flatnonzero(x):
            value = x[column]
            for row, coefficient in self.column_entries(int(column)):
                y[row] += coefficient * value
        return y

    def rmatvec(self, vector: ArrayLike) -> ComplexVector:
        x = np.asarray(vector, dtype=np.complex128)
        if x.shape != (self.dimension,) or not np.all(np.isfinite(x)):
            raise ValueError("input must be a finite sector vector")
        y = np.empty(self.dimension, dtype=np.complex128)
        for column in range(self.dimension):
            y[column] = sum(
                coefficient.conjugate() * x[row]
                for row, coefficient in self.column_entries(column)
            )
        return y


def physical_density_action(
    model: FermionicLindbladModel,
) -> IndexedGammaSectorAction:
    return IndexedGammaSectorAction(
        model,
        physical_matrix_unit_sector(model.n_sites),
        odd_response=False,
    )


def physical_response_action(
    model: FermionicLindbladModel, *, spin: int = 0, charge: int = 1
) -> IndexedGammaSectorAction:
    if spin not in (0, 1):
        raise ValueError("spin must be zero (up) or one (down)")
    if charge not in (-1, 1):
        raise ValueError("charge must be -1 or +1")
    q_up, q_down = ((charge, 0) if spin == 0 else (0, charge))
    return IndexedGammaSectorAction(
        model,
        physical_matrix_unit_sector(model.n_sites, q_up, q_down),
        odd_response=True,
    )


@dataclass(frozen=True, slots=True)
class ExplicitPhysicalSectorOperator:
    """Explicit CSC matrix assembled from a physical sector action."""

    matrix: csc_matrix
    action: IndexedGammaSectorAction
    preflight: SparseResponsePreflight
    assembly_wall_seconds: np.float64
    action_parity_defect: np.float64
    peak_rss_kib: int

    @property
    def dimension(self) -> int:
        return self.action.dimension

    @property
    def storage_bytes(self) -> int:
        return int(
            self.matrix.data.nbytes
            + self.matrix.indices.nbytes
            + self.matrix.indptr.nbytes
        )

    def matvec(self, vector: ArrayLike) -> ComplexVector:
        x = np.asarray(vector, dtype=np.complex128)
        if x.shape != (self.dimension,):
            raise ValueError("input has the wrong sector dimension")
        return np.asarray(self.matrix @ x, dtype=np.complex128)


def compile_physical_sector_operator(
    action: IndexedGammaSectorAction,
    *,
    maximum_bytes: int = 1 << 30,
    sample_count: int = 32,
    safety_factor: float = 2.0,
) -> ExplicitPhysicalSectorOperator:
    """Assemble a CSC sector matrix after a sampled storage preflight."""

    if not isinstance(action, IndexedGammaSectorAction):
        raise TypeError("action must be an IndexedGammaSectorAction")
    preflight = sparse_response_preflight(
        action,
        maximum_bytes=maximum_bytes,
        sample_count=sample_count,
        safety_factor=safety_factor,
    )
    if not preflight.allowed:
        raise MemoryError(
            "sampled physical-sector estimate requires "
            f"{preflight.estimated_csc_bytes} bytes, above the declared "
            f"{preflight.maximum_bytes}-byte limit"
        )
    started = perf_counter()
    n = action.dimension
    indptr = np.zeros(n + 1, dtype=np.int64)
    for column in range(n):
        indptr[column + 1] = indptr[column] + len(action.column_entries(column))
        required = int(indptr[column + 1]) * 20 + indptr.nbytes
        if required > maximum_bytes:
            raise MemoryError("exact physical-sector CSC count exceeds maximum_bytes")
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    data = np.empty(int(indptr[-1]), dtype=np.complex128)
    for column in range(n):
        start = int(indptr[column])
        for offset, (row, value) in enumerate(
            action.column_entries(column), start=start
        ):
            indices[offset] = row
            data[offset] = value
    matrix = csc_matrix((data, indices, indptr), shape=(n, n))
    matrix.eliminate_zeros()
    matrix.sort_indices()
    phase = np.arange(1, n + 1, dtype=np.float64)
    probe = np.cos(phase) + 1j * np.sin(phase)
    expected = action.matvec(probe)
    observed = matrix @ probe
    scale = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    parity = np.float64(np.linalg.norm(observed - expected) / scale)
    return ExplicitPhysicalSectorOperator(
        matrix=matrix,
        action=action,
        preflight=preflight,
        assembly_wall_seconds=np.float64(perf_counter() - started),
        action_parity_defect=parity,
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


@dataclass(frozen=True, slots=True)
class SparsePhysicalNESS:
    """Trace-fixed sparse NESS with a matrix-free residual cross-check."""

    operator: ExplicitPhysicalSectorOperator
    coefficients: ComplexVector
    trace: complex
    trace_defect: np.float64
    absolute_residual: np.float64
    relative_residual: np.float64
    scaled_residual: np.float64
    hermiticity_defect: np.float64
    minimum_eigenvalue: np.float64
    factor_nnz: int
    factorization_wall_seconds: np.float64
    solve_wall_seconds: np.float64
    peak_rss_kib: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficients",
            _immutable(self.coefficients, np.dtype(np.complex128)),
        )

    def density_matrix(self) -> NDArray[np.complex128]:
        d = self.operator.action.sector.fock_dimension
        rho = np.zeros((d, d), dtype=np.complex128)
        sector = self.operator.action.sector
        rho[sector.rows, sector.columns] = self.coefficients
        return rho


def solve_trace_fixed_static(
    operator: ExplicitPhysicalSectorOperator,
) -> SparsePhysicalNESS:
    """Solve ``L rho=0, Tr rho=1`` and reapply the matrix-free generator."""

    if not isinstance(operator, ExplicitPhysicalSectorOperator):
        raise TypeError("operator must be an ExplicitPhysicalSectorOperator")
    action = operator.action
    if action.odd_response:
        raise ValueError("trace-fixed solve requires the density generator")
    sector = action.sector
    trace = np.asarray(sector.rows == sector.columns, dtype=np.complex128)
    trace_indices = np.flatnonzero(trace)
    if trace_indices.size != sector.fock_dimension:
        raise RuntimeError("the complete Q=0 sector must contain every diagonal")
    constraint_row = int(trace_indices[0])
    constrained = operator.matrix.tolil(copy=True)
    constrained[constraint_row, :] = trace
    constrained = constrained.tocsc()
    rhs = np.zeros(operator.dimension, dtype=np.complex128)
    rhs[constraint_row] = 1.0
    factor_started = perf_counter()
    factor = splu(constrained)
    factor_seconds = perf_counter() - factor_started
    solve_started = perf_counter()
    rho = np.asarray(factor.solve(rhs), dtype=np.complex128)
    solve_seconds = perf_counter() - solve_started

    # This action is recomputed from the indexed-Gamma rules rather than from
    # the assembled CSC matrix used by SuperLU.
    residual = action.matvec(rho)
    absolute = np.linalg.norm(residual)
    rho_norm = max(float(np.linalg.norm(rho)), np.finfo(np.float64).tiny)
    matrix_norm = max(
        float(np.linalg.norm(operator.matrix.data)), np.finfo(np.float64).tiny
    )
    trace_value = complex(trace @ rho)
    d = sector.fock_dimension
    density = np.zeros((d, d), dtype=np.complex128)
    density[sector.rows, sector.columns] = rho
    density_norm = max(float(np.linalg.norm(density)), np.finfo(np.float64).tiny)
    hermiticity = np.linalg.norm(density - density.conj().T) / density_norm
    eigenvalues = np.linalg.eigvalsh((density + density.conj().T) / 2.0)
    return SparsePhysicalNESS(
        operator=operator,
        coefficients=rho,
        trace=trace_value,
        trace_defect=np.float64(abs(trace_value - 1.0)),
        absolute_residual=np.float64(absolute),
        relative_residual=np.float64(absolute / rho_norm),
        scaled_residual=np.float64(absolute / (matrix_norm * rho_norm)),
        hermiticity_defect=np.float64(hermiticity),
        minimum_eigenvalue=np.float64(eigenvalues[0]),
        factor_nnz=int(factor.L.nnz + factor.U.nnz),
        factorization_wall_seconds=np.float64(factor_seconds),
        solve_wall_seconds=np.float64(solve_seconds),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


@dataclass(frozen=True, slots=True)
class PhysicalResponseVectors:
    """Four regression sources and their algebraic trace sinks for one orbital."""

    plus_sector: PhysicalMatrixUnitSector
    minus_sector: PhysicalMatrixUnitSector
    creation_left: ComplexVector
    creation_right: ComplexVector
    annihilation_right: ComplexVector
    annihilation_left: ComplexVector
    annihilation_sink: ComplexVector
    creation_sink: ComplexVector


def physical_response_vectors(
    static_sector: PhysicalMatrixUnitSector,
    rho: ArrayLike,
    orbital: int,
) -> PhysicalResponseVectors:
    """Apply the physical source and sink operators without dense supermatrices."""

    if (static_sector.q_up, static_sector.q_down) != (0, 0):
        raise ValueError("rho must use the complete static Q=(0,0) sector")
    x = np.asarray(rho, dtype=np.complex128)
    if x.shape != (static_sector.dimension,) or not np.all(np.isfinite(x)):
        raise ValueError("rho must be a finite vector in the static sector")
    n_orbitals = 2 * static_sector.n_sites
    if isinstance(orbital, bool) or not isinstance(orbital, int):
        raise TypeError("orbital must be an integer")
    if not 0 <= orbital < n_orbitals:
        raise ValueError("orbital is outside the physical ordering")
    spin = orbital % 2
    plus = physical_matrix_unit_sector(
        static_sector.n_sites, *( (1, 0) if spin == 0 else (0, 1) )
    )
    minus = physical_matrix_unit_sector(
        static_sector.n_sites, *( (-1, 0) if spin == 0 else (0, -1) )
    )
    creation_left = np.zeros(plus.dimension, dtype=np.complex128)
    creation_right = np.zeros(plus.dimension, dtype=np.complex128)
    annihilation_right = np.zeros(minus.dimension, dtype=np.complex128)
    annihilation_left = np.zeros(minus.dimension, dtype=np.complex128)
    for index in np.flatnonzero(x):
        value = x[index]
        row = int(static_sector.rows[index])
        column = int(static_sector.columns[index])
        created = _create(row, orbital)
        if created is not None:
            target, sign = created
            creation_left[plus.index(target, column)] += sign * value
        annihilated = _annihilate(column, orbital)
        if annihilated is not None:
            target, sign = annihilated
            creation_right[plus.index(row, target)] += sign * value
        created = _create(column, orbital)
        if created is not None:
            target, sign = created
            annihilation_right[minus.index(row, target)] += sign * value
        annihilated = _annihilate(row, orbital)
        if annihilated is not None:
            target, sign = annihilated
            annihilation_left[minus.index(target, column)] += sign * value

    annihilation_sink = np.zeros(plus.dimension, dtype=np.complex128)
    for index, (row, column) in enumerate(zip(plus.rows, plus.columns, strict=True)):
        acted = _annihilate(int(row), orbital)
        if acted is not None and acted[0] == column:
            annihilation_sink[index] = acted[1]
    creation_sink = np.zeros(minus.dimension, dtype=np.complex128)
    for index, (row, column) in enumerate(
        zip(minus.rows, minus.columns, strict=True)
    ):
        acted = _create(int(row), orbital)
        if acted is not None and acted[0] == column:
            creation_sink[index] = acted[1]
    return PhysicalResponseVectors(
        plus_sector=plus,
        minus_sector=minus,
        creation_left=creation_left,
        creation_right=creation_right,
        annihilation_right=annihilation_right,
        annihilation_left=annihilation_left,
        annihilation_sink=annihilation_sink,
        creation_sink=creation_sink,
    )


__all__ = [
    "ExplicitPhysicalSectorOperator",
    "IndexedGammaSectorAction",
    "PhysicalMatrixUnitSector",
    "PhysicalResponseVectors",
    "SparsePhysicalNESS",
    "compile_physical_sector_operator",
    "physical_density_action",
    "physical_matrix_unit_sector",
    "physical_response_action",
    "physical_response_vectors",
    "solve_trace_fixed_static",
]

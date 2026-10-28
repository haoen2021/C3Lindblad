"""Explicit map from column-major density vectors to fermionic superkets.

The doubled Jordan--Wigner order is
``(c_0, tilde_c_0, c_1, tilde_c_1, ...)``.  The map is constructed exactly as
specified: column ``vec(E_mn)`` is ``Ehat_mn |I_F>``, where ``Ehat_mn`` is the
normal-fermion polynomial for the physical matrix unit, not a naive tensor
matrix unit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tests.reference.physical_fock import (
    _require_dense_allocation,
)

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]


def doubled_annihilation_operators(
    num_physical_orbitals: int, *, max_dense_bytes: int | None = None
) -> tuple[tuple[ComplexMatrix, ...], tuple[ComplexMatrix, ...]]:
    """Return normal and tilde annihilators in the interleaved global JW order."""

    if isinstance(num_physical_orbitals, bool) or not isinstance(
        num_physical_orbitals, int
    ):
        raise TypeError("num_physical_orbitals must be an integer")
    if num_physical_orbitals < 1:
        raise ValueError("num_physical_orbitals must be positive")
    num_doubled_orbitals = 2 * num_physical_orbitals
    doubled_dimension = 1 << num_doubled_orbitals
    _require_dense_allocation(
        (doubled_dimension, doubled_dimension),
        "independent doubled-Fock annihilator list",
        max_dense_bytes,
        copies=num_doubled_orbitals,
    )
    doubled: list[ComplexMatrix] = []
    for q in range(num_doubled_orbitals):
        operator = np.zeros((doubled_dimension, doubled_dimension), dtype=np.complex128)
        bit = 1 << q
        for mask in range(doubled_dimension):
            if mask & bit:
                target = mask ^ bit
                sign = -1.0 if (mask & (bit - 1)).bit_count() % 2 else 1.0
                operator[target, mask] = sign
        doubled.append(operator)
    return tuple(doubled[0::2]), tuple(doubled[1::2])


def interleaved_mask(
    normal_mask: int, tilde_mask: int, num_physical_orbitals: int
) -> int:
    """Interleave physical and tilde occupation bits into a doubled-Fock mask."""

    dimension = 1 << num_physical_orbitals
    if not 0 <= normal_mask < dimension or not 0 <= tilde_mask < dimension:
        raise ValueError("normal or tilde mask is outside the physical range")
    doubled_mask = 0
    for p in range(num_physical_orbitals):
        doubled_mask |= ((normal_mask >> p) & 1) << (2 * p)
        doubled_mask |= ((tilde_mask >> p) & 1) << (2 * p + 1)
    return doubled_mask


def fermionic_identity_ket(
    num_physical_orbitals: int, *, max_dense_bytes: int | None = None
) -> ComplexVector:
    """Construct ``|I_F> = sum_m (-i)^N_m |m,tilde m>_F``."""

    if isinstance(num_physical_orbitals, bool) or not isinstance(
        num_physical_orbitals, int
    ):
        raise TypeError("num_physical_orbitals must be an integer")
    if num_physical_orbitals < 1:
        raise ValueError("num_physical_orbitals must be positive")
    physical_dimension = 1 << num_physical_orbitals
    doubled_dimension = physical_dimension**2
    _require_dense_allocation(
        (doubled_dimension,), "fermionic identity ket", max_dense_bytes
    )
    identity_ket = np.zeros(doubled_dimension, dtype=np.complex128)
    for mask in range(physical_dimension):
        doubled_mask = interleaved_mask(mask, mask, num_physical_orbitals)
        identity_ket[doubled_mask] = (-1j) ** mask.bit_count()
    return identity_ket


def fermionic_trace_bra(
    num_physical_orbitals: int, *, max_dense_bytes: int | None = None
) -> ComplexVector:
    """Return the fixed trace functional ``<I_F|`` as a row-vector array."""

    return fermionic_identity_ket(
        num_physical_orbitals, max_dense_bytes=max_dense_bytes
    ).conj()


def embedded_physical_matrix_unit(
    row_mask: int,
    column_mask: int,
    normal_annihilation: Sequence[ComplexMatrix],
) -> ComplexMatrix:
    """Build ``Ehat_mn = C_m^dagger P_0 C_n`` in doubled Fock space.

    ``C_m^dagger`` is ordered by increasing occupied orbital, while ``C_n``
    is ordered by decreasing occupied orbital.  This is the exact fermionic
    polynomial for ``|m><n|`` under the declared physical basis convention.
    """

    num_physical_orbitals = len(normal_annihilation)
    physical_dimension = 1 << num_physical_orbitals
    if (
        not 0 <= row_mask < physical_dimension
        or not 0 <= column_mask < physical_dimension
    ):
        raise ValueError("matrix-unit mask is outside the physical range")
    doubled_dimension = normal_annihilation[0].shape[0]
    identity = np.eye(doubled_dimension, dtype=np.complex128)

    create_row = identity
    for p in range(num_physical_orbitals):
        if row_mask & (1 << p):
            create_row = create_row @ normal_annihilation[p].conj().T

    vacuum_projector = identity
    for operator in normal_annihilation:
        vacuum_projector = vacuum_projector @ (identity - operator.conj().T @ operator)

    annihilate_column = identity
    for p in range(num_physical_orbitals - 1, -1, -1):
        if column_mask & (1 << p):
            annihilate_column = annihilate_column @ normal_annihilation[p]
    return create_row @ vacuum_projector @ annihilate_column


def fermionic_mapping(
    num_physical_orbitals: int, *, max_dense_bytes: int | None = None
) -> ComplexMatrix:
    """Construct the monomial unitary ``S_F`` from every physical matrix unit."""

    physical_dimension = 1 << num_physical_orbitals
    doubled_dimension = physical_dimension**2
    cumulative_elements = (
        2 * num_physical_orbitals + 8
    ) * doubled_dimension * doubled_dimension + doubled_dimension
    _require_dense_allocation(
        (cumulative_elements,),
        "cumulative fermionic map, doubled operators, and construction workspaces",
        max_dense_bytes,
    )
    normal_annihilation, _ = doubled_annihilation_operators(
        num_physical_orbitals, max_dense_bytes=max_dense_bytes
    )
    identity_ket = fermionic_identity_ket(
        num_physical_orbitals, max_dense_bytes=max_dense_bytes
    )
    mapping = np.zeros((doubled_dimension, doubled_dimension), dtype=np.complex128)
    for column_mask in range(physical_dimension):
        for row_mask in range(physical_dimension):
            vector_index = row_mask + physical_dimension * column_mask
            matrix_unit = embedded_physical_matrix_unit(
                row_mask, column_mask, normal_annihilation
            )
            mapping[:, vector_index] = matrix_unit @ identity_ket
    return mapping


def fermionic_mapping_formula(num_physical_orbitals: int) -> ComplexMatrix:
    """Independent closed-form audit of the explicitly constructed ``S_F``.

    The parity exponent is
    ``sum_{p<q} n_p n_q + sum_{p<q} n_p m_q`` modulo two for column
    ``vec(E_mn)``.  It is tested against, but never substitutes for, the
    matrix-unit construction in the M2 oracle.
    """

    physical_dimension = 1 << num_physical_orbitals
    doubled_dimension = physical_dimension**2
    mapping = np.zeros((doubled_dimension, doubled_dimension), dtype=np.complex128)
    for column_mask in range(physical_dimension):
        for row_mask in range(physical_dimension):
            parity = 0
            for p in range(num_physical_orbitals):
                if not column_mask & (1 << p):
                    continue
                for q in range(p + 1, num_physical_orbitals):
                    parity ^= bool(column_mask & (1 << q))
                    parity ^= bool(row_mask & (1 << q))
            phase = ((-1j) ** column_mask.bit_count()) * (-1.0 if parity else 1.0)
            doubled_mask = interleaved_mask(
                row_mask, column_mask, num_physical_orbitals
            )
            vector_index = row_mask + physical_dimension * column_mask
            mapping[doubled_mask, vector_index] = phase
    return mapping


def operator_parity(num_physical_orbitals: int) -> ComplexMatrix:
    """Return doubled total parity, equal to physical matrix-unit parity."""

    doubled_dimension = 1 << (2 * num_physical_orbitals)
    signs = np.fromiter(
        (-1.0 if mask.bit_count() % 2 else 1.0 for mask in range(doubled_dimension)),
        dtype=np.float64,
        count=doubled_dimension,
    )
    return np.diag(signs).astype(np.complex128)


def normal_parity(num_physical_orbitals: int) -> ComplexMatrix:
    """Return parity of normal occupations alone in interleaved order."""

    doubled_dimension = 1 << (2 * num_physical_orbitals)
    signs = np.empty(doubled_dimension, dtype=np.float64)
    for mask in range(doubled_dimension):
        normal_count = sum((mask >> (2 * p)) & 1 for p in range(num_physical_orbitals))
        signs[mask] = -1.0 if normal_count % 2 else 1.0
    return np.diag(signs).astype(np.complex128)


def parity_projectors(
    num_physical_orbitals: int,
) -> tuple[ComplexMatrix, ComplexMatrix]:
    """Return even- and odd-operator projectors in doubled Fock space."""

    parity = operator_parity(num_physical_orbitals)
    identity = np.eye(parity.shape[0], dtype=np.complex128)
    return (identity + parity) / 2.0, (identity - parity) / 2.0


def quadratic_fermionic_mapping(
    num_physical_orbitals: int, *, max_dense_bytes: int | None = None
) -> ComplexMatrix:
    """Return the explicit odd-parity twist for one global quadratic ``paper_K``.

    ``S_Q = (P_even + parity_normal P_odd) S_F`` equals the identity-anchored
    matrix-unit map on even physical density operators.  On odd response
    operators it supplies the Klein factor that makes the printed quadratic
    generator valid.  Observable maps must record which of these two monomial
    maps they use.
    """

    doubled_dimension = 1 << (2 * num_physical_orbitals)
    _require_dense_allocation(
        ((2 * num_physical_orbitals + 15) * doubled_dimension * doubled_dimension,),
        "cumulative quadratic map twist and fermionic-map workspaces",
        max_dense_bytes,
    )
    even, odd = parity_projectors(num_physical_orbitals)
    twist = even + normal_parity(num_physical_orbitals) @ odd
    return twist @ fermionic_mapping(
        num_physical_orbitals, max_dense_bytes=max_dense_bytes
    )

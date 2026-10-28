"""Small physical-Fock fermion matrices built directly from occupation bits.

This module is intentionally independent of the superfermion signs and
term-compilation code.  With

``|mask> = product_p_increasing (c_p^dagger)^n_p |vac>``,

annihilating orbital ``p`` has sign ``(-1)**popcount(mask below p)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from c3l.conventions import TolerancePolicy
from c3l.interactions import SparseTwoBodyInteraction
from c3l.models import FermionicLindbladModel

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]


def _memory_limit(max_dense_bytes: int | None) -> int:
    limit = (
        TolerancePolicy.first().dense_memory_limit_bytes
        if max_dense_bytes is None
        else max_dense_bytes
    )
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("max_dense_bytes must be a positive integer")
    return limit


def _require_dense_allocation(
    shape: tuple[int, ...],
    label: str,
    max_dense_bytes: int | None,
    *,
    copies: int = 1,
) -> None:
    elements = 1
    for extent in shape:
        elements *= extent
    required = elements * np.dtype(np.complex128).itemsize * copies
    limit = _memory_limit(max_dense_bytes)
    if required > limit:
        raise MemoryError(
            f"{label} requires an estimated {required} bytes, above the "
            f"configured {limit}-byte dense-memory limit"
        )


def annihilation_operator(
    num_orbitals: int, orbital: int, *, max_dense_bytes: int | None = None
) -> ComplexMatrix:
    """Construct one Jordan--Wigner annihilator in the declared bit order."""

    if isinstance(num_orbitals, bool) or not isinstance(num_orbitals, int):
        raise TypeError("num_orbitals must be an integer")
    if isinstance(orbital, bool) or not isinstance(orbital, int):
        raise TypeError("orbital must be an integer")
    if num_orbitals < 1 or not 0 <= orbital < num_orbitals:
        raise ValueError("orbital is outside the physical-Fock range")

    dimension = 1 << num_orbitals
    _require_dense_allocation(
        (dimension, dimension), "one physical-Fock operator", max_dense_bytes
    )
    operator = np.zeros((dimension, dimension), dtype=np.complex128)
    bit = 1 << orbital
    lower_bits = bit - 1
    for mask in range(dimension):
        if mask & bit:
            target = mask ^ bit
            sign = -1.0 if (mask & lower_bits).bit_count() % 2 else 1.0
            operator[target, mask] = np.complex128(sign)
    operator.flags.writeable = False
    return operator


def annihilation_operators(
    num_orbitals: int, *, max_dense_bytes: int | None = None
) -> tuple[ComplexMatrix, ...]:
    """Return all physical annihilators in increasing Jordan--Wigner order."""

    if isinstance(num_orbitals, bool) or not isinstance(num_orbitals, int):
        raise TypeError("num_orbitals must be an integer")
    if num_orbitals < 1:
        raise ValueError("num_orbitals must be positive")
    dimension = 1 << num_orbitals
    _require_dense_allocation(
        (dimension, dimension),
        "physical-Fock annihilator list",
        max_dense_bytes,
        copies=num_orbitals,
    )
    return tuple(
        annihilation_operator(num_orbitals, p, max_dense_bytes=max_dense_bytes)
        for p in range(num_orbitals)
    )


def creation_operators(
    annihilation: Sequence[ComplexMatrix],
) -> tuple[ComplexMatrix, ...]:
    """Return the Euclidean adjoints of physical-Fock annihilators."""

    creation = []
    for operator in annihilation:
        dagger = np.array(operator.conj().T, dtype=np.complex128, copy=True)
        dagger.flags.writeable = False
        creation.append(dagger)
    return tuple(creation)


def number_operator(annihilation: ComplexMatrix) -> ComplexMatrix:
    """Return ``c^dagger c`` for one physical orbital."""

    number = np.array(annihilation.conj().T @ annihilation, dtype=np.complex128)
    number.flags.writeable = False
    return number


def vectorize(matrix: ArrayLike) -> ComplexVector:
    """Column-major ``vec`` with matrix row index varying fastest."""

    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError("vectorize expects a matrix")
    return np.array(array.reshape(-1, order="F"), dtype=np.complex128, copy=True)


def unvectorize(vector: ArrayLike, dimension: int | None = None) -> ComplexMatrix:
    """Invert column-major ``vec``; infer a square dimension when omitted."""

    array = np.asarray(vector, dtype=np.complex128)
    if array.ndim != 1:
        raise ValueError("unvectorize expects a one-dimensional vector")
    if dimension is None:
        dimension = int(np.sqrt(array.size))
    if dimension < 1 or dimension * dimension != array.size:
        raise ValueError(
            "vector length is incompatible with the requested square matrix"
        )
    return np.array(array.reshape((dimension, dimension), order="F"), copy=True)


def quadratic_hamiltonian(
    E: ArrayLike, annihilation: Sequence[ComplexMatrix]
) -> ComplexMatrix:
    """Build ``H = sum_ij E_ij c_i^dagger c_j`` directly."""

    one_body = np.asarray(E, dtype=np.complex128)
    num_orbitals = len(annihilation)
    if one_body.shape != (num_orbitals, num_orbitals):
        raise ValueError("E shape must match the physical annihilator list")
    dimension = annihilation[0].shape[0]
    H = np.zeros((dimension, dimension), dtype=np.complex128)
    for i in range(num_orbitals):
        creation_i = annihilation[i].conj().T
        for j in range(num_orbitals):
            H += one_body[i, j] * (creation_i @ annihilation[j])
    return H


def auxiliary_hamiltonian(
    model: FermionicLindbladModel, annihilation: Sequence[ComplexMatrix]
) -> ComplexMatrix:
    """Build the model Hamiltonian in its physical orbital order."""

    if len(annihilation) != model.n_spin_orbitals:
        raise ValueError("annihilator count does not match the model")
    energy = model.E if model.symmetry == "charge" else expand_spin_diagonal(model.E)
    hamiltonian = quadratic_hamiltonian(energy, annihilation)
    hamiltonian += interaction_hamiltonian(model.interaction, annihilation)
    return hamiltonian


def expand_spin_diagonal(matrix: ArrayLike) -> ComplexMatrix:
    """Replicate a spatial matrix with no spin mixing in site-major order."""

    spatial = np.asarray(matrix, dtype=np.complex128)
    if spatial.ndim != 2 or spatial.shape[0] != spatial.shape[1]:
        raise ValueError("spatial matrix must be square")
    n_sites = spatial.shape[0]
    expanded = np.zeros((2 * n_sites, 2 * n_sites), dtype=np.complex128)
    for i in range(n_sites):
        for j in range(n_sites):
            for spin in (0, 1):
                expanded[2 * i + spin, 2 * j + spin] = spatial[i, j]
    return expanded


def interaction_hamiltonian(
    interaction: SparseTwoBodyInteraction,
    annihilation: Sequence[ComplexMatrix],
) -> ComplexMatrix:
    """Build the exact small-Fock-space interaction Hamiltonian."""

    if not isinstance(interaction, SparseTwoBodyInteraction):
        raise TypeError("interaction must be a SparseTwoBodyInteraction")
    if len(annihilation) != interaction.n_spin_orbitals:
        raise ValueError("annihilator count does not match the interaction")
    shape = annihilation[0].shape
    if any(operator.shape != shape for operator in annihilation):
        raise ValueError("all annihilation operators must share one matrix shape")
    result = np.zeros(shape, dtype=np.complex128)
    for p, q, r, s, value in interaction.terms:
        result += (
            value
            * annihilation[p].conj().T
            @ annihilation[q].conj().T
            @ annihilation[s]
            @ annihilation[r]
        )
    return result

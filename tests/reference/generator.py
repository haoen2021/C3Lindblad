"""Explicit paper-convention generator in the interleaved superfermion Fock space."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .physical_fock import interaction_hamiltonian
from c3l.models import FermionicLindbladModel, QuadraticOpenSystem
from tests.reference.physical_fock import _require_dense_allocation

from .mapping import doubled_annihilation_operators, operator_parity

ComplexMatrix = NDArray[np.complex128]
PhysicalModel = QuadraticOpenSystem | FermionicLindbladModel


def _spin_expand(matrix: ComplexMatrix) -> ComplexMatrix:
    """Expand a spatial matrix in site-major, spin-minor order independently."""

    n_sites = matrix.shape[0]
    expanded = np.zeros((2 * n_sites, 2 * n_sites), dtype=np.complex128)
    for i in range(n_sites):
        for j in range(n_sites):
            for spin in (0, 1):
                expanded[2 * i + spin, 2 * j + spin] = matrix[i, j]
    return expanded


def _physical_one_body_matrices(
    model: PhysicalModel,
) -> tuple[ComplexMatrix, ComplexMatrix, ComplexMatrix]:
    if isinstance(model, QuadraticOpenSystem):
        return model.E, model.gamma_loss, model.gamma_gain
    if isinstance(model, FermionicLindbladModel):
        if model.symmetry == "charge":
            return model.E, model.gamma_loss, model.gamma_gain
        return (
            _spin_expand(model.E),
            _spin_expand(model.gamma_loss),
            _spin_expand(model.gamma_gain),
        )
    raise TypeError(
        "model must be a validated QuadraticOpenSystem or FermionicLindbladModel"
    )


def quadratic_mode_matrix(
    E: ComplexMatrix, gamma_loss: ComplexMatrix, gamma_gain: ComplexMatrix
) -> ComplexMatrix:
    """Return the paper normal-mode matrix ``h`` with unmodified block indices."""

    Lambda = gamma_loss + gamma_gain
    Omega = gamma_gain - gamma_loss
    return np.block(
        [
            [E + 1j * Omega, 2.0 * gamma_gain],
            [-2.0 * gamma_loss, E - 1j * Omega],
        ]
    ).astype(np.complex128, copy=False)


def _superfermion_generator_parts(
    model: PhysicalModel, *, max_dense_bytes: int | None = None
) -> tuple[ComplexMatrix, ComplexMatrix]:
    """Return parity-diagonal and pair-transfer parts of ``paper_K``.

    For a generic physical-orbital quadratic model the scalar is
    ``-Tr(E + i Lambda)``. Spin expansion of a ``FermionicLindbladModel`` turns this
    into the paper's ``-2 Tr(E + i Lambda)`` before adding
    ``U(n_up n_down - tilde_n_up tilde_n_down)``.
    """

    E, gamma_loss, gamma_gain = _physical_one_body_matrices(model)
    num_physical_orbitals = E.shape[0]
    doubled_dimension = 1 << (2 * num_physical_orbitals)
    _require_dense_allocation(
        ((4 * num_physical_orbitals + 14) * doubled_dimension * doubled_dimension,),
        "cumulative explicit superfermion paper_K operators and workspaces",
        max_dense_bytes,
    )
    normal, tilde = doubled_annihilation_operators(
        num_physical_orbitals, max_dense_bytes=max_dense_bytes
    )
    sharp = tuple(operator.conj().T for operator in normal) + tuple(tilde)
    dual = tuple(normal) + tuple(operator.conj().T for operator in tilde)
    h = quadratic_mode_matrix(E, gamma_loss, gamma_gain)
    diagonal = np.zeros((doubled_dimension, doubled_dimension), dtype=np.complex128)
    transfer = np.zeros_like(diagonal)
    for a, sharp_a in enumerate(sharp):
        for b, dual_b in enumerate(dual):
            term = h[a, b] * (sharp_a @ dual_b)
            if (a < num_physical_orbitals) != (b < num_physical_orbitals):
                transfer += term
            else:
                diagonal += term

    Lambda = gamma_loss + gamma_gain
    identity = np.eye(doubled_dimension, dtype=np.complex128)
    diagonal -= np.trace(E + 1j * Lambda) * identity

    if isinstance(model, FermionicLindbladModel) and model.has_interactions:
        diagonal += interaction_hamiltonian(model.interaction, normal)
        for p, q, r, s, value in model.interaction.terms:
            diagonal -= (
                value.conjugate()
                * tilde[p].conj().T
                @ tilde[q].conj().T
                @ tilde[s]
                @ tilde[r]
            )
    return diagonal, transfer


def printed_quadratic_paper_K(
    model: PhysicalModel, *, max_dense_bytes: int | None = None
) -> ComplexMatrix:
    """Construct the printed polynomial: even under ``S_F``, global under ``S_Q``."""

    diagonal, transfer = _superfermion_generator_parts(
        model, max_dense_bytes=max_dense_bytes
    )
    return diagonal + transfer


def superfermion_paper_K(
    model: PhysicalModel, *, max_dense_bytes: int | None = None
) -> ComplexMatrix:
    """Construct the full ``S_F (i physical_generator) S_F^-1``.

    The matrix-unit map fixes recycling terms with the printed sign on even
    operator parity and the opposite sign on odd parity.  Total doubled parity
    is the exact Klein factor; the even physical-density sector is unchanged.
    """

    diagonal, transfer = _superfermion_generator_parts(
        model, max_dense_bytes=max_dense_bytes
    )
    E, _, _ = _physical_one_body_matrices(model)
    return diagonal + transfer @ operator_parity(E.shape[0])


def superfermion_parity(num_physical_orbitals: int) -> ComplexMatrix:
    """Return ``(-1)^(N+tilde N)``, equal to physical operator parity under S_F."""

    return operator_parity(num_physical_orbitals)


def parity_projectors(
    num_physical_orbitals: int,
) -> tuple[ComplexMatrix, ComplexMatrix]:
    """Return even- and odd-operator projectors in doubled Fock space."""

    parity = superfermion_parity(num_physical_orbitals)
    identity = np.eye(parity.shape[0], dtype=np.complex128)
    return (identity + parity) / 2.0, (identity - parity) / 2.0

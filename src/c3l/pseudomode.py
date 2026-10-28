"""Finite bath representations and their embedding into a Lindblad model.

RealTimeBath performs fitting through :mod:`c3l.fitting`. This module
stores the resulting mode matrices, evaluates their branch correlations,
and assembles the finite bath used by the CC and response solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .interactions import SparseTwoBodyInteraction
from .models import FermionicLindbladModel

Branch = Literal["lesser", "greater"]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class PseudomodeRealization:
    r"""Hermitian one-body matrix, damping matrix, and coupling vector.

    For branch ``b`` the correlation is

    .. math::

       C_b(t)=g^\dagger\exp[(s_b iH-\Gamma)t]g,

    with ``s_lesser=+1`` and ``s_greater=-1``.

    ``H`` contains mode energies and coherent mode-to-mode couplings;
    ``damping`` contains decay rates and dissipative mode couplings;
    ``g`` couples the impurity channel to these modes. For ``m`` modes their
    shapes are ``(m,m)``, ``(m,m)``, and ``(m,)``. A realization is one finite
    representation of a correlation function, not a new fitting algorithm.
    Lesser and greater branches have separate realizations. The fitting
    adapter converts RealTimeBath's signs to the convention above.
    """

    H: ComplexArray
    damping: ComplexArray
    g: ComplexArray

    @property
    def size(self) -> int:
        return int(self.g.size)


def realization_bcf(
    t: ArrayLike, realization: PseudomodeRealization, branch: Branch
) -> ComplexArray:
    """Evaluate the branch correlation of one finite realization."""

    times = np.asarray(t, dtype=np.float64)
    if branch not in ("lesser", "greater"):
        raise ValueError("branch must be 'lesser' or 'greater'")
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("t must be a finite vector")
    H = np.asarray(realization.H, dtype=np.complex128)
    damping = np.asarray(realization.damping, dtype=np.complex128)
    g = np.asarray(realization.g, dtype=np.complex128)
    if H.shape != damping.shape or H.shape != (g.size, g.size):
        raise ValueError("H, damping, and g have incompatible shapes")
    sign = 1.0 if branch == "lesser" else -1.0
    generator = sign * 1j * H - damping
    eigenvalues, right = np.linalg.eig(generator)
    weights = (g.conj() @ right) * np.linalg.solve(right, g)
    return np.asarray(weights @ np.exp(np.outer(eigenvalues, times)))


def matrix_pseudomode_model(
    impurity_one_body: ArrayLike,
    orbital_matrix: ArrayLike,
    lesser: PseudomodeRealization,
    greater: PseudomodeRealization,
    interaction: SparseTwoBodyInteraction,
    *,
    units: str = "arbitrary",
) -> FermionicLindbladModel:
    r"""Embed ``Gamma(omega)=M g(omega)`` in the original orbital basis.

    The positive-semidefinite matrix ``M`` is diagonalized only to define
    independent bath channels.  The impurity block and all requested
    observables remain in their supplied orbital basis.

    This convenience builder assumes a frequency-independent orbital matrix
    times a common scalar spectral shape, with spin-independent one-body
    and bath matrices. It does not fit an arbitrary matrix-valued bath.
    More general bath layouts can be assembled directly as a
    :class:`FermionicLindbladModel` from their one-body, gain and loss matrices.
    """

    h_impurity = np.asarray(impurity_one_body, dtype=np.complex128)
    matrix = np.asarray(orbital_matrix, dtype=np.complex128)
    if (
        h_impurity.ndim != 2
        or h_impurity.shape[0] != h_impurity.shape[1]
        or matrix.shape != h_impurity.shape
    ):
        raise ValueError("impurity_one_body and orbital_matrix must share a square shape")
    if not np.all(np.isfinite(h_impurity)) or not np.all(np.isfinite(matrix)):
        raise ValueError("impurity and orbital matrices must be finite")
    scale = max(float(np.linalg.norm(matrix, ord=2)), 1.0)
    threshold = 1.0e-12 * scale
    if np.linalg.norm(h_impurity - h_impurity.conj().T, ord="fro") > threshold:
        raise ValueError("impurity_one_body must be Hermitian")
    if np.linalg.norm(matrix - matrix.conj().T, ord="fro") > threshold:
        raise ValueError("orbital_matrix must be Hermitian")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues[0] < -threshold:
        raise ValueError("orbital_matrix must be positive semidefinite")
    channels = np.flatnonzero(eigenvalues > threshold)
    n_impurity = h_impurity.shape[0]
    if interaction.n_spin_orbitals != 2 * n_impurity:
        raise ValueError("interaction must use the impurity spin-orbital space")

    def validate(realization: PseudomodeRealization, name: str) -> None:
        H = np.asarray(realization.H, dtype=np.complex128)
        damping = np.asarray(realization.damping, dtype=np.complex128)
        coupling = np.asarray(realization.g, dtype=np.complex128)
        if H.shape != damping.shape or H.shape != (coupling.size, coupling.size):
            raise ValueError(f"{name} realization arrays have incompatible shapes")
        if not all(np.all(np.isfinite(value)) for value in (H, damping, coupling)):
            raise ValueError(f"{name} realization must be finite")

    validate(lesser, "lesser")
    validate(greater, "greater")
    bath_per_channel = lesser.size + greater.size
    n_sites = n_impurity + len(channels) * bath_per_channel
    E = np.zeros((n_sites, n_sites), dtype=np.complex128)
    loss = np.zeros_like(E)
    gain = np.zeros_like(E)
    E[:n_impurity, :n_impurity] = h_impurity
    offset = n_impurity
    for channel in channels:
        vector = eigenvectors[:, channel]
        scale_channel = np.sqrt(max(float(eigenvalues[channel]), 0.0))
        for realization, pumped in ((lesser, True), (greater, False)):
            block = np.arange(offset, offset + realization.size)
            E[np.ix_(block, block)] = realization.H
            coupling = scale_channel * np.outer(vector, realization.g.conj())
            E[np.ix_(np.arange(n_impurity), block)] = coupling
            E[np.ix_(block, np.arange(n_impurity))] = coupling.conj().T
            if pumped:
                gain[np.ix_(block, block)] = realization.damping
            else:
                loss[np.ix_(block, block)] = realization.damping
            offset += realization.size

    embedded = SparseTwoBodyInteraction(2 * n_sites, interaction.terms)
    return FermionicLindbladModel(
        E,
        loss,
        gain,
        U=0.0,
        impurity_site=0,
        units=units,
        interaction_sites=tuple(range(n_impurity)),
        two_body_interaction=embedded,
    )


__all__ = [
    "PseudomodeRealization",
    "matrix_pseudomode_model",
    "realization_bcf",
]

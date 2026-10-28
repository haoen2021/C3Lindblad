"""Archived lesser/greater bath arrays and SIAM/dimer Hamiltonian assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bath_targets import semicircle_bcf
from c3l.models import FermionicLindbladModel


ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]


def pseudomode_bcf(
    t: ArrayLike,
    H: ArrayLike,
    Gamma: ArrayLike,
    g: ArrayLike,
) -> ComplexVector:
    r"""Evaluate ``g^dag exp[(-i H-Gamma)t] g`` by a small eigendecomposition."""

    times = np.asarray(t, dtype=np.float64)
    h = np.asarray(H, dtype=np.complex128)
    damping = np.asarray(Gamma, dtype=np.complex128)
    coupling = np.asarray(g, dtype=np.complex128)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("t must be a finite nonempty one-dimensional grid")
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("H must be square")
    if damping.shape != h.shape or coupling.shape != (h.shape[0],):
        raise ValueError("Gamma and g must match H")
    if not all(np.all(np.isfinite(x)) for x in (h, damping, coupling)):
        raise ValueError("H, Gamma, and g must be finite")
    eigenvalues, right = np.linalg.eig(-1j * h - damping)
    weights = (coupling.conj() @ right) * np.linalg.solve(right, coupling)
    return np.asarray(
        weights @ np.exp(np.outer(eigenvalues, times)), dtype=np.complex128
    )


@dataclass(frozen=True, slots=True)
class CoupledPseudomodeFit:
    """One deterministic reconstructed lesser/greater pseudomode pair."""

    N: int
    H_lesser: ComplexMatrix
    Gamma_lesser: ComplexMatrix
    g_lesser: ComplexVector
    H_greater: ComplexMatrix
    Gamma_greater: ComplexMatrix
    g_greater: ComplexVector

def coupled_pseudomode_model(
    fit: CoupledPseudomodeFit,
    *,
    U: float,
    epsilon: float | None = None,
) -> FermionicLindbladModel:
    """Embed one reconstructed fit as pumped lesser and damped greater modes.

    Spatial ordering is impurity, ``N`` lesser modes, then ``N`` greater
    modes.  The ordering is repeated independently for up and down spin by
    :class:`FermionicLindbladModel`.  The repository dissipator is

    ``Gamma * (2 d rho d^dag - {d^dag d,rho})``;

    hence an isolated annihilator decays with ``-Gamma`` and no factor-of-two
    conversion is applied to the paper exponent ``-i H-Gamma``.
    """

    if not isinstance(fit, CoupledPseudomodeFit):
        raise TypeError("fit must be a CoupledPseudomodeFit")
    interaction = float(U)
    impurity_energy = -0.5 * interaction if epsilon is None else float(epsilon)
    N = fit.N
    n = 1 + 2 * N
    E = np.zeros((n, n), dtype=np.complex128)
    loss = np.zeros_like(E)
    gain = np.zeros_like(E)
    lesser = np.arange(1, 1 + N)
    greater = np.arange(1 + N, n)
    E[0, 0] = impurity_energy
    E[np.ix_(lesser, lesser)] = fit.H_lesser
    E[np.ix_(greater, greater)] = fit.H_greater
    E[lesser, 0] = fit.g_lesser
    E[0, lesser] = fit.g_lesser.conj()
    E[greater, 0] = fit.g_greater
    E[0, greater] = fit.g_greater.conj()
    gain[np.ix_(lesser, lesser)] = fit.Gamma_lesser
    loss[np.ix_(greater, greater)] = fit.Gamma_greater
    return FermionicLindbladModel(
        E,
        loss,
        gain,
        U=interaction,
        impurity_site=0,
        units="Huang_SM_S5_Gamma_bath",
    )


def coupled_pseudomode_dimer_model(
    fit: CoupledPseudomodeFit,
    *,
    U: float,
    t12: float = 0.5,
) -> FermionicLindbladModel:
    """Embed two interacting sites with independent copies of one fit.

    The spatial order is ``d1,d2,L1[N],G1[N],L2[N],G2[N]``.  Both impurity
    onsite energies are ``-U/2``, and the only intersite one-body element is
    ``E[d1,d2]=E[d2,d1]=-t12``.  Each local bath has its own gain/loss block;
    no dissipative or Hamiltonian bath coupling connects the two copies.
    """

    if not isinstance(fit, CoupledPseudomodeFit):
        raise TypeError("fit must be a CoupledPseudomodeFit")
    interaction = float(U)
    hopping = float(t12)
    if not np.isfinite(interaction) or not np.isfinite(hopping):
        raise ValueError("U and t12 must be finite")
    N = fit.N
    n = 2 + 4 * N
    E = np.zeros((n, n), dtype=np.complex128)
    loss = np.zeros_like(E)
    gain = np.zeros_like(E)
    E[0, 0] = E[1, 1] = -0.5 * interaction
    E[0, 1] = E[1, 0] = -hopping

    for impurity, offset in ((0, 2), (1, 2 + 2 * N)):
        lesser = np.arange(offset, offset + N)
        greater = np.arange(offset + N, offset + 2 * N)
        E[np.ix_(lesser, lesser)] = fit.H_lesser
        E[np.ix_(greater, greater)] = fit.H_greater
        E[lesser, impurity] = fit.g_lesser
        E[impurity, lesser] = fit.g_lesser.conj()
        E[greater, impurity] = fit.g_greater
        E[impurity, greater] = fit.g_greater.conj()
        gain[np.ix_(lesser, lesser)] = fit.Gamma_lesser
        loss[np.ix_(greater, greater)] = fit.Gamma_greater

    return FermionicLindbladModel(
        E,
        loss,
        gain,
        U=interaction,
        impurity_site=0,
        units="coupled_Lindblad_pseudomode_Gamma_bath_two_impurity_dimer",
        interaction_sites=(0, 1),
    )


__all__ = [
    "CoupledPseudomodeFit",
    "coupled_pseudomode_model",
    "coupled_pseudomode_dimer_model",
    "pseudomode_bcf",
    "semicircle_bcf",
]

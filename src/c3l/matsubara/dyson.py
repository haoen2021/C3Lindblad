r"""PM noninteracting Green function, hybridization, and Dyson self-energy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..models import FermionicLindbladModel
from .grid import ComplexGrid, MatsubaraGrid


def impurity_sites(
    model: FermionicLindbladModel, sites: Sequence[int] | None = None
) -> tuple[int, ...]:
    """Validate the impurity block ``S`` in the model's block basis.

    Indices are spatial sites in spin mode and physical orbitals in charge
    mode, matching ``component_eom_sources(site=...)``.  The default is the
    model's ``interaction_sites``.  Every interacting site must be included,
    otherwise :math:`G_0^{-1}-G^{-1}` is not the impurity self-energy.
    """

    if not isinstance(model, FermionicLindbladModel):
        raise TypeError("model must be a FermionicLindbladModel")
    chosen = tuple(model.interaction_sites if sites is None else sites)
    if not chosen:
        raise ValueError("at least one impurity site is required")
    if any(isinstance(site, (bool, np.bool_)) or not isinstance(site, (int, np.integer)) for site in chosen):
        raise TypeError("impurity sites must be integers")
    chosen = tuple(int(site) for site in chosen)
    if any(not 0 <= site < model.n_sites for site in chosen):
        raise ValueError("an impurity site is outside the model")
    if len(set(chosen)) != len(chosen):
        raise ValueError("impurity sites must be unique")
    if model.has_interactions and not set(model.interaction_sites) <= set(chosen):
        raise ValueError("impurity sites must include every interaction site")
    return chosen


def noninteracting_retarded_green(
    model: FermionicLindbladModel,
    shifts: ArrayLike,
    sites: Sequence[int] | None = None,
) -> ComplexGrid:
    r"""Return :math:`[(zI-E+i\Lambda)^{-1}]_{SS}` at each shift ``z``."""

    block = impurity_sites(model, sites)
    z = np.asarray(shifts, dtype=np.complex128)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("shifts must be a finite nonempty vector")
    n = model.n_sites
    identity = np.eye(n, dtype=np.complex128)
    effective = np.asarray(model.E, dtype=np.complex128) - 1j * (
        np.asarray(model.gamma_loss) + np.asarray(model.gamma_gain)
    )
    inverse = z[:, None, None] * identity[None, :, :] - effective[None, :, :]
    columns = np.broadcast_to(identity[:, list(block)], (z.size, n, len(block)))
    solution = np.linalg.solve(inverse, columns)
    return np.ascontiguousarray(solution[:, list(block), :])


def matsubara_noninteracting_green(
    model: FermionicLindbladModel,
    grid: MatsubaraGrid,
    sites: Sequence[int] | None = None,
) -> ComplexGrid:
    """PM noninteracting impurity Green function on a Matsubara grid."""

    return grid.from_upper(noninteracting_retarded_green(model, grid.upper_shifts, sites))


@dataclass(frozen=True, slots=True, eq=False)
class MatsubaraSelfEnergy:
    r"""Auxiliary Dyson quantities on a Matsubara grid (axes ``n, a, b``).

    ``hybridization`` is the PM bath :math:`\Delta_{\rm PM}(i\omega_n)=
    i\omega_n-h-G_0^{-1}` with ``one_body`` :math:`h=E_{SS}`.
    """

    grid: MatsubaraGrid
    sites: tuple[int, ...]
    one_body: NDArray[np.complex128]
    noninteracting: ComplexGrid
    hybridization: ComplexGrid
    self_energy: ComplexGrid
    G0_condition_number: NDArray[np.float64]
    G_condition_number: NDArray[np.float64]
    conditioning_limited: NDArray[np.bool_]


def matsubara_self_energy(
    model: FermionicLindbladModel,
    grid: MatsubaraGrid,
    green: ArrayLike,
    *,
    sites: Sequence[int] | None = None,
    conditioning_limit: float = 1.0e10,
) -> MatsubaraSelfEnergy:
    r"""Evaluate :math:`\Sigma=G_0^{-1}-G^{-1}` for ``green`` on ``grid``.

    ``green`` may come from any solver that returns the PM impurity block on
    the grid (for example GPU-resident Arnoldi evaluated at
    ``grid.upper_shifts`` and mapped with :meth:`MatsubaraGrid.from_upper`).
    """

    from ..spectra.reconstruction import matrix_auxiliary_self_energy

    block = impurity_sites(model, sites)
    G = np.asarray(green, dtype=np.complex128)
    if G.shape != (len(grid), len(block), len(block)) or not np.all(np.isfinite(G)):
        raise ValueError("green must be finite with shape (len(grid), len(sites), len(sites))")
    G0 = matsubara_noninteracting_green(model, grid, block)
    result = matrix_auxiliary_self_energy(G0, G, conditioning_limit=conditioning_limit)
    one_body = np.asarray(model.E, dtype=np.complex128)[np.ix_(block, block)].copy()
    identity = np.eye(len(block), dtype=np.complex128)
    hybridization = (
        grid.iw[:, None, None] * identity[None, :, :]
        - one_body[None, :, :]
        - np.linalg.inv(G0)
    )
    return MatsubaraSelfEnergy(
        grid=grid,
        sites=block,
        one_body=one_body,
        noninteracting=G0,
        hybridization=hybridization,
        self_energy=result.retarded,
        G0_condition_number=result.G0_condition_number,
        G_condition_number=result.G_condition_number,
        conditioning_limited=result.conditioning_limited,
    )


def matsubara_dyson_green(
    grid: MatsubaraGrid,
    one_body: ArrayLike,
    hybridization: ArrayLike,
    self_energy: ArrayLike,
) -> ComplexGrid:
    r"""Solve :math:`G=[i\omega_n-h-\Delta(i\omega_n)-\Sigma(i\omega_n)]^{-1}`.

    With the PM hybridization this returns the PM Green function; with the
    target (fitted or DMFT) hybridization it is the reconstructed impurity
    Green function.
    """

    delta = np.asarray(hybridization, dtype=np.complex128)
    sigma = np.asarray(self_energy, dtype=np.complex128)
    h = np.asarray(one_body, dtype=np.complex128)
    if (
        delta.ndim != 3
        or delta.shape[0] != len(grid)
        or delta.shape[1] != delta.shape[2]
        or sigma.shape != delta.shape
        or h.shape != delta.shape[1:]
    ):
        raise ValueError("one_body, hybridization, and self_energy have incompatible shapes")
    if not all(np.all(np.isfinite(value)) for value in (delta, sigma, h)):
        raise ValueError("Dyson inputs must be finite")
    identity = np.eye(h.shape[0], dtype=np.complex128)
    inverse = grid.iw[:, None, None] * identity[None, :, :] - h[None, :, :] - delta - sigma
    return np.linalg.solve(inverse, np.broadcast_to(identity, inverse.shape))


__all__ = [
    "MatsubaraSelfEnergy",
    "impurity_sites",
    "matsubara_dyson_green",
    "matsubara_noninteracting_green",
    "matsubara_self_energy",
    "noninteracting_retarded_green",
]

r"""Matsubara-axis impurity Green functions and self-energies.

For a pseudomode (PM) model whose bath represents an equilibrium bath at
inverse temperature ``beta``, the impurity retarded function :math:`G^R(z)`
is analytic for :math:`\operatorname{Im} z>0`.  Its values at the fermionic
Matsubara frequencies :math:`\omega_n=(2n+1)\pi/\beta` give the imaginary-axis
Green function,

.. math::

   G(i\omega_n)=G^R(i\omega_n)\quad(\omega_n>0),\qquad
   G(i\omega_n)=\big[G^R(-i\omega_n)\big]^\dagger\quad(\omega_n<0),

so a Matsubara grid costs one shift :math:`z=i|\omega_n|` per distinct
:math:`|\omega_n|` in the same shift-invariant Krylov space used for
real-axis spectra.  The self-energy is the auxiliary PM Dyson self-energy on
the impurity block ``S``,

.. math::

   \Sigma(i\omega_n)=G_{0}^{-1}(i\omega_n)-G^{-1}(i\omega_n),\qquad
   G_{0}^R(z)=\big[(z-E+i\Lambda)^{-1}\big]_{SS},\qquad
   \Lambda=\Gamma_{\rm loss}+\Gamma_{\rm gain},

which is exact block elimination of the bath whenever every interacting site
lies in ``S``.  This is the quantity an imaginary-time DMFT loop consumes;
:func:`matsubara_dyson_green` inserts it into any target hybridization, for
example the continuous bath that the pseudomodes were fitted to.

Grids are either explicit (indices, frequencies, or a symmetric ``n_iw``
window) or the fermionic discrete Lehmann representation (DLR) grid for a
given ``beta``, accuracy ``eps`` and cutoff :math:`\Lambda=\beta\omega_{\max}`.
DLR grids use the optional ``pydlr`` package (``pip install "c3l[dlr]"``);
the ``pydlr.dlr`` object is kept on the grid for imaginary-time transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import FermionicLindbladModel

ComplexGrid = NDArray[np.complex128]

_MATSUBARA_TOLERANCE = 1.0e-8


@dataclass(frozen=True, slots=True, eq=False)
class MatsubaraGrid:
    r"""Fermionic Matsubara points :math:`i\omega_n`, :math:`\omega_n=(2n+1)\pi/\beta`.

    ``indices`` are the integers ``n`` in the caller's order, which is kept in
    every array evaluated on the grid.  ``upper_shifts`` are the distinct
    points :math:`z=i|\omega_n|` at which a retarded solver must be evaluated;
    :meth:`from_upper` maps those values back onto the grid.
    """

    beta: float
    indices: NDArray[np.int64]
    kind: str = "explicit"
    eps: float | None = None
    lamb: float | None = None
    dlr: Any = field(default=None, repr=False)
    _upper_odd: NDArray[np.int64] = field(init=False, repr=False)
    _upper_index: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        beta = float(self.beta)
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be finite and positive")
        raw = np.asarray(self.indices)
        if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("indices must be a nonempty integer vector")
        indices = raw.astype(np.int64)
        if np.unique(indices).size != indices.size:
            raise ValueError("Matsubara indices must be unique")
        if self.kind not in ("explicit", "dlr"):
            raise ValueError("kind must be 'explicit' or 'dlr'")
        indices.setflags(write=False)
        odd, inverse = np.unique(np.abs(2 * indices + 1), return_inverse=True)
        odd.setflags(write=False)
        inverse = inverse.astype(np.int64)
        inverse.setflags(write=False)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "_upper_odd", odd)
        object.__setattr__(self, "_upper_index", inverse)

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def omega_n(self) -> NDArray[np.float64]:
        """Real Matsubara frequencies :math:`\\omega_n` in grid order."""

        return (2 * self.indices + 1) * np.pi / self.beta

    @property
    def iw(self) -> ComplexGrid:
        """Complex Matsubara points :math:`i\\omega_n` in grid order."""

        return 1j * self.omega_n

    @property
    def upper_shifts(self) -> ComplexGrid:
        r"""Distinct upper-half-plane points :math:`i|\omega_n|`, increasing."""

        return 1j * self._upper_odd * np.pi / self.beta

    def from_upper(self, values: ArrayLike) -> ComplexGrid:
        r"""Map retarded values at :attr:`upper_shifts` onto the grid.

        ``values`` has shape ``(len(upper_shifts),)`` for a scalar function
        or ``(len(upper_shifts), m, m)`` for a matrix.  Negative frequencies
        receive the conjugate transpose,
        :math:`G(-i\omega)=G(i\omega)^\dagger`.
        """

        upper = np.asarray(values, dtype=np.complex128)
        if upper.ndim not in (1, 3) or upper.shape[0] != self._upper_odd.size:
            raise ValueError("values must have shape (len(upper_shifts),) or (len(upper_shifts), m, m)")
        if upper.ndim == 3 and upper.shape[1] != upper.shape[2]:
            raise ValueError("matrix values must be square")
        out = upper[self._upper_index].copy()
        negative = self.indices < 0
        if upper.ndim == 1:
            out[negative] = out[negative].conj()
        else:
            out[negative] = out[negative].conj().swapaxes(1, 2)
        return out


def matsubara_grid(
    beta: float,
    *,
    indices: ArrayLike | None = None,
    omega_n: ArrayLike | None = None,
    n_iw: int | None = None,
) -> MatsubaraGrid:
    r"""Build an explicit fermionic Matsubara grid.

    Give exactly one of ``indices`` (integers ``n``), ``omega_n`` (real
    frequencies, or the purely imaginary points :math:`i\omega_n`), or
    ``n_iw`` (the symmetric window ``n = -n_iw, ..., n_iw - 1``, as in TRIQS
    ``MeshImFreq``).  Frequencies must lie on the fermionic lattice of this
    ``beta`` to relative accuracy ``1e-8``.
    """

    given = [value is not None for value in (indices, omega_n, n_iw)]
    if sum(given) != 1:
        raise ValueError("give exactly one of indices, omega_n, or n_iw")
    inverse_temperature = float(beta)
    if not np.isfinite(inverse_temperature) or inverse_temperature <= 0.0:
        raise ValueError("beta must be finite and positive")
    if n_iw is not None:
        if isinstance(n_iw, bool) or not isinstance(n_iw, (int, np.integer)) or n_iw <= 0:
            raise ValueError("n_iw must be a positive integer")
        return MatsubaraGrid(inverse_temperature, np.arange(-int(n_iw), int(n_iw)))
    if indices is not None:
        return MatsubaraGrid(inverse_temperature, np.asarray(indices))
    frequency = np.asarray(omega_n)
    if np.iscomplexobj(frequency):
        if np.any(np.abs(frequency.real) > _MATSUBARA_TOLERANCE * np.maximum(np.abs(frequency), 1.0)):
            raise ValueError("complex Matsubara points must be purely imaginary")
        frequency = frequency.imag
    frequency = np.asarray(frequency, dtype=np.float64)
    if frequency.ndim != 1 or frequency.size == 0 or not np.all(np.isfinite(frequency)):
        raise ValueError("omega_n must be a finite nonempty vector")
    odd = frequency * inverse_temperature / np.pi
    nearest = np.rint(odd)
    if (
        np.any(np.abs(odd - nearest) > _MATSUBARA_TOLERANCE * np.maximum(np.abs(odd), 1.0))
        or np.any(nearest.astype(np.int64) % 2 == 0)
    ):
        raise ValueError("omega_n are not fermionic Matsubara frequencies for this beta")
    return MatsubaraGrid(inverse_temperature, (nearest.astype(np.int64) - 1) // 2)


def dlr_matsubara_grid(
    beta: float,
    eps: float,
    *,
    lamb: float | None = None,
    wmax: float | None = None,
) -> MatsubaraGrid:
    r"""Fermionic DLR Matsubara nodes for ``beta``, accuracy ``eps`` and cutoff.

    The DLR needs a dimensionless cutoff :math:`\Lambda=\beta\omega_{\max}`,
    where :math:`\omega_{\max}` bounds the support of the impurity spectral
    function (bandwidth plus interaction scale).  Give either ``lamb`` or
    ``wmax``.  Nodes keep ``pydlr``'s order so ``grid.dlr`` transforms
    (``dlr_from_matsubara``, ``tau_from_dlr``, ...) apply directly to arrays
    evaluated on the grid.
    """

    if (lamb is None) == (wmax is None):
        raise ValueError("give exactly one of lamb or wmax")
    inverse_temperature = float(beta)
    if not np.isfinite(inverse_temperature) or inverse_temperature <= 0.0:
        raise ValueError("beta must be finite and positive")
    cutoff = float(lamb) if lamb is not None else inverse_temperature * float(wmax)
    accuracy = float(eps)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("lamb (or beta * wmax) must be finite and positive")
    if not np.isfinite(accuracy) or not 0.0 < accuracy < 1.0:
        raise ValueError("eps must lie in (0, 1)")
    try:
        from pydlr import dlr
    except ImportError as exc:  # pragma: no cover - exercised only without pydlr
        raise ImportError(
            'DLR grids require pydlr; install it with pip install "c3l[dlr]"'
        ) from exc
    basis = dlr(lamb=cutoff, eps=accuracy, xi=-1)
    return MatsubaraGrid(
        inverse_temperature,
        np.asarray(basis.dlrmf, dtype=np.int64).reshape(-1),
        kind="dlr",
        eps=accuracy,
        lamb=cutoff,
        dlr=basis,
    )


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

    from .spectra.reconstruction import matrix_auxiliary_self_energy

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


@dataclass(frozen=True, slots=True, eq=False)
class MatsubaraGreen:
    """Impurity G(iω_n) from one response operator (axes ``n, sink, source``)."""

    grid: MatsubaraGrid
    sites: tuple[int, ...]
    green: ComplexGrid
    upper: Any

    @property
    def max_relative_residual(self) -> float:
        """Largest Arnoldi relative residual over sources, channels and shifts."""

        return float(
            max(
                np.max(self.upper.addition_relative_residuals),
                np.max(self.upper.removal_relative_residuals),
            )
        )


def component_matsubara_green(
    operator: object,
    sources: Sequence[Any],
    grid: MatsubaraGrid,
    *,
    maximum_krylov_dimension: int = 320,
    residual_tolerance: float = 1.0e-10,
    minimum_krylov_dimension: int = 16,
    check_interval: int = 8,
) -> MatsubaraGreen:
    """Evaluate the EOM impurity Green function on a Matsubara grid.

    ``operator`` is any exact response action (``ComponentSigmaOperator``,
    ``FullS135ResponseOperator`` or their JAX counterparts) and ``sources``
    are ``component_eom_sources`` for the impurity sites.  One Arnoldi space
    per source and channel is shared by all :attr:`MatsubaraGrid.upper_shifts`.
    """

    from .cc.component_response import component_retarded_matrix_green_shifts

    if not isinstance(grid, MatsubaraGrid):
        raise TypeError("grid must be a MatsubaraGrid")
    upper = component_retarded_matrix_green_shifts(
        operator,
        sources,
        grid.upper_shifts,
        maximum_krylov_dimension=maximum_krylov_dimension,
        residual_tolerance=residual_tolerance,
        minimum_krylov_dimension=minimum_krylov_dimension,
        check_interval=check_interval,
    )
    return MatsubaraGreen(grid=grid, sites=upper.sites, green=grid.from_upper(upper.retarded), upper=upper)


@dataclass(frozen=True, slots=True, eq=False)
class MatsubaraImpuritySolution:
    r"""CCSD + EOM impurity solution on a Matsubara grid.

    ``green`` and ``self_energy`` have axes ``(n, a, b)`` over ``sites`` in
    the model's block basis (one spin block in spin mode).  Use
    :meth:`dyson` to insert ``self_energy`` into another hybridization.
    """

    grid: MatsubaraGrid
    sites: tuple[int, ...]
    rank: str
    response_spin: int
    green: ComplexGrid
    self_energy: ComplexGrid
    noninteracting: ComplexGrid
    hybridization: ComplexGrid
    one_body: NDArray[np.complex128]
    amplitudes: NDArray[np.complex128]
    static_residual: float | None
    response: MatsubaraGreen
    dyson_terms: MatsubaraSelfEnergy

    def dyson(self, hybridization: ArrayLike) -> ComplexGrid:
        r""":math:`[i\omega_n-h-\Delta-\Sigma]^{-1}` for a target ``hybridization``."""

        return matsubara_dyson_green(self.grid, self.one_body, hybridization, self.self_energy)


def solve_matsubara(
    model: Any,
    grid: MatsubaraGrid,
    *,
    sites: Sequence[int] | None = None,
    rank: str = "S13",
    response_spin: int = 0,
    amplitudes: ArrayLike | None = None,
    policy: Any = None,
    platform: str | None = None,
    production: bool = False,
    maximum_krylov_dimension: int = 320,
    residual_tolerance: float = 1.0e-10,
    minimum_krylov_dimension: int = 16,
    check_interval: int = 8,
) -> MatsubaraImpuritySolution:
    """Solve CCSD and the EOM response, returning G(iω_n) and Σ(iω_n).

    ``model`` is a :class:`~c3l.FermionicLindbladModel` (compiled with the
    default full-HF reference) or an already compiled
    ``ComponentStaticProblem`` with a custom reference.  Converged
    ``amplitudes`` skip the static solve.  ``rank`` selects the ``"S13"``
    retained or ``"S135"`` response space.  ``platform=None`` uses the NumPy
    kernels; ``"cpu"`` or ``"gpu"`` uses the JAX static solver and response
    action (JAX needs ``JAX_ENABLE_X64``).  In spin mode, ``response_spin``
    selects the spin block.
    """

    from .cc.component_response import ComponentSigmaOperator
    from .cc.component_solver import solve_component_ccsd
    from .cc.component_tensor import (
        ComponentEOMProblem,
        ComponentStaticProblem,
        component_eom_context,
        component_eom_sources,
        component_static_residual,
    )
    from .cc.eom import eom_ccsd_problem
    from .cc.factorized import factorized_cc_problem

    if not isinstance(grid, MatsubaraGrid):
        raise TypeError("grid must be a MatsubaraGrid")
    if rank not in ("S13", "S135"):
        raise ValueError("rank must be 'S13' or 'S135'")
    if platform not in (None, "cpu", "gpu"):
        raise ValueError("platform must be None, 'cpu', or 'gpu'")
    if isinstance(model, FermionicLindbladModel):
        static = ComponentStaticProblem.compile(
            factorized_cc_problem(model, "CCSD", build_next_rank_masks=False)
        )
    elif isinstance(model, ComponentStaticProblem):
        static = model
    else:
        raise TypeError("model must be a FermionicLindbladModel or ComponentStaticProblem")
    physical = static.oracle.model
    block = impurity_sites(physical, sites)

    if amplitudes is None:
        if platform is None:
            solution = solve_component_ccsd(static, policy=policy)
        else:
            from .cc.component_jax import solve_component_ccsd_jax

            solution = solve_component_ccsd_jax(
                static, policy=policy, platform=platform, production=production
            )
        if not solution.converged:
            raise RuntimeError(
                f"static CCSD did not converge (residual {solution.packed_residual_norm:.3e})"
            )
        converged = np.asarray(solution.amplitudes, dtype=np.complex128)
        static_residual: float | None = float(solution.packed_residual_norm)
    else:
        converged = np.asarray(amplitudes, dtype=np.complex128)
        retained = component_static_residual(static, converged).retained
        static_residual = float(np.linalg.norm(static.oracle.tilde.pack(retained, check=False)))

    eom = eom_ccsd_problem(static.oracle, response_spin=response_spin, build_next_rank_masks=False)
    context = component_eom_context(ComponentEOMProblem.compile(eom, static, charge=1), converged)
    sources = tuple(component_eom_sources(context, site=site) for site in block)
    if rank == "S13":
        if platform is None:
            operator: object = ComponentSigmaOperator(context)
        else:
            from .cc.component_jax import JaxRetainedEOMOperator

            operator = JaxRetainedEOMOperator.from_cpu(
                context, platform=platform, production=production
            )
    else:
        from .cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator

        operator = FullS135ResponseOperator(CompiledFullS135Action(context))
        if platform is not None:
            from .cc.s135_jax import JaxFullS135ResponseOperator

            operator = JaxFullS135ResponseOperator.from_cpu(
                operator, platform=platform, production=production
            )

    response = component_matsubara_green(
        operator,
        sources,
        grid,
        maximum_krylov_dimension=maximum_krylov_dimension,
        residual_tolerance=residual_tolerance,
        minimum_krylov_dimension=minimum_krylov_dimension,
        check_interval=check_interval,
    )
    terms = matsubara_self_energy(physical, grid, response.green, sites=block)
    return MatsubaraImpuritySolution(
        grid=grid,
        sites=block,
        rank=rank,
        response_spin=int(response_spin),
        green=response.green,
        self_energy=terms.self_energy,
        noninteracting=terms.noninteracting,
        hybridization=terms.hybridization,
        one_body=terms.one_body,
        amplitudes=converged,
        static_residual=static_residual,
        response=response,
        dyson_terms=terms,
    )


__all__ = [
    "MatsubaraGreen",
    "MatsubaraGrid",
    "MatsubaraImpuritySolution",
    "MatsubaraSelfEnergy",
    "component_matsubara_green",
    "dlr_matsubara_grid",
    "impurity_sites",
    "matsubara_dyson_green",
    "matsubara_grid",
    "matsubara_noninteracting_green",
    "matsubara_self_energy",
    "noninteracting_retarded_green",
    "solve_matsubara",
]

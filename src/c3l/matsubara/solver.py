r"""EOM response on a Matsubara grid and the end-to-end CCSD impurity solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..models import FermionicLindbladModel
from .dyson import MatsubaraSelfEnergy, impurity_sites, matsubara_dyson_green, matsubara_self_energy
from .grid import ComplexGrid, MatsubaraGrid


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

    from ..cc.component_response import component_retarded_matrix_green_shifts

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

    from ..cc.component_response import ComponentSigmaOperator
    from ..cc.component_solver import solve_component_ccsd
    from ..cc.component_tensor import (
        ComponentEOMProblem,
        ComponentStaticProblem,
        component_eom_context,
        component_eom_sources,
        component_static_residual,
    )
    from ..cc.eom import eom_ccsd_problem
    from ..cc.factorized import factorized_cc_problem

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
            from ..cc.component_jax import solve_component_ccsd_jax

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
            from ..cc.component_jax import JaxRetainedEOMOperator

            operator = JaxRetainedEOMOperator.from_cpu(
                context, platform=platform, production=production
            )
    else:
        from ..cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator

        operator = FullS135ResponseOperator(CompiledFullS135Action(context))
        if platform is not None:
            from ..cc.s135_jax import JaxFullS135ResponseOperator

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
    "MatsubaraImpuritySolution",
    "component_matsubara_green",
    "solve_matsubara",
]

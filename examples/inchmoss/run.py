"""Erpenbeck et al. (2024) Eq. (16) and Fig. 6 benchmark inputs."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from ..parameters import parameters
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, roots_legendre

from c3l.fitting import fit_fermionic_bath
from c3l.cc.component_solver import ComponentCCSolution, solve_component_ccsd
from c3l.cc.component_tensor import (
    ComponentEOMContext,
    ComponentEOMProblem,
    ComponentEOMSources,
    ComponentStaticProblem,
    component_eom_context,
    component_eom_sources,
)
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import FactorizedCCProblem, factorized_cc_problem
from c3l.cc.newton_krylov import NewtonKrylovPolicy
from c3l.interactions import SparseTwoBodyInteraction
from c3l.pseudomode import (
    PseudomodeRealization,
    matrix_pseudomode_model,
    realization_bcf,
)


RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Branch = Literal["lesser", "greater"]
ROOT = Path.cwd() / "inchmoss"
LITERAL_RESULTS = ROOT.parent / "results" / "inchmoss_fig6_printed_eq16_v1"
CORRECTED_RESULTS = ROOT.parent / "results" / "inchmoss_fig6_ccsd_eom_v1"
RESULTS = CORRECTED_RESULTS


@dataclass(frozen=True, slots=True)
class Case:
    """Literal dimensionless parameters in units of ``v=1``."""

    beta: float
    epsilon_0: float
    gamma: float
    U: float
    V: float
    W: float
    gamma_prime: float
    nu: float
    omega_c: float


@dataclass(frozen=True, slots=True)
class ScalarBathFit:
    """Accepted scalar equilibrium pseudomode pair and direct fit errors."""

    m: int
    lesser: PseudomodeRealization
    greater: PseudomodeRealization
    fit_end: float
    training_points: int
    validation_points: int
    function_evaluations: int
    fit_wall_seconds: float
    relative_l2_error: float
    maximum_absolute_error: float
    equal_time_relative_error: float


def case() -> Case:
    """Return the fixed Fig. 6 parameter set."""

    p = parameters("inchmoss")
    return Case(beta=p["beta"], epsilon_0=p["epsilon_0"], U=p["U"],
                V=p["V"], W=p["V_prime"], gamma=p["nu"],
                gamma_prime=p["nu_prime"], nu=p["bath"]["edge_steepness"],
                omega_c=p["bath"]["omega_c"])


def impurity_one_body() -> ComplexArray:
    """Return the two-orbital one-body block in the paper's orbital basis."""

    parameters = case()
    return np.asarray(
        (
            (parameters.epsilon_0, parameters.gamma),
            (parameters.gamma, parameters.epsilon_0),
        ),
        dtype=np.complex128,
    )


def caption_orbital_hybridization_matrix() -> ComplexArray:
    r"""Return the per-lead Fig. 6 caption matrix ``[[3,1],[1,3]]/4``."""

    return np.asarray(parameters("inchmoss")["bath"]["matrix_per_lead"], dtype=np.complex128)


def orbital_hybridization_matrix() -> ComplexArray:
    """Return the sum of the two identical equilibrium lead matrices."""

    return parameters("inchmoss")["bath"]["identical_leads"] * caption_orbital_hybridization_matrix()


def wide_band_shape(omega: ArrayLike) -> RealArray:
    r"""Evaluate the centered smooth cutoff with ``D(0) approximately 1``."""

    parameters = case()
    values = np.asarray(omega, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("omega must be finite")
    logarithm = np.logaddexp(
        0.0, parameters.nu * (values - parameters.omega_c)
    ) + np.logaddexp(0.0, -parameters.nu * (values + parameters.omega_c))
    return np.asarray(np.exp(-logarithm), dtype=np.float64)


def scalar_spectral_density(omega: ArrayLike) -> RealArray:
    r"""Return ``D(omega)/(2 pi)`` for the conventional paper ``Gamma``."""

    return np.asarray(wide_band_shape(omega) / (2.0 * np.pi), dtype=np.float64)


def scalar_bcf(
    time: ArrayLike,
    branch: Branch,
    *,
    quadrature_order: int = 2048,
) -> ComplexArray:
    r"""Evaluate the equilibrium scalar lesser/greater bath correlation."""

    values = np.asarray(time, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("time must be a finite nonempty vector")
    if branch not in ("lesser", "greater"):
        raise ValueError("branch must be 'lesser' or 'greater'")
    if quadrature_order < 64:
        raise ValueError("quadrature_order must be at least 64")
    nodes, weights = roots_legendre(quadrature_order)
    integration_bound = 30.0
    omega = integration_bound * nodes
    measure = integration_bound * weights * scalar_spectral_density(omega)
    fermi = expit(-case().beta * omega)
    if branch == "lesser":
        occupation = fermi
        phase = np.exp(1j * np.outer(omega, values))
    else:
        occupation = 1.0 - fermi
        phase = np.exp(-1j * np.outer(omega, values))
    return np.asarray((measure * occupation) @ phase, dtype=np.complex128)


def fit_scalar_bath(
    m: int,
    *,
    training_points: int = 251,
    validation_points: int = 1001,
    max_nfev: int = 3000,
) -> ScalarBathFit:
    """Fit both scalar bath correlations with RealTimeBath.

    Use the archived bath file to reproduce a previously reported curve;
    a new fit can give different pseudomode parameters.
    """

    if training_points < 3 or validation_points < 3 or max_nfev < 1:
        raise ValueError("fit grids need at least three points and max_nfev > 0")
    fit_end = 5.0
    time = np.linspace(0.0, fit_end, training_points)
    started = perf_counter()
    fitted = fit_fermionic_bath(
        time,
        scalar_bcf(time, "lesser"),
        scalar_bcf(time, "greater"),
        n_modes=m,
        max_nfev=max_nfev,
    )
    elapsed = perf_counter() - started
    validation_time = np.linspace(0.0, fit_end, validation_points)
    target = np.asarray([scalar_bcf(validation_time, branch)
                         for branch in ("lesser", "greater")])
    observed = np.asarray([
        realization_bcf(validation_time, getattr(fitted, branch), branch)
        for branch in ("lesser", "greater")
    ])
    difference = observed - target
    relative_l2 = np.sqrt(
        np.sum(np.trapezoid(abs(difference)**2, x=validation_time))
        / np.sum(np.trapezoid(abs(target)**2, x=validation_time))
    )
    evaluations = sum(int(d.get("details", {}).get("optimization_nfev", 0))
                      for d in fitted.diagnostics.values())
    return ScalarBathFit(
        m=m,
        lesser=fitted.lesser,
        greater=fitted.greater,
        fit_end=fit_end,
        training_points=training_points,
        validation_points=validation_points,
        function_evaluations=evaluations,
        fit_wall_seconds=elapsed,
        relative_l2_error=float(relative_l2),
        maximum_absolute_error=float(np.max(abs(difference))),
        equal_time_relative_error=float(np.linalg.norm(difference[:, 0]) / np.linalg.norm(target[:, 0])),
    )


def save_bath_fit(fit: ScalarBathFit) -> Path:
    """Save one compact accepted scalar fit in the benchmark directory."""

    path = ROOT / "data" / f"bath_N{fit.m}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        m=np.int64(fit.m),
        H_lesser=fit.lesser.H,
        damping_lesser=fit.lesser.damping,
        g_lesser=fit.lesser.g,
        H_greater=fit.greater.H,
        damping_greater=fit.greater.damping,
        g_greater=fit.greater.g,
        fit_end=np.float64(fit.fit_end),
        training_points=np.int64(fit.training_points),
        validation_points=np.int64(fit.validation_points),
        function_evaluations=np.int64(fit.function_evaluations),
        fit_wall_seconds=np.float64(fit.fit_wall_seconds),
        relative_l2_error=np.float64(fit.relative_l2_error),
        maximum_absolute_error=np.float64(fit.maximum_absolute_error),
        equal_time_relative_error=np.float64(fit.equal_time_relative_error),
    )
    return path


def load_bath_fit(m: int = 5, *, path: Path | None = None) -> ScalarBathFit:
    """Load one accepted scalar fit without rerunning its optimization."""

    path = ROOT / "data" / f"bath_N{int(m)}.npz" if path is None else Path(path)
    with np.load(path, allow_pickle=False) as data:
        return ScalarBathFit(
            m=int(data["m"]),
            lesser=PseudomodeRealization(
                np.asarray(data["H_lesser"]),
                np.asarray(data["damping_lesser"]),
                np.asarray(data["g_lesser"]),
            ),
            greater=PseudomodeRealization(
                np.asarray(data["H_greater"]),
                np.asarray(data["damping_greater"]),
                np.asarray(data["g_greater"]),
            ),
            fit_end=float(data["fit_end"]),
            training_points=int(data["training_points"]),
            validation_points=int(data["validation_points"]),
            function_evaluations=int(data["function_evaluations"]),
            fit_wall_seconds=float(data["fit_wall_seconds"]),
            relative_l2_error=float(data["relative_l2_error"]),
            maximum_absolute_error=float(data["maximum_absolute_error"]),
            equal_time_relative_error=float(data["equal_time_relative_error"]),
        )


def build_model(
    fit: ScalarBathFit | None = None,
    *,
    corrected_convention: bool = True,
):
    """Assemble the literal model or the explicitly corrected variant."""

    accepted = load_bath_fit() if fit is None else fit
    return matrix_pseudomode_model(
        impurity_one_body(),
        orbital_hybridization_matrix(),
        accepted.lesser,
        accepted.greater,
        corrected_interaction() if corrected_convention else interaction(),
        units="v",
    )


def build_cc_problem(
    fit: ScalarBathFit | None = None,
    *,
    corrected_convention: bool = True,
) -> FactorizedCCProblem:
    """Build CCSD in the existing default self-consistent full-HF frame."""

    return factorized_cc_problem(
        build_model(fit, corrected_convention=corrected_convention),
        "CCSD",
        build_next_rank_masks=False,
    )


def build_static_problem(
    fit: ScalarBathFit | None = None,
    *,
    corrected_convention: bool = True,
) -> ComponentStaticProblem:
    """Compile the existing component CCSD equations for the accepted bath."""

    return ComponentStaticProblem.compile(
        build_cc_problem(fit, corrected_convention=corrected_convention)
    )


def direct_static_policy() -> NewtonKrylovPolicy:
    """Use the established direct-solve policy with a physical 1e-8 gate."""

    return NewtonKrylovPolicy(
        tolerance=1.0e-8,
        maximum_newton_iterations=12,
        gmres_restart=80,
        gmres_maximum_restart_cycles=2,
        minimum_gmres_relative_tolerance=1.0e-8,
        maximum_gmres_relative_tolerance=0.5,
        maximum_line_search_steps=4,
        forcing_strategy="eisenstat_walker",
        initial_gmres_relative_tolerance=0.5,
    )


def solve_static_cpu(
    *,
    corrected_convention: bool = True,
    output_directory: Path | None = None,
) -> ComponentCCSolution:
    """Attempt the physical ``s=1`` static root directly on CPU."""

    if output_directory is None:
        output_directory = (
            CORRECTED_RESULTS if corrected_convention else LITERAL_RESULTS
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    problem = build_static_problem(corrected_convention=corrected_convention)
    latest = output_directory / "static_cpu_direct_latest.npz"

    def checkpoint(iteration: int, amplitudes: ComplexArray, residual: float) -> None:
        np.savez_compressed(
            latest,
            amplitudes=amplitudes,
            iteration=np.int64(iteration),
            packed_residual=np.float64(residual),
        )
        print(
            f"accepted Newton step {iteration}: packed residual {residual:.12e}",
            flush=True,
        )

    policy = direct_static_policy()
    solution = solve_component_ccsd(
        problem,
        policy=policy,
        compute_rank6=False,
        accepted_step_callback=checkpoint,
    )
    np.savez_compressed(
        output_directory / "static_cpu_direct.npz",
        amplitudes=solution.amplitudes,
        variables=solution.variables,
        converged=np.bool_(solution.converged),
        packed_residual=np.float64(solution.packed_residual_norm),
        residual_norm=np.float64(solution.residual_norm),
        reference_residual=np.float64(solution.reference_residual),
    )
    summary = {
        "route": "direct_s1_from_full_hf",
        "interaction_convention": (
            "decoupling_corrected" if corrected_convention else "literal_printed_eq16"
        ),
        "converged": solution.converged,
        "packed_residual": solution.packed_residual_norm,
        "residual_norm": solution.residual_norm,
        "reference_residual": solution.reference_residual,
        "iterations": solution.iterations,
        "residual_evaluations": solution.residual_evaluations,
        "jvp_evaluations": solution.jvp_evaluations,
        "wall_seconds": solution.wall_seconds,
        "peak_rss_kib": solution.peak_rss_kib,
        "dimensions": {
            "complex_amplitudes": problem.oracle.num_amplitudes,
            "independent_real_variables": problem.oracle.num_variables,
        },
        "factorized_records": len(problem.oracle.products),
        "policy": policy.to_dict(),
        "history": [asdict(step) for step in solution.history],
    }
    (output_directory / "static_cpu_direct.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return solution


def load_static_amplitudes(
    path: Path | None = None,
    *,
    corrected_convention: bool = True,
) -> ComplexArray:
    """Load the accepted direct static amplitudes."""

    if path is None:
        root = CORRECTED_RESULTS if corrected_convention else LITERAL_RESULTS
        path = root / "static_cpu_direct.npz"
    with np.load(path, allow_pickle=False) as data:
        if not bool(data["converged"]):
            raise RuntimeError("the saved direct static root is not converged")
        return np.asarray(data["amplitudes"], dtype=np.complex128)


def build_eom_context(
    *,
    spin: int = 0,
    charge: int = 1,
    corrected_convention: bool = True,
) -> tuple[ComponentEOMContext, tuple[ComponentEOMSources, ...]]:
    """Build the accepted two-source retained EOM context."""

    static = build_static_problem(corrected_convention=corrected_convention)
    eom = eom_ccsd_problem(
        static.oracle,
        response_spin=int(spin),
        build_next_rank_masks=False,
    )
    problem = ComponentEOMProblem.compile(eom, static, charge=int(charge))
    context = component_eom_context(
        problem,
        load_static_amplitudes(corrected_convention=corrected_convention),
    )
    sources = tuple(component_eom_sources(context, site=site) for site in (0, 1))
    return context, sources


def _interaction(spin_flip_coefficient: float) -> SparseTwoBodyInteraction:
    """Assemble Eq. (16) with an explicit canonical spin-flip coefficient."""

    parameters = case()
    terms: list[tuple[int, int, int, int, complex]] = []

    def orbital(site: int, spin: int) -> int:
        return 2 * site + spin

    # U sum_ij n_i,up n_j,down.
    for i in (0, 1):
        for j in (0, 1):
            p = orbital(i, 0)
            q = orbital(j, 1)
            terms.append((p, q, p, q, complex(parameters.U)))

    # n_a a_b^dag a_c = -a_a^dag a_b^dag a_a a_c.
    for i in (0, 1):
        for spin in (0, 1):
            a = orbital(i, spin)
            opposite = 1 - spin
            for source, target in ((1, 0), (0, 1)):
                b = orbital(target, opposite)
                c = orbital(source, opposite)
                terms.append((a, b, c, a, complex(-parameters.V)))

    # a_a^dag a_b a_c^dag a_d = -a_a^dag a_c^dag a_b a_d.
    for spin in (0, 1):
        opposite = 1 - spin
        a = orbital(0, spin)
        b = orbital(0, opposite)
        c = orbital(1, opposite)
        d = orbital(1, spin)
        terms.append((a, c, d, b, complex(spin_flip_coefficient)))

    # Pair hopping is printed as a_0u^dag a_0d^dag a_1u a_1d + h.c.
    up_0, down_0, up_1, down_1 = range(4)
    terms.extend(
        (
            (up_0, down_0, down_1, up_1, complex(parameters.gamma_prime)),
            (down_1, up_1, up_0, down_0, complex(parameters.gamma_prime)),
        )
    )
    return SparseTwoBodyInteraction(4, tuple(terms))


def printed_interaction() -> SparseTwoBodyInteraction:
    r"""Return the literal printed ``+W`` operator after normal ordering."""

    return interaction()


def interaction() -> SparseTwoBodyInteraction:
    r"""Return the primary literal printed Eq. (16) interaction."""

    return _interaction(-case().W)


def corrected_interaction() -> SparseTwoBodyInteraction:
    r"""Return the decoupling-consistent corrected-convention interaction."""

    return _interaction(case().W)


__all__ = [
    "Case",
    "ScalarBathFit",
    "build_cc_problem",
    "build_eom_context",
    "build_model",
    "build_static_problem",
    "caption_orbital_hybridization_matrix",
    "case",
    "corrected_interaction",
    "fit_scalar_bath",
    "impurity_one_body",
    "interaction",
    "load_bath_fit",
    "load_static_amplitudes",
    "orbital_hybridization_matrix",
    "printed_interaction",
    "save_bath_fit",
    "solve_static_cpu",
    "scalar_bcf",
    "scalar_spectral_density",
    "wide_band_shape",
]

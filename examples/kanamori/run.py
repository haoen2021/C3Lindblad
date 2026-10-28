"""Two-orbital full Kanamori interaction with a matrix-valued Bethe bath."""

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
from c3l.ci.reference import full_hartree_fock_reference
from c3l.gaussian_mean_field import solve_ghf
from c3l.interactions import SparseTwoBodyInteraction
from c3l.models import homotopy_model
from c3l.models import spin_lift as expand_spin_diagonal
from c3l.pseudomode import (
    PseudomodeRealization,
    matrix_pseudomode_model,
    realization_bcf,
)


RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Branch = Literal["lesser", "greater"]
OneBodyConvention = Literal["literal_zero", "particle_hole_symmetric", "chemical_potential"]
ROOT = Path.cwd() / "kanamori"
RESULTS_ROOT = ROOT.parent / "results"


@dataclass(frozen=True, slots=True)
class Case:
    """Fixed continuous-Bethe parameters in units of ``t=1``."""

    t: float
    U: float
    J: float
    beta: float
    r: float
    D: float


@dataclass(frozen=True, slots=True)
class ScalarBathFit:
    """One accepted low-temperature scalar pseudomode pair."""

    beta: float
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
    """Return the low-temperature continuous-Bethe endpoint."""

    p = parameters("kanamori")
    return Case(t=p["t"], U=p["U"], J=p["J"], beta=p["beta"][1],
                r=p["r"], D=p["bath_half_bandwidth"])


def particle_hole_symmetric_energy() -> float:
    """Return the half-filled energy -(3U-5J)/2."""

    parameters = case()
    return float(-(3.0 * parameters.U - 5.0 * parameters.J) / 2.0)


def impurity_one_body(convention: OneBodyConvention) -> ComplexArray:
    """Return the requested one-body energy convention."""

    if convention == "literal_zero":
        energy = 0.0
    elif convention == "particle_hole_symmetric":
        energy = particle_hole_symmetric_energy()
    elif convention == "chemical_potential":
        energy = -float(parameters("kanamori")["mu"])
    else:
        raise ValueError("unknown one-body convention")
    return np.asarray(energy * np.eye(2), dtype=np.complex128)


def orbital_hybridization_matrix() -> ComplexArray:
    r"""Return ``t^2 [[1,r],[r,1]]`` in the requested orbital basis."""

    parameters = case()
    return parameters.t**2 * np.asarray(
        ((1.0, parameters.r), (parameters.r, 1.0)), dtype=np.complex128
    )


def semicircle_density(omega: ArrayLike) -> RealArray:
    r"""Evaluate ``2 sqrt(D^2-omega^2)/(pi D^2)`` on its support."""

    values = np.asarray(omega, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("omega must be finite")
    D = case().D
    return np.asarray(
        2.0 * np.sqrt(np.maximum(0.0, D**2 - values**2)) / (np.pi * D**2),
        dtype=np.float64,
    )


def bethe_retarded(z: ArrayLike) -> ComplexArray:
    r"""Return the causal Stieltjes transform of the normalized semicircle."""

    values = np.asarray(z, dtype=np.complex128)
    if not np.all(np.isfinite(values)):
        raise ValueError("z must be finite")
    D = case().D
    root = np.sqrt(values - D) * np.sqrt(values + D)
    return np.asarray(2.0 / (values + root), dtype=np.complex128)


def scalar_bcf(
    time: ArrayLike,
    branch: Branch,
    *,
    beta: float | None = None,
    quadrature_order: int = 2048,
) -> ComplexArray:
    """Evaluate the equilibrium semicircular lesser/greater correlation."""

    values = np.asarray(time, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("time must be a finite nonempty vector")
    if branch not in ("lesser", "greater"):
        raise ValueError("branch must be 'lesser' or 'greater'")
    inverse_temperature = case().beta if beta is None else float(beta)
    if not np.isfinite(inverse_temperature) or inverse_temperature <= 0.0:
        raise ValueError("beta must be positive")
    if quadrature_order < 64:
        raise ValueError("quadrature_order must be at least 64")
    nodes, weights = roots_legendre(quadrature_order)
    omega = case().D * nodes
    measure = case().D * weights * semicircle_density(omega)
    measure = measure / np.sum(measure)
    fermi = expit(-inverse_temperature * omega)
    if branch == "lesser":
        occupation = fermi
        phase = np.exp(1j * np.outer(omega, values))
    else:
        occupation = 1.0 - fermi
        phase = np.exp(-1j * np.outer(omega, values))
    return np.asarray((measure * occupation) @ phase, dtype=np.complex128)


def fit_scalar_bath(
    beta: float,
    m: int,
    *,
    training_points: int = 321,
    validation_points: int = 1001,
    max_nfev: int = 3000,
) -> ScalarBathFit:
    """Fit both scalar bath correlations with RealTimeBath.

    Use the archived bath file to reproduce a previously reported curve;
    a new fit can give different pseudomode parameters.
    """

    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive")

    if training_points < 3 or validation_points < 3 or max_nfev < 1:
        raise ValueError("fit grids need at least three points and max_nfev > 0")
    fit_end = beta
    time = np.linspace(0.0, fit_end, training_points)
    started = perf_counter()
    fitted = fit_fermionic_bath(
        time,
        scalar_bcf(time, "lesser", beta=beta),
        scalar_bcf(time, "greater", beta=beta),
        n_modes=m,
        max_nfev=max_nfev,
    )
    elapsed = perf_counter() - started
    validation_time = np.linspace(0.0, fit_end, validation_points)
    target = np.asarray([scalar_bcf(validation_time, branch, beta=beta)
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
        beta=beta,
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


def _bath_path(beta: float, m: int) -> Path:
    beta_label = (
        str(int(beta)) if float(beta).is_integer() else str(beta).replace(".", "p")
    )
    return ROOT / "data" / f"bath_beta{beta_label}_N{m}.npz"


def save_bath_fit(fit: ScalarBathFit) -> Path:
    """Save one compact accepted Bethe fit."""

    path = _bath_path(fit.beta, fit.m)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        beta=np.float64(fit.beta),
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


def load_bath_fit(beta: float, m: int) -> ScalarBathFit:
    """Load one accepted Bethe fit without repeating its optimization."""

    with np.load(_bath_path(beta, m), allow_pickle=False) as data:
        return ScalarBathFit(
            beta=float(data["beta"]),
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


def build_model(beta: float, m: int, convention: OneBodyConvention):
    """Assemble the finite rank-one Bethe model in the paper's orbital basis."""

    fit = load_bath_fit(beta, m)
    return matrix_pseudomode_model(
        impurity_one_body(convention),
        orbital_hybridization_matrix(),
        fit.lesser,
        fit.greater,
        interaction(),
        units="t",
    )


def _regulated_reference(model, dark_reference_rate: float):
    """Solve the full-HF reference with one exactly removed dark regulator."""

    rate = float(dark_reference_rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("dark_reference_rate must be positive")
    dark = np.zeros_like(model.E)
    dark[:2, :2] = 0.5 * np.asarray(((1.0, -1.0), (-1.0, 1.0)))
    reference_loss = np.asarray(model.gamma_loss) + rate * dark
    reference_gain = np.asarray(model.gamma_gain) + rate * dark
    hf = solve_ghf(
        expand_spin_diagonal(model.E),
        model.interaction,
        expand_spin_diagonal(reference_loss),
        expand_spin_diagonal(reference_gain),
        tolerance=1.0e-10,
        max_iterations=400,
        restricted=True,
        terminal_least_squares=True,
    )
    if (
        not hf.converged
        or hf.self_consistency_residual > 1.0e-10
        or hf.lyapunov_residual > 1.0e-10
        or hf.physicality_defect > 1.0e-10
    ):
        raise RuntimeError("the regulated full-HF reference did not converge")
    reference = full_hartree_fock_reference(
        model,
        hf.fock,
        hf.gamma,
        reference_gamma_loss=reference_loss,
        reference_gamma_gain=reference_gain,
    )
    return reference


def build_cc_problem(
    beta: float,
    m: int,
    convention: OneBodyConvention,
    *,
    dark_reference_rate: float = 0.05,
) -> FactorizedCCProblem:
    """Build CCSD with an exactly removed Gaussian dark-channel regulator."""

    model = build_model(beta, m, convention)
    reference = _regulated_reference(model, dark_reference_rate)
    return factorized_cc_problem(
        model,
        "CCSD",
        build_next_rank_masks=False,
        reference=reference,
    )


def build_static_problem(
    beta: float, m: int, convention: OneBodyConvention
) -> ComponentStaticProblem:
    """Compile the component CCSD equations for one fixed endpoint."""

    return ComponentStaticProblem.compile(build_cc_problem(beta, m, convention))


def results_directory(beta: float, convention: OneBodyConvention) -> Path:
    """Return the versioned compact-output directory for one endpoint."""

    beta_label = (
        str(int(beta)) if float(beta).is_integer() else str(beta).replace(".", "p")
    )
    return RESULTS_ROOT / f"kanamori_full_beta{beta_label}_{convention}_v2"


def direct_static_policy() -> NewtonKrylovPolicy:
    """Return the established direct physical-endpoint Newton policy."""

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
    beta: float, m: int, convention: OneBodyConvention
) -> ComponentCCSolution:
    """Attempt one direct ``s=1`` static CCSD endpoint on CPU."""

    output = results_directory(beta, convention)
    output.mkdir(parents=True, exist_ok=True)
    problem = build_static_problem(beta, m, convention)

    def checkpoint(iteration: int, amplitudes: ComplexArray, residual: float) -> None:
        np.savez_compressed(
            output / "static_cpu_direct_latest.npz",
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
        output / "static_cpu_direct.npz",
        amplitudes=solution.amplitudes,
        variables=solution.variables,
        converged=np.bool_(solution.converged),
        packed_residual=np.float64(solution.packed_residual_norm),
        residual_norm=np.float64(solution.residual_norm),
        reference_residual=np.float64(solution.reference_residual),
    )
    summary = {
        "beta": float(beta),
        "modes_per_correlation": int(m),
        "one_body_convention": convention,
        "dark_reference_rate": 0.05,
        "dark_reference_dissipation_removed_exactly": True,
        "route": "direct_s1_from_full_hf",
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
    (output / "static_cpu_direct.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return solution


def continue_static_cpu(
    beta: float,
    m: int,
    convention: OneBodyConvention,
    *,
    interactions: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0),
) -> ComponentCCSolution:
    """Use the bounded frozen-reference homotopy after a failed direct solve."""

    if (
        not interactions
        or interactions[0] != 0.0
        or interactions[-1] != 1.0
        or len(interactions) > 7
        or any(right <= left for left, right in zip(interactions, interactions[1:]))
    ):
        raise ValueError(
            "continuation nodes must increase from 0 to 1 with at most seven nodes"
        )
    output = results_directory(beta, convention)
    output.mkdir(parents=True, exist_ok=True)
    physical = build_model(beta, m, convention)
    template = _regulated_reference(physical, 0.05)
    previous: ComplexArray | None = None
    records: list[dict[str, object]] = []
    final: ComponentCCSolution | None = None
    policy = direct_static_policy()
    for node_index, value in enumerate(interactions):
        node_model = homotopy_model(physical, template.E_up, value)
        node_reference = full_hartree_fock_reference(
            node_model,
            template.fock,
            template.gamma,
            reference_gamma_loss=template.reference_gamma_loss,
            reference_gamma_gain=template.reference_gamma_gain,
        )
        problem = ComponentStaticProblem.compile(
            factorized_cc_problem(
                node_model,
                "CCSD",
                build_next_rank_masks=False,
                reference=node_reference,
            )
        )

        def checkpoint(
            iteration: int, amplitudes: ComplexArray, residual: float
        ) -> None:
            np.savez_compressed(
                output / "static_cpu_continuation_latest.npz",
                amplitudes=amplitudes,
                node=np.float64(value),
                iteration=np.int64(iteration),
                packed_residual=np.float64(residual),
            )
            print(
                f"s={value:g} accepted Newton step {iteration}: "
                f"packed residual {residual:.12e}",
                flush=True,
            )

        solution = solve_component_ccsd(
            problem,
            previous,
            policy=policy,
            compute_rank6=False,
            accepted_step_callback=checkpoint,
        )
        change = (
            0.0
            if previous is None
            else float(np.linalg.norm(solution.amplitudes - previous))
        )
        records.append(
            {
                "node_index": node_index,
                "s": float(value),
                "converged": solution.converged,
                "packed_residual": solution.packed_residual_norm,
                "amplitude_change": change,
                "iterations": solution.iterations,
                "jvp_evaluations": solution.jvp_evaluations,
                "wall_seconds": solution.wall_seconds,
                "history": [asdict(step) for step in solution.history],
            }
        )
        final = solution
        if not solution.converged:
            break
        previous = solution.amplitudes
    assert final is not None
    accepted = bool(
        len(records) == len(interactions)
        and all(bool(record["converged"]) for record in records)
    )
    np.savez_compressed(
        output / "static_cpu_continuation.npz",
        amplitudes=final.amplitudes,
        variables=final.variables,
        converged=np.bool_(accepted),
        packed_residual=np.float64(final.packed_residual_norm),
        residual_norm=np.float64(final.residual_norm),
        reference_residual=np.float64(final.reference_residual),
        interactions=np.asarray(interactions),
    )
    summary = {
        "beta": float(beta),
        "modes_per_correlation": int(m),
        "one_body_convention": convention,
        "route": "bounded_frozen_full_hf_continuation",
        "interactions": list(interactions),
        "converged": accepted,
        "packed_residual": final.packed_residual_norm,
        "nodes": records,
        "policy": policy.to_dict(),
    }
    (output / "static_cpu_continuation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return final


def load_static_amplitudes(beta: float, convention: OneBodyConvention) -> ComplexArray:
    """Load one accepted direct static root."""

    output = results_directory(beta, convention)
    for name in (
        "static_cpu_direct_restart320.npz",
        "static_cpu_direct_restart160.npz",
        "static_cpu_direct.npz",
        "static_cpu_continuation.npz",
    ):
        path = output / name
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            if bool(data["converged"]):
                return np.asarray(data["amplitudes"], dtype=np.complex128)
    raise RuntimeError("no accepted direct or continuation static root exists")


def build_eom_context(
    beta: float,
    m: int,
    convention: OneBodyConvention,
    *,
    spin: int = 0,
    charge: int = 1,
) -> tuple[ComponentEOMContext, tuple[ComponentEOMSources, ...]]:
    """Build the accepted two-source retained EOM context."""

    static = build_static_problem(beta, m, convention)
    eom = eom_ccsd_problem(
        static.oracle,
        response_spin=int(spin),
        build_next_rank_masks=False,
    )
    problem = ComponentEOMProblem.compile(eom, static, charge=int(charge))
    context = component_eom_context(problem, load_static_amplitudes(beta, convention))
    sources = tuple(component_eom_sources(context, site=site) for site in (0, 1))
    return context, sources


def interaction() -> SparseTwoBodyInteraction:
    r"""Return density, spin-exchange and pair-hopping terms with Hund coupling J."""

    parameters = case()
    terms: list[tuple[int, int, int, int, complex]] = []

    def orbital(site: int, spin: int) -> int:
        return 2 * site + spin

    for i in (0, 1):
        down = orbital(i, 1)
        up = orbital(i, 0)
        terms.append((down, up, down, up, complex(parameters.U)))
    for i in (0, 1):
        for j in (0, 1):
            if i == j:
                continue
            down = orbital(i, 1)
            up = orbital(j, 0)
            terms.append(
                (down, up, down, up, complex(parameters.U - 2.0 * parameters.J))
            )
    for spin in (0, 1):
        p = orbital(1, spin)
        q = orbital(0, spin)
        terms.append((p, q, p, q, complex(parameters.U - 3.0 * parameters.J)))

    for i in (0, 1):
        for j in (0, 1):
            if i == j:
                continue
            i_up = orbital(i, 0)
            i_down = orbital(i, 1)
            j_up = orbital(j, 0)
            j_down = orbital(j, 1)
            terms.extend(
                (
                    (i_up, j_down, j_up, i_down, complex(parameters.J)),
                    (i_up, i_down, j_up, j_down, complex(parameters.J)),
                )
            )
    return SparseTwoBodyInteraction(4, tuple(terms))


__all__ = [
    "Case",
    "ScalarBathFit",
    "bethe_retarded",
    "build_cc_problem",
    "build_eom_context",
    "build_model",
    "build_static_problem",
    "case",
    "continue_static_cpu",
    "fit_scalar_bath",
    "impurity_one_body",
    "interaction",
    "load_bath_fit",
    "load_static_amplitudes",
    "orbital_hybridization_matrix",
    "particle_hole_symmetric_energy",
    "results_directory",
    "scalar_bcf",
    "save_bath_fit",
    "semicircle_density",
    "solve_static_cpu",
]

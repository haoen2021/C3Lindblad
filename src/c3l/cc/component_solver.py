"""Matrix-free Newton--Krylov solver using component-tensor CCSD actions."""

from __future__ import annotations

from dataclasses import dataclass
import resource
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from .component_tensor import (
    ComponentRankDiagnostic,
    ComponentStaticAction,
    ComponentStaticLinearization,
    ComponentStaticProblem,
    component_static_linearization,
    component_static_linearized_jvp,
    component_static_rank6_diagnostic,
    component_static_residual,
)
from .newton_krylov import (
    NewtonKrylovPolicy,
    NewtonKrylovStep,
    RapidityPreconditionerDiagnostics,
    _forcing_tolerance,
    _preconditioner,
)


RealVector = NDArray[np.float64]
ComplexVector = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class ComponentCCSolution:
    problem: ComponentStaticProblem
    variables: RealVector
    amplitudes: ComplexVector
    residual: ComponentStaticAction
    residual_norm: float
    packed_residual_norm: float
    reference_residual: float
    rank6: ComponentRankDiagnostic | None
    converged: bool
    iterations: int
    residual_evaluations: int
    jvp_evaluations: int
    history: tuple[NewtonKrylovStep, ...]
    preconditioner: RapidityPreconditionerDiagnostics
    wall_seconds: float
    peak_rss_kib: int


def _real_residual(problem: ComponentStaticProblem, variables: ArrayLike) -> RealVector:
    x = np.asarray(variables, dtype=np.float64)
    amplitudes = problem.oracle.tilde.unpack(x)
    residual = component_static_residual(problem, amplitudes).retained
    return np.asarray(problem.oracle.tilde.pack(residual, check=False), dtype=np.float64)


def _real_linearized_jvp(
    context: ComponentStaticLinearization,
    direction: ArrayLike,
) -> RealVector:
    dx = np.asarray(direction, dtype=np.float64)
    oracle = context.problem.oracle
    if dx.shape != (oracle.num_variables,):
        raise ValueError("JVP direction has the wrong shape")
    tangent = oracle.tilde.unpack(dx)
    result = component_static_linearized_jvp(context, tangent).retained
    return np.asarray(oracle.tilde.pack(result, check=False), dtype=np.float64)


def _acceptable_inexact_newton_step(
    residual: RealVector,
    jacobian_step: RealVector,
    linear_defect: RealVector,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    """Check the true linear defect and nonlinear-merit descent condition."""

    residual_norm = float(np.linalg.norm(residual))
    defect_target = max(
        relative_tolerance * residual_norm,
        absolute_tolerance,
    )
    roundoff_margin = 1.0 + np.sqrt(np.finfo(float).eps)
    defect_norm = float(np.linalg.norm(linear_defect))
    merit_slope = float(np.dot(residual, jacobian_step))
    return bool(
        np.isfinite(defect_norm)
        and defect_norm <= roundoff_margin * defect_target
        and np.isfinite(merit_slope)
        and merit_slope < 0.0
    )


def solve_component_ccsd(
    problem: ComponentStaticProblem,
    initial_amplitudes: ArrayLike | None = None,
    *,
    policy: NewtonKrylovPolicy | None = None,
    rank6_tile_size: int = 4096,
    compute_rank6: bool = False,
    accepted_step_callback: Callable[[int, ComplexVector, float], None] | None = None,
) -> ComponentCCSolution:
    """Solve the tilde-real CCSD equations with component residuals and JVPs.

    compute_rank6=False avoids evaluating the omitted static rank-six shell.
    Enable it explicitly for a truncation diagnostic; it does not enter the
    CCSD equations or their convergence test and can be expensive."""

    controls = NewtonKrylovPolicy() if policy is None else policy
    oracle = problem.oracle
    started = perf_counter()
    if initial_amplitudes is None:
        x = np.zeros(oracle.num_variables, dtype=np.float64)
    else:
        initial = np.asarray(initial_amplitudes, dtype=np.complex128)
        if initial.shape != (oracle.num_amplitudes,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial amplitudes do not match the component CCSD problem")
        x = np.asarray(oracle.tilde.pack(initial, check=False), dtype=np.float64)

    preconditioner, preconditioner_diagnostics = _preconditioner(oracle)
    residual_evaluations = 0
    jvp_evaluations = 0
    history: list[NewtonKrylovStep] = []
    converged = False
    previous_residual_norm: float | None = None
    previous_forcing_tolerance: float | None = None

    def residual(values: RealVector) -> RealVector:
        nonlocal residual_evaluations
        residual_evaluations += 1
        return _real_residual(problem, values)

    for iteration in range(controls.maximum_newton_iterations + 1):
        amplitudes = oracle.tilde.unpack(x)
        context = component_static_linearization(problem, amplitudes)
        residual_evaluations += 1
        r = np.asarray(
            oracle.tilde.pack(context.residual.retained, check=False),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(r))
        if norm <= controls.tolerance:
            del context
            converged = True
            break
        if iteration == controls.maximum_newton_iterations:
            del context
            break

        def jvp(direction: RealVector) -> RealVector:
            nonlocal jvp_evaluations
            jvp_evaluations += 1
            return _real_linearized_jvp(context, direction)

        jacobian = LinearOperator(
            (oracle.num_variables, oracle.num_variables),
            matvec=jvp,
            dtype=np.float64,
        )
        linear_tolerance = _forcing_tolerance(
            norm,
            controls,
            previous_norm=previous_residual_norm,
            previous_tolerance=previous_forcing_tolerance,
        )
        linear_absolute_tolerance = (
            0.0
            if controls.forcing_strategy == "legacy"
            else controls.gmres_absolute_tolerance_factor * controls.tolerance
        )
        linear_history: list[float] = []
        step, info = gmres(
            jacobian,
            -r,
            M=preconditioner,
            rtol=linear_tolerance,
            atol=linear_absolute_tolerance,
            restart=controls.gmres_restart,
            maxiter=controls.gmres_maximum_restart_cycles,
            callback=lambda value: linear_history.append(float(value)),
            callback_type="pr_norm",
        )
        step = np.asarray(step, dtype=np.float64)
        jacobian_step = jvp(step)
        linear_defect = jacobian_step + r
        true_linear_relative = float(
            np.linalg.norm(linear_defect) / max(norm, np.finfo(float).tiny)
        )
        del jacobian, jvp, context
        if (
            controls.forcing_strategy == "eisenstat_walker"
            and not _acceptable_inexact_newton_step(
                r,
                jacobian_step,
                linear_defect,
                relative_tolerance=linear_tolerance,
                absolute_tolerance=linear_absolute_tolerance,
            )
        ):
            history.append(
                NewtonKrylovStep(
                    iteration=iteration,
                    residual_norm=norm,
                    gmres_relative_tolerance=linear_tolerance,
                    gmres_info=int(info),
                    gmres_iterations=len(linear_history),
                    gmres_preconditioned_residual_history=tuple(linear_history),
                    true_linear_relative_residual=true_linear_relative,
                    step_norm=float(np.linalg.norm(step)),
                    damping=0.0,
                    accepted_residual_norm=norm,
                )
            )
            break
        damping = 1.0
        accepted_norm = np.inf
        accepted = False
        for _ in range(controls.maximum_line_search_steps):
            trial = residual(x + damping * step)
            accepted_norm = float(np.linalg.norm(trial))
            if accepted_norm <= (1.0 - controls.armijo * damping) * norm:
                x += damping * step
                accepted = True
                if accepted_step_callback is not None:
                    accepted_step_callback(
                        iteration,
                        np.asarray(oracle.tilde.unpack(x), dtype=np.complex128),
                        accepted_norm,
                    )
                break
            damping *= 0.5
        history.append(
            NewtonKrylovStep(
                iteration=iteration,
                residual_norm=norm,
                gmres_relative_tolerance=linear_tolerance,
                gmres_info=int(info),
                gmres_iterations=len(linear_history),
                gmres_preconditioned_residual_history=tuple(linear_history),
                true_linear_relative_residual=true_linear_relative,
                step_norm=float(np.linalg.norm(step)),
                damping=float(damping),
                accepted_residual_norm=accepted_norm,
            )
        )
        if not accepted:
            break
        previous_residual_norm = norm
        previous_forcing_tolerance = linear_tolerance

    amplitudes = np.asarray(oracle.tilde.unpack(x), dtype=np.complex128)
    final = component_static_residual(problem, amplitudes)
    retained_norm = float(np.linalg.norm(final.retained))
    packed_norm = float(
        np.linalg.norm(oracle.tilde.pack(final.retained, check=False))
    )
    rank6 = None
    if compute_rank6:
        zero = np.zeros(oracle.num_amplitudes, dtype=np.complex128)
        bare_scale = float(
            np.linalg.norm(component_static_residual(problem, zero).retained)
        )
        rank6 = component_static_rank6_diagnostic(
            problem,
            amplitudes,
            normalization_scale=bare_scale,
            tile_size=rank6_tile_size,
        )
    return ComponentCCSolution(
        problem=problem,
        variables=np.asarray(x, dtype=np.float64),
        amplitudes=amplitudes,
        residual=final,
        residual_norm=retained_norm,
        packed_residual_norm=packed_norm,
        reference_residual=float(abs(final.reference)),
        rank6=rank6,
        converged=bool(converged and packed_norm <= controls.tolerance),
        iterations=len(history),
        residual_evaluations=residual_evaluations,
        jvp_evaluations=jvp_evaluations,
        history=tuple(history),
        preconditioner=preconditioner_diagnostics,
        wall_seconds=float(perf_counter() - started),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


__all__ = ["ComponentCCSolution", "solve_component_ccsd"]

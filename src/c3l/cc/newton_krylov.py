"""Newton--Krylov controls and diagonal preconditioning for component CC.

The nonlinear equations are the retained connected coefficients of
``exp(-T) L exp(T)|0>`` in the independent real tilde coordinates.  Each
Newton correction is obtained from analytic Jacobian-vector products; this
module never assembles a Jacobian.  The diagonal reference rapidities provide
the right scale for a real-linear preconditioner, and an Armijo backtracking
line search selects steps that decrease the residual merit function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator


from .factorized import FactorizedCCProblem


RealVector = NDArray[np.float64]
ComplexVector = NDArray[np.complex128]


def _frozen_complex(values: np.ndarray) -> ComplexVector:
    array = np.ascontiguousarray(values, dtype=np.complex128)
    result = np.frombuffer(array.tobytes(), dtype=np.complex128).reshape(array.shape)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class NewtonKrylovPolicy:
    """Numerical controls for the CCSD static solve."""

    tolerance: float = 1.0e-10
    maximum_newton_iterations: int = 20
    gmres_restart: int = 40
    gmres_maximum_restart_cycles: int = 12
    minimum_gmres_relative_tolerance: float = 1.0e-8
    maximum_gmres_relative_tolerance: float = 1.0e-3
    maximum_line_search_steps: int = 14
    armijo: float = 1.0e-4
    forcing_strategy: str = "legacy"
    initial_gmres_relative_tolerance: float = 1.0e-2
    eisenstat_walker_gamma: float = 0.9
    eisenstat_walker_exponent: float = 1.5
    gmres_absolute_tolerance_factor: float = 0.1

    def __post_init__(self) -> None:
        positive = (
            self.tolerance,
            self.minimum_gmres_relative_tolerance,
            self.maximum_gmres_relative_tolerance,
            self.armijo,
            self.initial_gmres_relative_tolerance,
            self.eisenstat_walker_gamma,
            self.eisenstat_walker_exponent,
            self.gmres_absolute_tolerance_factor,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Newton--Krylov tolerances must be finite and positive")
        if self.minimum_gmres_relative_tolerance > self.maximum_gmres_relative_tolerance:
            raise ValueError("minimum GMRES tolerance exceeds the maximum")
        if self.maximum_gmres_relative_tolerance >= 1.0 or self.armijo >= 1.0:
            raise ValueError("relative and Armijo tolerances must be smaller than one")
        if self.forcing_strategy not in ("legacy", "eisenstat_walker"):
            raise ValueError("unknown inexact-Newton forcing strategy")
        if not 0.0 < self.eisenstat_walker_gamma < 1.0:
            raise ValueError("Eisenstat--Walker gamma must lie in (0,1)")
        if not 1.0 < self.eisenstat_walker_exponent <= 2.0:
            raise ValueError("Eisenstat--Walker exponent must lie in (1,2]")
        if not 0.0 < self.gmres_absolute_tolerance_factor <= 1.0:
            raise ValueError("GMRES absolute-tolerance factor must lie in (0,1]")
        if (
            self.forcing_strategy == "eisenstat_walker"
            and not (
                self.minimum_gmres_relative_tolerance
                <= self.initial_gmres_relative_tolerance
                <= self.maximum_gmres_relative_tolerance
            )
        ):
            raise ValueError("initial GMRES tolerance lies outside the forcing bounds")
        integers = (
            self.maximum_newton_iterations,
            self.gmres_restart,
            self.gmres_maximum_restart_cycles,
            self.maximum_line_search_steps,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integers
        ):
            raise ValueError("Newton--Krylov iteration limits must be positive integers")

    def to_dict(self) -> dict[str, float | int | str]:
        result: dict[str, float | int | str] = {
            "tolerance": self.tolerance,
            "maximum_newton_iterations": self.maximum_newton_iterations,
            "gmres_restart": self.gmres_restart,
            "gmres_maximum_restart_cycles": self.gmres_maximum_restart_cycles,
            "minimum_gmres_relative_tolerance": self.minimum_gmres_relative_tolerance,
            "maximum_gmres_relative_tolerance": self.maximum_gmres_relative_tolerance,
            "maximum_line_search_steps": self.maximum_line_search_steps,
            "armijo": self.armijo,
        }
        if self.forcing_strategy == "eisenstat_walker":
            result.update(
                {
                    "forcing_strategy": self.forcing_strategy,
                    "initial_gmres_relative_tolerance": (
                        self.initial_gmres_relative_tolerance
                    ),
                    "eisenstat_walker_gamma": self.eisenstat_walker_gamma,
                    "eisenstat_walker_exponent": self.eisenstat_walker_exponent,
                    "gmres_absolute_tolerance_factor": (
                        self.gmres_absolute_tolerance_factor
                    ),
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class RapidityPreconditionerDiagnostics:
    """Conditioning of the diagonal reference approximation, not of full J."""

    minimum_magnitude: float
    maximum_magnitude: float
    magnitude_condition: float
    regularization_threshold: float
    regularized_amplitudes: int


def rapidity_diagonal(problem: FactorizedCCProblem) -> ComplexVector:
    r"""Return the physical-generator diagonal $-i\sum_{p\in\mu}\lambda_p$."""

    diagonal = np.asarray(
        [
            -1j
            * sum(
                (
                    problem.rapidities[p]
                    for p in range(problem.cluster_basis.num_modes)
                    if mask & (1 << p)
                ),
                start=0.0 + 0.0j,
            )
            for mask in problem.cluster_basis.masks
        ],
        dtype=np.complex128,
    )
    return _frozen_complex(diagonal)


def _preconditioner(
    problem: FactorizedCCProblem,
) -> tuple[LinearOperator, RapidityPreconditionerDiagnostics]:
    diagonal = np.array(rapidity_diagonal(problem), copy=True)
    magnitudes = np.abs(diagonal)
    scale = max(float(np.max(magnitudes, initial=0.0)), 1.0)
    threshold = 1.0e-12 * scale
    small = magnitudes < threshold
    if np.any(small):
        diagonal[small] = threshold * np.exp(
            1j * np.angle(diagonal[small] + 1.0e-300)
        )
    finite = magnitudes[~small]
    minimum = float(np.min(finite)) if finite.size else threshold
    maximum = float(np.max(magnitudes, initial=threshold))

    def solve(values: np.ndarray) -> RealVector:
        direction = problem.tilde.unpack(np.asarray(values, dtype=np.float64))
        return np.asarray(
            problem.tilde.pack(direction / diagonal, check=False), dtype=np.float64
        )

    operator = LinearOperator(
        (problem.num_variables, problem.num_variables),
        matvec=solve,
        dtype=np.float64,
    )
    diagnostics = RapidityPreconditionerDiagnostics(
        minimum_magnitude=minimum,
        maximum_magnitude=maximum,
        magnitude_condition=maximum / max(minimum, threshold),
        regularization_threshold=threshold,
        regularized_amplitudes=int(np.count_nonzero(small)),
    )
    return operator, diagnostics


@dataclass(frozen=True, slots=True)
class NewtonKrylovStep:
    iteration: int
    residual_norm: float
    gmres_relative_tolerance: float
    gmres_info: int
    gmres_iterations: int
    gmres_preconditioned_residual_history: tuple[float, ...]
    true_linear_relative_residual: float
    step_norm: float
    damping: float
    accepted_residual_norm: float


def _forcing_tolerance(
    norm: float,
    policy: NewtonKrylovPolicy,
    *,
    previous_norm: float | None = None,
    previous_tolerance: float | None = None,
) -> float:
    """Return the relative defect requested from one inexact Newton solve."""

    if policy.forcing_strategy == "legacy":
        value = 0.1 * max(norm, policy.tolerance)
    elif previous_norm is None:
        value = policy.initial_gmres_relative_tolerance
    else:
        ratio = norm / max(previous_norm, np.finfo(float).tiny)
        value = policy.eisenstat_walker_gamma * (
            ratio**policy.eisenstat_walker_exponent
        )
        if previous_tolerance is not None:
            safeguard = policy.eisenstat_walker_gamma * (
                previous_tolerance**policy.eisenstat_walker_exponent
            )
            value = max(value, safeguard)
    return float(
        min(
            policy.maximum_gmres_relative_tolerance,
            max(policy.minimum_gmres_relative_tolerance, value),
        )
    )


__all__ = [
    "NewtonKrylovPolicy",
    "NewtonKrylovStep",
    "RapidityPreconditionerDiagnostics",
    "rapidity_diagonal",
]

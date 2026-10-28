"""Matrix-free retained EOM response solvers."""

from __future__ import annotations

from dataclasses import dataclass
import resource
from time import perf_counter
from typing import Sequence

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from .component_tensor import (
    ComponentEOMContext,
    ComponentEOMProblem,
    ComponentEOMSources,
    component_eom_sigma,
)
from c3l.green.krylov import ArnoldiResolvent, arnoldi_resolvent


ComplexVector = NDArray[np.complex128]
ComplexMatrix = NDArray[np.complex128]


def _rapidity_diagonal(problem: ComponentEOMProblem) -> ComplexVector:
    rapidities = problem.static.oracle.rapidities
    n = problem.coordinates.n
    out = np.zeros(problem.coordinates.dimension, dtype=np.complex128)
    for block in problem.coordinates.blocks:
        value = np.zeros(block.vector_positions.size, dtype=np.complex128)
        axis = 0
        for group, count in enumerate(block.key):
            for _ in range(count):
                value += rapidities[group * n + block.indices[:, axis]]
                axis += 1
        out[block.vector_positions] = value
    return out


class ComponentSigmaOperator:
    """Instrumented matrix-free retained sigma action."""

    def __init__(self, context: ComponentEOMContext):
        self.context = context
        self.dimension = context.problem.coordinates.dimension
        self.diagonal = _rapidity_diagonal(context.problem)
        self.action_calls = 0
        self.action_wall_seconds = 0.0

    def matvec(self, vector: ArrayLike) -> ComplexVector:
        started = perf_counter()
        result = component_eom_sigma(self.context, vector).retained
        self.action_calls += 1
        self.action_wall_seconds += perf_counter() - started
        return np.asarray(result, dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class ComponentShiftedSolution:
    solution: ComplexVector
    converged: bool
    info: int
    iterations: int
    true_relative_residual: float
    residual_history: tuple[float, ...]
    wall_seconds: float


def solve_component_shifted(
    operator: ComponentSigmaOperator,
    shift: complex,
    rhs: ArrayLike,
    *,
    rtol: float = 1.0e-10,
    restart: int = 40,
    maxiter: int = 200,
    initial_solution: ArrayLike | None = None,
) -> ComponentShiftedSolution:
    """Solve ``(z I-A)x=b`` and reapply the full component action."""

    b = np.asarray(rhs, dtype=np.complex128)
    if b.shape != (operator.dimension,) or not np.all(np.isfinite(b)):
        raise ValueError("shifted-response source has the wrong shape or is nonfinite")
    z = complex(shift)
    diagonal = z - operator.diagonal
    threshold = np.sqrt(np.finfo(np.float64).eps)
    inverse = np.ones_like(diagonal)
    safe = np.abs(diagonal) > threshold
    inverse[safe] = 1.0 / diagonal[safe]
    A = LinearOperator(
        (operator.dimension, operator.dimension),
        matvec=lambda x: z * x - operator.matvec(x),
        dtype=np.complex128,
    )
    M = LinearOperator(
        (operator.dimension, operator.dimension),
        matvec=lambda x: inverse * x,
        dtype=np.complex128,
    )
    x0 = None if initial_solution is None else np.asarray(initial_solution, dtype=np.complex128)
    history: list[float] = []
    started = perf_counter()
    x, info = gmres(
        A,
        b,
        x0=x0,
        M=M,
        rtol=rtol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        callback=lambda value: history.append(float(value)),
        callback_type="pr_norm",
    )
    residual = b - (z * x - operator.matvec(x))
    true = float(np.linalg.norm(residual) / max(np.linalg.norm(b), np.finfo(float).tiny))
    return ComponentShiftedSolution(
        solution=np.asarray(x, dtype=np.complex128),
        converged=bool(info == 0 and true <= max(10.0 * rtol, 1.0e-12)),
        info=int(info),
        iterations=len(history),
        true_relative_residual=true,
        residual_history=tuple(history),
        wall_seconds=float(perf_counter() - started),
    )


@dataclass(frozen=True, slots=True)
class ComponentRetardedGreen:
    omega: NDArray[np.float64]
    eta: float
    addition: ComplexVector
    removal: ComplexVector
    retarded: ComplexVector
    addition_solutions: tuple[ComponentShiftedSolution, ...]
    removal_solutions: tuple[ComponentShiftedSolution, ...]
    wall_seconds: float
    peak_rss_kib: int


@dataclass(frozen=True, slots=True)
class ComponentRetardedMatrixGreen:
    """Frequency-domain Green matrix from one response solve per source column."""

    omega: NDArray[np.float64]
    eta: float
    sites: tuple[int, ...]
    addition: NDArray[np.complex128]
    removal: NDArray[np.complex128]
    retarded: NDArray[np.complex128]
    addition_solutions: tuple[tuple[ComponentShiftedSolution, ...], ...]
    removal_solutions: tuple[tuple[ComponentShiftedSolution, ...], ...]
    wall_seconds: float
    peak_rss_kib: int


@dataclass(frozen=True, slots=True)
class ComponentRetardedMatrixArnoldiGreen:
    """Shift-invariant one-sided Arnoldi matrix response over many frequencies."""

    omega: NDArray[np.float64]
    eta: float
    sites: tuple[int, ...]
    addition: NDArray[np.complex128]
    removal: NDArray[np.complex128]
    retarded: NDArray[np.complex128]
    addition_projections: tuple[ArnoldiResolvent, ...]
    removal_projections: tuple[ArnoldiResolvent, ...]
    addition_relative_residuals: NDArray[np.float64]
    removal_relative_residuals: NDArray[np.float64]
    component_action_calls: int
    component_action_wall_seconds: float
    wall_seconds: float
    peak_rss_kib: int


def component_retarded_green(
    operator: ComponentSigmaOperator,
    sources: ComponentEOMSources,
    omega: ArrayLike,
    eta: float,
    *,
    rtol: float = 1.0e-10,
    restart: int = 40,
    maxiter: int = 200,
) -> ComponentRetardedGreen:
    """Compute the two positive-time contributions to the retarded response."""

    frequency = np.asarray(omega, dtype=np.float64)
    p = sources.addition.size
    if operator.dimension < p:
        raise ValueError("response operator is smaller than the retained source")
    zeros = np.zeros(operator.dimension - p, dtype=np.complex128)
    addition_source = np.concatenate((sources.addition, zeros))
    removal_source = np.concatenate((sources.removal, zeros))
    sink = np.concatenate((sources.sink, zeros))
    addition = np.empty(frequency.size, dtype=np.complex128)
    removal = np.empty_like(addition)
    addition_solves: list[ComponentShiftedSolution] = []
    removal_solves: list[ComponentShiftedSolution] = []
    previous_addition = None
    previous_removal = None
    started = perf_counter()
    for i, value in enumerate(frequency):
        z = complex(float(value), float(eta))
        add = solve_component_shifted(
            operator,
            z,
            addition_source,
            rtol=rtol,
            restart=restart,
            maxiter=maxiter,
            initial_solution=previous_addition,
        )
        rem = solve_component_shifted(
            operator,
            z,
            removal_source,
            rtol=rtol,
            restart=restart,
            maxiter=maxiter,
            initial_solution=previous_removal,
        )
        if not add.converged or not rem.converged:
            raise RuntimeError("a component EOM shifted solve failed to converge")
        previous_addition = add.solution
        previous_removal = rem.solution
        addition_solves.append(add)
        removal_solves.append(rem)
        addition[i] = sink @ add.solution
        removal[i] = sink @ rem.solution
    return ComponentRetardedGreen(
        omega=frequency,
        eta=float(eta),
        addition=addition,
        removal=removal,
        retarded=addition + removal,
        addition_solutions=tuple(addition_solves),
        removal_solutions=tuple(removal_solves),
        wall_seconds=float(perf_counter() - started),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


def component_retarded_matrix_green(
    operator: ComponentSigmaOperator,
    sources: Sequence[ComponentEOMSources],
    omega: ArrayLike,
    eta: float,
    *,
    rtol: float = 1.0e-10,
    restart: int = 40,
    maxiter: int = 200,
) -> ComponentRetardedMatrixGreen:
    """Compute all sink rows with one addition/removal solve per source column.

    ``sources[J]`` contains the two QRT source columns for site ``J`` and its
    sink row.  At each frequency the two shifted systems are solved once for
    each source site; every requested sink is then contracted with those same
    solutions.  No block Krylov space or explicit EOM matrix is constructed.
    The returned array axes are ``(omega, sink_site, source_site)``.
    """

    columns = tuple(sources)
    if not columns:
        raise ValueError("at least one impurity source is required")
    if len({item.site for item in columns}) != len(columns):
        raise ValueError("impurity source sites must be unique")
    if any(item.charge != columns[0].charge for item in columns):
        raise ValueError("all impurity sources must use one response charge")
    retained_dimension = columns[0].addition.size
    if any(
        item.addition.shape != (retained_dimension,)
        or item.removal.shape != (retained_dimension,)
        or item.sink.shape != (retained_dimension,)
        for item in columns
    ):
        raise ValueError("impurity sources and sinks must share one retained basis")
    if operator.dimension < retained_dimension:
        raise ValueError("response operator is smaller than the retained sources")
    frequency = np.asarray(omega, dtype=np.float64)
    broadening = float(eta)
    if (
        frequency.ndim != 1
        or frequency.size == 0
        or not np.all(np.isfinite(frequency))
        or not np.isfinite(broadening)
        or broadening <= 0.0
    ):
        raise ValueError("omega and positive eta must be finite")

    padding = operator.dimension - retained_dimension
    zeros = np.zeros(padding, dtype=np.complex128)
    sinks = np.vstack([np.concatenate((item.sink, zeros)) for item in columns])
    shape = (frequency.size, len(columns), len(columns))
    addition = np.empty(shape, dtype=np.complex128)
    removal = np.empty(shape, dtype=np.complex128)
    addition_records: list[tuple[ComponentShiftedSolution, ...]] = []
    removal_records: list[tuple[ComponentShiftedSolution, ...]] = []
    started = perf_counter()
    for source_index, source in enumerate(columns):
        addition_source = np.concatenate((source.addition, zeros))
        removal_source = np.concatenate((source.removal, zeros))
        previous_addition = None
        previous_removal = None
        source_addition: list[ComponentShiftedSolution] = []
        source_removal: list[ComponentShiftedSolution] = []
        for frequency_index, value in enumerate(frequency):
            shift = complex(float(value), broadening)
            add = solve_component_shifted(
                operator,
                shift,
                addition_source,
                rtol=rtol,
                restart=restart,
                maxiter=maxiter,
                initial_solution=previous_addition,
            )
            rem = solve_component_shifted(
                operator,
                shift,
                removal_source,
                rtol=rtol,
                restart=restart,
                maxiter=maxiter,
                initial_solution=previous_removal,
            )
            if not add.converged or not rem.converged:
                raise RuntimeError("a component EOM matrix shifted solve failed")
            previous_addition = add.solution
            previous_removal = rem.solution
            source_addition.append(add)
            source_removal.append(rem)
            addition[frequency_index, :, source_index] = sinks @ add.solution
            removal[frequency_index, :, source_index] = sinks @ rem.solution
        addition_records.append(tuple(source_addition))
        removal_records.append(tuple(source_removal))

    return ComponentRetardedMatrixGreen(
        omega=frequency,
        eta=broadening,
        sites=tuple(item.site for item in columns),
        addition=addition,
        removal=removal,
        retarded=addition + removal,
        addition_solutions=tuple(addition_records),
        removal_solutions=tuple(removal_records),
        wall_seconds=float(perf_counter() - started),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


def component_retarded_matrix_green_arnoldi(
    operator: ComponentSigmaOperator,
    sources: Sequence[ComponentEOMSources],
    omega: ArrayLike,
    eta: float,
    *,
    maximum_krylov_dimension: int = 320,
    residual_tolerance: float = 1.0e-10,
    minimum_krylov_dimension: int = 16,
    check_interval: int = 8,
) -> ComponentRetardedMatrixArnoldiGreen:
    """Reuse one one-sided Arnoldi basis per source/channel over all shifts.

    This is the many-frequency companion to :func:`component_retarded_matrix_green`.
    It leaves the direct GMRES implementation unchanged as the strict oracle.
    Each addition or removal source builds its own ordinary one-sided Arnoldi
    space; all sink rows and all frequencies are contracted from that space.
    """

    columns = tuple(sources)
    if not columns:
        raise ValueError("at least one impurity source is required")
    if len({item.site for item in columns}) != len(columns):
        raise ValueError("impurity source sites must be unique")
    if any(item.charge != columns[0].charge for item in columns):
        raise ValueError("all impurity sources must use one response charge")
    retained_dimension = columns[0].addition.size
    if any(
        item.addition.shape != (retained_dimension,)
        or item.removal.shape != (retained_dimension,)
        or item.sink.shape != (retained_dimension,)
        for item in columns
    ):
        raise ValueError("impurity sources and sinks must share one retained basis")
    if operator.dimension < retained_dimension:
        raise ValueError("response operator is smaller than the retained sources")
    frequency = np.asarray(omega, dtype=np.float64)
    broadening = float(eta)
    if (
        frequency.ndim != 1
        or frequency.size == 0
        or not np.all(np.isfinite(frequency))
        or not np.isfinite(broadening)
        or broadening <= 0.0
    ):
        raise ValueError("omega and positive eta must be finite")
    if (
        isinstance(maximum_krylov_dimension, bool)
        or not isinstance(maximum_krylov_dimension, int)
        or isinstance(minimum_krylov_dimension, bool)
        or not isinstance(minimum_krylov_dimension, int)
        or isinstance(check_interval, bool)
        or not isinstance(check_interval, int)
    ):
        raise TypeError("Krylov dimensions and check_interval must be integers")
    if (
        maximum_krylov_dimension <= 0
        or minimum_krylov_dimension <= 0
        or check_interval <= 0
        or not np.isfinite(residual_tolerance)
        or residual_tolerance <= 0.0
    ):
        raise ValueError("Krylov controls and residual tolerance must be positive")
    maximum_dimension = min(maximum_krylov_dimension, operator.dimension)
    minimum_dimension = min(minimum_krylov_dimension, maximum_dimension)
    if maximum_dimension <= 0 or minimum_dimension <= 0:
        raise ValueError("Krylov dimensions must be positive")

    shifts = frequency.astype(np.complex128) + 1j * broadening
    padding = operator.dimension - retained_dimension
    zeros = np.zeros(padding, dtype=np.complex128)
    sinks = np.vstack([np.concatenate((item.sink, zeros)) for item in columns])
    shape = (frequency.size, len(columns), len(columns))
    addition = np.empty(shape, dtype=np.complex128)
    removal = np.empty(shape, dtype=np.complex128)
    addition_projections: list[ArnoldiResolvent] = []
    removal_projections: list[ArnoldiResolvent] = []
    before_calls = operator.action_calls
    before_action_seconds = operator.action_wall_seconds
    started = perf_counter()

    def project(source: ComplexVector) -> ArnoldiResolvent:
        projection = arnoldi_resolvent(
            operator,
            np.concatenate((source, zeros)),
            shifts,
            maximum_dimension=maximum_dimension,
            residual_tolerance=float(residual_tolerance),
            minimum_dimension=minimum_dimension,
            check_interval=check_interval,
        )
        if not projection.converged:
            raise RuntimeError("a component EOM matrix Arnoldi projection failed")
        return projection

    for source_index, source in enumerate(columns):
        add = project(source.addition)
        rem = project(source.removal)
        addition_projections.append(add)
        removal_projections.append(rem)
        for values, projection in ((addition, add), (removal, rem)):
            m = projection.dimension
            reduced = projection.coefficients(shifts)
            projected_sinks = sinks @ projection.basis[:, :m]
            values[:, :, source_index] = reduced @ projected_sinks.T

    addition_residuals = np.asarray(
        [projection.relative_residuals(shifts) for projection in addition_projections],
        dtype=np.float64,
    )
    removal_residuals = np.asarray(
        [projection.relative_residuals(shifts) for projection in removal_projections],
        dtype=np.float64,
    )
    action_calls = operator.action_calls - before_calls
    expected_calls = sum(
        projection.dimension
        for projection in addition_projections + removal_projections
    )
    if action_calls != expected_calls:
        raise RuntimeError("component action count disagrees with Arnoldi dimensions")
    return ComponentRetardedMatrixArnoldiGreen(
        omega=frequency,
        eta=broadening,
        sites=tuple(item.site for item in columns),
        addition=addition,
        removal=removal,
        retarded=addition + removal,
        addition_projections=tuple(addition_projections),
        removal_projections=tuple(removal_projections),
        addition_relative_residuals=addition_residuals,
        removal_relative_residuals=removal_residuals,
        component_action_calls=int(action_calls),
        component_action_wall_seconds=float(
            operator.action_wall_seconds - before_action_seconds
        ),
        wall_seconds=float(perf_counter() - started),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


def component_retarded_time(
    operator: ComponentSigmaOperator,
    sources: ComponentEOMSources,
    time: ArrayLike,
    *,
    maximum_krylov_dimension: int = 160,
    residual_tolerance: float = 1.0e-11,
) -> ComplexVector:
    """Propagate the combined positive-time source in one Arnoldi space."""

    t = np.asarray(time, dtype=np.float64)
    p = sources.addition.size
    zeros = np.zeros(operator.dimension - p, dtype=np.complex128)
    source = np.concatenate((sources.addition + sources.removal, zeros))
    sink = np.concatenate((sources.sink, zeros))
    probe = np.asarray([1j], dtype=np.complex128)
    krylov = arnoldi_resolvent(
        operator,
        source,
        probe,
        maximum_dimension=min(maximum_krylov_dimension, operator.dimension),
        residual_tolerance=residual_tolerance,
        minimum_dimension=min(16, operator.dimension),
        check_interval=min(8, operator.dimension),
    )
    m = krylov.dimension
    H = krylov.hessenberg[:m, :m]
    reduced_source = np.zeros(m, dtype=np.complex128)
    reduced_source[0] = krylov.source_norm
    projected_sink = sink @ krylov.basis[:, :m]
    out = np.empty(t.size, dtype=np.complex128)
    for i, value in enumerate(t):
        state = scipy.linalg.expm(-1j * float(value) * H) @ reduced_source
        out[i] = -1j * (projected_sink @ state)
    return out


@dataclass(frozen=True, slots=True)
class ComponentRetardedMatrixTime:
    """Direct-time impurity Green matrix from one solve per source column."""

    time: NDArray[np.float64]
    sites: tuple[int, ...]
    addition: NDArray[np.complex128]
    removal: NDArray[np.complex128]
    retarded: NDArray[np.complex128]
    channel_sum_defect: float
    combined_projections: tuple[ArnoldiResolvent, ...]
    addition_projections: tuple[ArnoldiResolvent, ...]
    removal_projections: tuple[ArnoldiResolvent, ...]
    combined_relative_time_residuals: NDArray[np.float64]
    addition_relative_time_residuals: NDArray[np.float64]
    removal_relative_time_residuals: NDArray[np.float64]
    wall_seconds: float
    peak_rss_kib: int


def _matrix_time_from_projections(
    projections: Sequence[ArnoldiResolvent],
    sinks: ComplexMatrix,
    time: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Contract every sink row with each source-column Arnoldi propagation."""

    n_sites = sinks.shape[0]
    out = np.empty((time.size, n_sites, len(projections)), dtype=np.complex128)
    for source, projection in enumerate(projections):
        m = projection.dimension
        projected_sinks = sinks @ projection.basis[:, :m]
        reduced = projection.time_coefficients(time)
        out[:, :, source] = -1j * (reduced @ projected_sinks.T)
    return out


def component_retarded_matrix_time(
    operator: ComponentSigmaOperator,
    sources: Sequence[ComponentEOMSources],
    time: ArrayLike,
    validation_shifts: ArrayLike,
    *,
    maximum_krylov_dimension: int = 160,
    residual_tolerance: float = 1.0e-11,
    include_channels: bool = True,
) -> ComponentRetardedMatrixTime:
    """Compute all sink rows from one retained-EOM solve per source orbital.

    ``sources[J]`` supplies the addition/removal columns for impurity orbital
    ``J`` and its sink row.  The official retarded matrix uses two combined
    anticommutator columns for a two-site dimer.  Optional separate channel
    projections diagnose addition/removal cancellation without changing that
    source-column count.
    """

    columns = tuple(sources)
    if not columns:
        raise ValueError("at least one impurity source is required")
    if len({item.site for item in columns}) != len(columns):
        raise ValueError("impurity source sites must be unique")
    if any(item.charge != columns[0].charge for item in columns):
        raise ValueError("all impurity sources must use one response charge")
    retained_dimension = columns[0].addition.size
    if any(
        item.addition.shape != (retained_dimension,)
        or item.removal.shape != (retained_dimension,)
        or item.sink.shape != (retained_dimension,)
        for item in columns
    ):
        raise ValueError("impurity sources and sinks must share one retained basis")
    if operator.dimension < retained_dimension:
        raise ValueError("response operator is smaller than the retained sources")
    t = np.asarray(time, dtype=np.float64)
    z = np.asarray(validation_shifts, dtype=np.complex128)
    if (
        t.ndim != 1
        or t.size == 0
        or np.any(t < 0.0)
        or not np.all(np.isfinite(t))
        or z.ndim != 1
        or z.size == 0
        or not np.all(np.isfinite(z))
    ):
        raise ValueError("time and validation shifts must be finite nonempty vectors")

    padding = operator.dimension - retained_dimension
    zeros = np.zeros(padding, dtype=np.complex128)
    sinks = np.vstack(
        [np.concatenate((item.sink, zeros)) for item in columns]
    )
    maximum_dimension = min(maximum_krylov_dimension, operator.dimension)

    def project(vector: ComplexVector) -> ArnoldiResolvent:
        return arnoldi_resolvent(
            operator,
            np.concatenate((vector, zeros)),
            z,
            maximum_dimension=maximum_dimension,
            residual_tolerance=residual_tolerance,
            minimum_dimension=min(16, operator.dimension),
            check_interval=min(8, operator.dimension),
        )

    started = perf_counter()
    combined_projections = tuple(
        project(item.addition + item.removal) for item in columns
    )
    retarded = _matrix_time_from_projections(combined_projections, sinks, t)
    combined_time_residuals = np.asarray(
        [projection.relative_time_residuals(t) for projection in combined_projections],
        dtype=np.float64,
    )
    if include_channels:
        addition_projections = tuple(project(item.addition) for item in columns)
        removal_projections = tuple(project(item.removal) for item in columns)
        addition = _matrix_time_from_projections(addition_projections, sinks, t)
        removal = _matrix_time_from_projections(removal_projections, sinks, t)
        addition_time_residuals = np.asarray(
            [projection.relative_time_residuals(t) for projection in addition_projections],
            dtype=np.float64,
        )
        removal_time_residuals = np.asarray(
            [projection.relative_time_residuals(t) for projection in removal_projections],
            dtype=np.float64,
        )
        defect = float(np.max(abs(retarded - addition - removal)))
    else:
        addition_projections = ()
        removal_projections = ()
        addition_time_residuals = np.empty((0, t.size), dtype=np.float64)
        removal_time_residuals = np.empty((0, t.size), dtype=np.float64)
        addition = np.full_like(retarded, np.nan + 1j * np.nan)
        removal = np.full_like(retarded, np.nan + 1j * np.nan)
        defect = float("nan")
    return ComponentRetardedMatrixTime(
        time=t,
        sites=tuple(item.site for item in columns),
        addition=addition,
        removal=removal,
        retarded=retarded,
        channel_sum_defect=defect,
        combined_projections=combined_projections,
        addition_projections=addition_projections,
        removal_projections=removal_projections,
        combined_relative_time_residuals=combined_time_residuals,
        addition_relative_time_residuals=addition_time_residuals,
        removal_relative_time_residuals=removal_time_residuals,
        wall_seconds=float(perf_counter() - started),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


__all__ = [
    "ComponentRetardedGreen",
    "ComponentRetardedMatrixArnoldiGreen",
    "ComponentRetardedMatrixGreen",
    "ComponentRetardedMatrixTime",
    "ComponentShiftedSolution",
    "ComponentSigmaOperator",
    "component_retarded_green",
    "component_retarded_matrix_green",
    "component_retarded_matrix_green_arnoldi",
    "component_retarded_matrix_time",
    "component_retarded_time",
    "solve_component_shifted",
]

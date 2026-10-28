"""JAX static CCSD and S13 response operators; complex128 throughout."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
import resource
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import component_tensor as _ct
from .component_solver import ComponentCCSolution
from .component_tensor import (
    ComponentEOMContext,
    ComponentEOMSources,
    ComponentStaticProblem,
    component_static_residual,
)
from .newton_krylov import (
    NewtonKrylovPolicy,
    NewtonKrylovStep,
    _forcing_tolerance,
    _preconditioner,
    rapidity_diagonal as static_rapidity_diagonal,
)
from .s135_full import (
    _RankFiveGroup,
    _contraction_signature,
    rapidity_diagonal as response_rapidity_diagonal,
)
from .s135_jax import (
    FullS135JaxPlan,
    JaxFullS135ResponseOperator,
    JaxRuntime,
    _lower_acted_banks,
    _lower_coordinates,
    _lower_prefix_buckets,
    _lower_prefixes,
    _lower_projection_kernels,
    _lower_projections,
    _readonly,
    require_jax_runtime,
)


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class JaxStaticProfile:
    """Cold/warm static residual and JVP timings with allocator telemetry."""

    plan_build_seconds: float
    device_upload_seconds: float
    first_residual_seconds: float
    warm_residual_seconds: tuple[float, ...]
    first_jvp_seconds: float
    warm_jvp_seconds: tuple[float, ...]
    end_to_end_residual_seconds: tuple[float, ...]
    end_to_end_jvp_seconds: tuple[float, ...]
    device_memory: dict[str, int]


class JaxStaticCCSDOperator:
    """Existing component CCSD residual and forward JVP on one JAX device."""

    def __init__(
        self,
        problem: ComponentStaticProblem,
        *,
        jax,
        jnp,
        device,
        runtime: JaxRuntime,
    ):
        require_jax_runtime(runtime, production=False)
        from ._static_engine import get_engine, pack, unpack
        from ._static_schema import lower, make_schema

        self.problem = problem
        self.jax = jax
        self.jnp = jnp
        self.device = device
        self.runtime = runtime
        self.dimension = problem.oracle.num_amplitudes
        self.real_dimension = problem.oracle.num_variables
        started = perf_counter()
        schema = make_schema(problem.coordinates.n, problem.oracle.model.interaction_sites, symmetry=problem.oracle.model.symmetry)
        self.layout, host_data = lower(problem, schema)
        self.engine = get_engine(self.layout)
        self.plan_build_seconds = float(perf_counter() - started)
        started = perf_counter()
        self.data = jax.device_put(host_data, device)
        jax.block_until_ready(self.data)
        self.device_upload_seconds = float(perf_counter() - started)
        self._residual_kernel = partial(self.engine.residual, data=self.data)
        self._jvp_kernel = partial(self.engine.jvp, data=self.data)
        self._unpack_real_kernel = partial(unpack, maps=self.data["tilde"])
        self._pack_complex_kernel = partial(pack, maps=self.data["tilde"])

    @classmethod
    def from_cpu(
        cls,
        problem: ComponentStaticProblem,
        *,
        platform: str = "gpu",
        production: bool = True,
    ) -> "JaxStaticCCSDOperator":
        import jax
        import jax.numpy as jnp
        import jaxlib

        devices = jax.devices(platform)
        if not devices:
            raise RuntimeError(f"JAX reported no {platform} execution device")
        device = devices[0]
        runtime = JaxRuntime(
            jax_version=str(jax.__version__),
            jaxlib_version=str(jaxlib.__version__),
            enable_x64=bool(jax.config.jax_enable_x64),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
        )
        require_jax_runtime(runtime, production=production, platform=platform)
        started = perf_counter()
        instance = cls(
            problem,
            jax=jax,
            jnp=jnp,
            device=device,
            runtime=runtime,
        )
        instance.plan_build_seconds = float(perf_counter() - started)
        return instance

    def device_residual(self, amplitudes: object):
        with self.jax.default_device(self.device):
            value = self.jnp.asarray(amplitudes, dtype=self.jnp.complex128)
        if value.shape != (self.dimension,):
            raise ValueError("static JAX amplitudes have the wrong shape")
        return self._residual_kernel(self.jax.device_put(value, self.device))

    def residual(self, amplitudes: ArrayLike) -> ComplexArray:
        value = np.asarray(amplitudes, dtype=np.complex128)
        if value.shape != (self.dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("static JAX amplitudes have the wrong shape or are nonfinite")
        return np.asarray(self.device_residual(value), dtype=np.complex128)

    def device_jvp(self, amplitudes: object, direction: object):
        with self.jax.default_device(self.device):
            value = self.jnp.asarray(amplitudes, dtype=self.jnp.complex128)
            tangent = self.jnp.asarray(direction, dtype=self.jnp.complex128)
        if value.shape != (self.dimension,) or tangent.shape != value.shape:
            raise ValueError("static JAX JVP arrays have the wrong shape")
        return self._jvp_kernel(self.jax.device_put(value, self.device), self.jax.device_put(tangent, self.device))

    def jvp(self, amplitudes: ArrayLike, direction: ArrayLike) -> ComplexArray:
        value = np.asarray(amplitudes, dtype=np.complex128)
        tangent = np.asarray(direction, dtype=np.complex128)
        if (
            value.shape != (self.dimension,)
            or tangent.shape != value.shape
            or not np.all(np.isfinite(value))
            or not np.all(np.isfinite(tangent))
        ):
            raise ValueError("static JAX JVP arrays have the wrong shape or are nonfinite")
        return np.asarray(
            self.device_jvp(value, tangent), dtype=np.complex128
        )

    def device_memory_record(self) -> dict[str, int]:
        raw = self.device.memory_stats()
        if not raw:
            return {}
        return {
            str(key): int(value)
            for key, value in raw.items()
            if isinstance(value, (int, np.integer))
        }

    def profile(
        self,
        amplitudes: ArrayLike,
        direction: ArrayLike,
        *,
        warm_repeats: int = 5,
    ) -> JaxStaticProfile:
        """Synchronize cold/warm device kernels and host transfer boundaries."""

        if warm_repeats <= 0:
            raise ValueError("warm_repeats must be positive")
        value = np.asarray(amplitudes, dtype=np.complex128)
        tangent = np.asarray(direction, dtype=np.complex128)
        if value.shape != (self.dimension,) or tangent.shape != value.shape:
            raise ValueError("profile arrays do not match the static dimension")
        device_value = self.jax.device_put(value, self.device)
        device_tangent = self.jax.device_put(tangent, self.device)
        device_value.block_until_ready()
        device_tangent.block_until_ready()

        started = perf_counter()
        result = self._residual_kernel(device_value)
        result.block_until_ready()
        first_residual = perf_counter() - started
        warm_residual: list[float] = []
        for _ in range(warm_repeats):
            started = perf_counter()
            result = self._residual_kernel(device_value)
            result.block_until_ready()
            warm_residual.append(perf_counter() - started)

        started = perf_counter()
        result = self._jvp_kernel(device_value, device_tangent)
        result.block_until_ready()
        first_jvp = perf_counter() - started
        warm_jvp: list[float] = []
        for _ in range(warm_repeats):
            started = perf_counter()
            result = self._jvp_kernel(device_value, device_tangent)
            result.block_until_ready()
            warm_jvp.append(perf_counter() - started)

        end_residual: list[float] = []
        end_jvp: list[float] = []
        for _ in range(warm_repeats):
            started = perf_counter()
            self.residual(value)
            end_residual.append(perf_counter() - started)
            started = perf_counter()
            self.jvp(value, tangent)
            end_jvp.append(perf_counter() - started)
        return JaxStaticProfile(
            plan_build_seconds=float(self.plan_build_seconds),
            device_upload_seconds=float(self.device_upload_seconds),
            first_residual_seconds=float(first_residual),
            warm_residual_seconds=tuple(float(item) for item in warm_residual),
            first_jvp_seconds=float(first_jvp),
            warm_jvp_seconds=tuple(float(item) for item in warm_jvp),
            end_to_end_residual_seconds=tuple(float(item) for item in end_residual),
            end_to_end_jvp_seconds=tuple(float(item) for item in end_jvp),
            device_memory=self.device_memory_record(),
        )


def solve_component_ccsd_jax(
    problem: ComponentStaticProblem,
    initial_amplitudes: ArrayLike | None = None,
    *,
    policy: NewtonKrylovPolicy | None = None,
    platform: str = "gpu",
    production: bool = True,
    accepted_step_callback: Callable[[int, ComplexArray, float], None] | None = None,
) -> ComponentCCSolution:
    """Solve the existing tilde-real CCSD equations with device GMRES vectors."""

    controls = NewtonKrylovPolicy() if policy is None else policy
    operator = JaxStaticCCSDOperator.from_cpu(
        problem,
        platform=platform,
        production=production,
    )
    oracle = problem.oracle
    jax = operator.jax
    jnp = operator.jnp
    started = perf_counter()
    if initial_amplitudes is None:
        host_x = np.zeros(oracle.num_variables, dtype=np.float64)
    else:
        initial = np.asarray(initial_amplitudes, dtype=np.complex128)
        if initial.shape != (oracle.num_amplitudes,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial amplitudes do not match the component CCSD problem")
        host_x = np.asarray(oracle.tilde.pack(initial, check=False), dtype=np.float64)
    x = jax.device_put(host_x, operator.device)

    _, preconditioner_diagnostics = _preconditioner(oracle)
    diagonal = np.array(static_rapidity_diagonal(oracle), copy=True)
    magnitudes = np.abs(diagonal)
    diagonal_scale = max(float(np.max(magnitudes, initial=0.0)), 1.0)
    diagonal_threshold = 1.0e-12 * diagonal_scale
    small = magnitudes < diagonal_threshold
    if np.any(small):
        diagonal[small] = diagonal_threshold * np.exp(
            1j * np.angle(diagonal[small] + 1.0e-300)
        )
    device_diagonal = jax.device_put(diagonal, operator.device)
    restart = min(int(controls.gmres_restart), oracle.num_variables)

    def unpack(values):
        return operator._unpack_real_kernel(values)

    def pack(values):
        return operator._pack_complex_kernel(values)

    def residual(values):
        return pack(operator._residual_kernel(unpack(values)))

    def linear_action(amplitudes, direction):
        return pack(
            operator._jvp_kernel(amplitudes, unpack(direction))
        )

    from ._static_gmres import get_cycle

    def precondition(values):
        return pack(unpack(values) / device_diagonal)

    compiled_cycle = get_cycle(operator.layout, restart)
    residual_evaluations = 0
    jvp_evaluations = 0
    history: list[NewtonKrylovStep] = []
    converged = False
    previous_residual_norm: float | None = None
    previous_forcing_tolerance: float | None = None

    for iteration in range(controls.maximum_newton_iterations + 1):
        amplitudes = unpack(x)
        r = residual(x)
        r.block_until_ready()
        residual_evaluations += 1
        norm = float(np.asarray(jnp.linalg.norm(r)))
        if norm <= controls.tolerance:
            converged = True
            break
        if iteration == controls.maximum_newton_iterations:
            break

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
        step = jnp.zeros_like(x)
        linear_history: list[float] = []
        jacobian_step = jnp.zeros_like(r)
        true_linear_relative = np.inf
        for _ in range(controls.gmres_maximum_restart_cycles):
            step = compiled_cycle(amplitudes, -r, step, operator.data, device_diagonal)
            step.block_until_ready()
            jvp_evaluations += restart + 1
            jacobian_step = linear_action(amplitudes, step)
            jacobian_step.block_until_ready()
            jvp_evaluations += 1
            defect = jacobian_step + r
            defect_norm = float(np.asarray(jnp.linalg.norm(defect)))
            true_linear_relative = defect_norm / max(
                norm, np.finfo(float).tiny
            )
            preconditioned_relative = jnp.linalg.norm(
                precondition(defect)
            ) / jnp.maximum(
                jnp.linalg.norm(precondition(r)),
                jnp.finfo(jnp.float64).tiny,
            )
            linear_history.append(float(np.asarray(preconditioned_relative)))
            if defect_norm <= max(
                linear_tolerance * norm,
                linear_absolute_tolerance,
            ):
                break

        merit_slope = float(np.asarray(jnp.dot(r, jacobian_step)))
        linear_accepted = bool(
            np.isfinite(true_linear_relative)
            and true_linear_relative
            <= max(
                linear_tolerance,
                linear_absolute_tolerance / max(norm, np.finfo(float).tiny),
            )
            * (1.0 + np.sqrt(np.finfo(float).eps))
            and np.isfinite(merit_slope)
            and merit_slope < 0.0
        )
        if not linear_accepted:
            history.append(
                NewtonKrylovStep(
                    iteration=iteration,
                    residual_norm=norm,
                    gmres_relative_tolerance=linear_tolerance,
                    gmres_info=1,
                    gmres_iterations=len(linear_history) * restart,
                    gmres_preconditioned_residual_history=tuple(linear_history),
                    true_linear_relative_residual=true_linear_relative,
                    step_norm=float(np.asarray(jnp.linalg.norm(step))),
                    damping=0.0,
                    accepted_residual_norm=norm,
                )
            )
            break

        damping = 1.0
        accepted_norm = np.inf
        accepted = False
        for _ in range(controls.maximum_line_search_steps):
            trial_x = x + damping * step
            trial_r = residual(trial_x)
            trial_r.block_until_ready()
            residual_evaluations += 1
            accepted_norm = float(np.asarray(jnp.linalg.norm(trial_r)))
            if accepted_norm <= (1.0 - controls.armijo * damping) * norm:
                x = trial_x
                accepted = True
                if accepted_step_callback is not None:
                    accepted_step_callback(
                        iteration,
                        np.asarray(unpack(x), dtype=np.complex128),
                        accepted_norm,
                    )
                break
            damping *= 0.5
        history.append(
            NewtonKrylovStep(
                iteration=iteration,
                residual_norm=norm,
                gmres_relative_tolerance=linear_tolerance,
                gmres_info=0,
                gmres_iterations=len(linear_history) * restart,
                gmres_preconditioned_residual_history=tuple(linear_history),
                true_linear_relative_residual=true_linear_relative,
                step_norm=float(np.asarray(jnp.linalg.norm(step))),
                damping=float(damping),
                accepted_residual_norm=accepted_norm,
            )
        )
        if not accepted:
            break
        previous_residual_norm = norm
        previous_forcing_tolerance = linear_tolerance

    variables = np.asarray(x, dtype=np.float64)
    amplitudes_host = np.asarray(unpack(x), dtype=np.complex128)
    final = component_static_residual(problem, amplitudes_host)
    retained_norm = float(np.linalg.norm(final.retained))
    packed_norm = float(
        np.linalg.norm(oracle.tilde.pack(final.retained, check=False))
    )
    return ComponentCCSolution(
        problem=problem,
        variables=variables,
        amplitudes=amplitudes_host,
        residual=final,
        residual_norm=retained_norm,
        packed_residual_norm=packed_norm,
        reference_residual=float(abs(final.reference)),
        rank6=None,
        converged=bool(converged and packed_norm <= controls.tolerance),
        iterations=len(history),
        residual_evaluations=residual_evaluations,
        jvp_evaluations=jvp_evaluations,
        history=tuple(history),
        preconditioner=preconditioner_diagnostics,
        wall_seconds=float(perf_counter() - started),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


@dataclass(frozen=True, slots=True)
class _CompiledRetainedAction:
    context: ComponentEOMContext
    groups: tuple[object, ...]
    storage_bytes: int


@dataclass(frozen=True, slots=True)
class JaxShiftedSolution:
    """One device-GMRES solution with an independently evaluated residual."""

    solution: ComplexArray
    relative_residual: float
    converged: bool
    info: int
    wall_seconds: float


def _retained_plan(context: ComponentEOMContext) -> FullS135JaxPlan:
    grouped = {}
    contractions = {}
    for plan in context.response_plan:
        if plan.input_degree not in (1, 3):
            continue
        creation_degree = sum(form.degrees[0] for form in plan.creation_factors)
        output_degree = creation_degree + plan.input_degree - len(plan.contractions)
        if output_degree not in (1, 3):
            continue
        key = (
            plan.input_degree, _contraction_signature(plan.contractions),
            creation_degree, output_degree,
        )
        grouped.setdefault(key, []).append(_ct._Term(plan.coefficient, plan.creation_factors))
        contractions[key] = plan.contractions
    groups = tuple(
        _RankFiveGroup(
            input_degree=key[0], contractions=contractions[key],
            creation=_ct._materialize_terms_raw(
                terms, context.problem.coordinates.n, {key[2]}, batch_size=16,
            ),
            creation_degree=key[2], acted_degree=key[0] - len(contractions[key]),
            output_degree=key[3],
        )
        for key, terms in grouped.items()
    )
    action = _CompiledRetainedAction(
        context=context,
        groups=groups,
        storage_bytes=int(
            sum(
                value.nbytes
                for group in groups
                for value in group.creation.blocks.values()
            )
        ),
    )
    coordinates = _lower_coordinates(context.problem.coordinates)
    prefixes, prefix_indices = _lower_prefixes(action, context.problem.coordinates.n)
    projections = _lower_projections(action, prefix_indices)
    prefix_buckets, prefix_outputs = _lower_prefix_buckets(
        prefixes, tuple(block.key for block in coordinates)
    )
    acted_banks, acted_rows = _lower_acted_banks(prefix_buckets)
    projection_kernels = _lower_projection_kernels(
        coordinates,
        projections,
        prefix_outputs,
        acted_rows,
    )
    diagonal = _readonly(
        response_rapidity_diagonal(
            context.problem.static.oracle.rapidities,
            context.problem.coordinates,
        ),
        np.dtype(np.complex128),
    )
    dimension = context.problem.coordinates.dimension
    return FullS135JaxPlan(
        n=context.problem.coordinates.n,
        p_dimension=dimension,
        s5_dimension=0,
        coordinates=coordinates,
        prefixes=prefixes,
        projections=projections,
        prefix_buckets=prefix_buckets,
        acted_banks=acted_banks,
        projection_kernels=projection_kernels,
        rapidity_diagonal=diagonal,
        creation_bytes=action.storage_bytes,
        projected_creation_bytes=int(
            sum(
                term.creation.nbytes
                for kernel in projection_kernels
                for term in kernel.terms
            )
        ),
    )


class JaxRetainedEOMOperator(JaxFullS135ResponseOperator):
    """Exact retained ``S1+S3`` right action on one JAX device."""

    def _device_action(self, vector):
        """Expose the exact linear map for transposition by device GMRES."""

        return self.jax.jvp(
            super()._device_action,
            (self.jnp.zeros_like(vector),),
            (vector,),
        )[1]

    @classmethod
    def from_cpu(
        cls,
        context: ComponentEOMContext,
        *,
        platform: str = "gpu",
        production: bool = True,
        projection_tile_size: int = 16384,
    ) -> "JaxRetainedEOMOperator":
        import jax
        import jax.numpy as jnp
        import jaxlib

        devices = jax.devices(platform)
        if not devices:
            raise RuntimeError(f"JAX reported no {platform} execution device")
        device = devices[0]
        runtime = JaxRuntime(
            jax_version=str(jax.__version__),
            jaxlib_version=str(jaxlib.__version__),
            enable_x64=bool(jax.config.jax_enable_x64),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
        )
        require_jax_runtime(runtime, production=production, platform=platform)
        started = perf_counter()
        plan = _retained_plan(context)
        plan_seconds = perf_counter() - started
        instance = cls(
            plan,
            jax=jax,
            jnp=jnp,
            device=device,
            runtime=runtime,
            projection_tile_size=projection_tile_size,
        )
        instance.plan_build_seconds = float(plan_seconds)
        return instance

    def device_matvec(self, vector: object):
        """Apply the retained action without crossing a host array boundary."""

        with self.jax.default_device(self.device):
            value = self.jnp.asarray(vector, dtype=self.jnp.complex128)
        if value.shape != (self.dimension,):
            raise ValueError("retained JAX input has the wrong shape")
        return self._device_action(value)

    def _solve_shifted_device(
        self,
        shift: complex,
        device_source,
        *,
        rtol: float,
        restart: int,
        maxiter: int,
        initial_solution=None,
    ):
        from jax.scipy.sparse.linalg import gmres

        def action(vector):
            return shift * vector - self._device_action(vector)

        started = perf_counter()
        solution, raw_info = gmres(
            action,
            device_source,
            x0=initial_solution,
            tol=float(rtol),
            atol=0.0,
            restart=int(restart),
            maxiter=int(maxiter),
            solve_method="incremental",
        )
        solution.block_until_ready()
        residual = action(solution) - device_source
        relative = self.jnp.linalg.norm(residual) / self.jnp.maximum(
            self.jnp.linalg.norm(device_source),
            self.jnp.finfo(self.jnp.float64).tiny,
        )
        relative.block_until_ready()
        return (
            solution,
            float(np.asarray(relative)),
            int(np.asarray(raw_info)),
            float(perf_counter() - started),
        )

    def solve_shifted(
        self,
        shift: complex,
        source: ArrayLike,
        *,
        rtol: float = 1.0e-10,
        restart: int = 40,
        maxiter: int = 200,
        initial_solution: ArrayLike | None = None,
    ) -> JaxShiftedSolution:
        """Solve ``(z I-A)x=b`` with all Krylov vectors on the JAX device."""

        z = complex(shift)
        b = np.asarray(source, dtype=np.complex128)
        if (
            b.shape != (self.dimension,)
            or not np.all(np.isfinite(b))
            or not np.isfinite(z.real)
            or not np.isfinite(z.imag)
            or not np.isfinite(rtol)
            or rtol <= 0.0
            or restart <= 0
            or maxiter <= 0
        ):
            raise ValueError("shifted JAX solve inputs are invalid")
        x0 = None
        if initial_solution is not None:
            host_x0 = np.asarray(initial_solution, dtype=np.complex128)
            if host_x0.shape != b.shape or not np.all(np.isfinite(host_x0)):
                raise ValueError("initial shifted JAX solution is invalid")
            x0 = self.jax.device_put(host_x0, self.device)
        device_source = self.jax.device_put(b, self.device)

        solution, relative_value, info, wall_seconds = self._solve_shifted_device(
            z,
            device_source,
            initial_solution=x0,
            rtol=rtol,
            restart=restart,
            maxiter=maxiter,
        )
        return JaxShiftedSolution(
            solution=np.asarray(solution, dtype=np.complex128),
            relative_residual=relative_value,
            converged=bool(info == 0 and relative_value <= 10.0 * rtol),
            info=info,
            wall_seconds=wall_seconds,
        )


@dataclass(frozen=True, slots=True)
class JaxRetardedMatrixGreen:
    """Retained matrix response and true device-solve residuals."""

    omega: NDArray[np.float64]
    eta: float
    sites: tuple[int, ...]
    addition: ComplexArray
    removal: ComplexArray
    retarded: ComplexArray
    addition_relative_residuals: NDArray[np.float64]
    removal_relative_residuals: NDArray[np.float64]
    wall_seconds: float
    peak_rss_kib: int
    device_memory: dict[str, int]


@dataclass(frozen=True, slots=True)
class JaxRetardedMatrixArnoldiGreen:
    """Shift-invariant retained response from device-resident Arnoldi bases."""

    omega: NDArray[np.float64]
    eta: float
    sites: tuple[int, ...]
    addition: ComplexArray
    removal: ComplexArray
    retarded: ComplexArray
    addition_relative_residuals: NDArray[np.float64]
    removal_relative_residuals: NDArray[np.float64]
    krylov_dimensions: tuple[int, ...]
    wall_seconds: float
    peak_rss_kib: int
    device_memory: dict[str, int]


@dataclass(frozen=True, slots=True)
class JaxRetardedMatrixTime:
    """Device-Arnoldi time response with compact shifted residual checks."""

    time: NDArray[np.float64]
    sites: tuple[int, ...]
    retarded: ComplexArray
    relative_residuals: NDArray[np.float64]
    krylov_dimensions: tuple[int, ...]
    orthogonality_defects: tuple[float, ...]
    projected_backend: str
    wall_seconds: float
    peak_rss_kib: int
    device_memory: dict[str, int]


def component_retarded_matrix_green_jax(
    context: ComponentEOMContext,
    sources: Sequence[ComponentEOMSources],
    omega: ArrayLike,
    eta: float,
    *,
    platform: str = "gpu",
    production: bool = True,
    rtol: float = 1.0e-10,
    restart: int = 40,
    maxiter: int = 200,
) -> JaxRetardedMatrixGreen:
    """Solve every retained source/channel with device-resident Krylov vectors."""

    columns = tuple(sources)
    if not columns or len({item.site for item in columns}) != len(columns):
        raise ValueError("retained JAX sources must be nonempty and unique")
    if any(item.charge != context.problem.charge for item in columns):
        raise ValueError("retained JAX sources must match the context charge")
    dimension = context.problem.coordinates.dimension
    if any(
        item.addition.shape != (dimension,)
        or item.removal.shape != (dimension,)
        or item.sink.shape != (dimension,)
        for item in columns
    ):
        raise ValueError("retained JAX sources and sinks have incompatible shapes")
    frequency = np.asarray(omega, dtype=np.float64)
    broadening = float(eta)
    if (
        frequency.ndim != 1
        or frequency.size == 0
        or not np.all(np.isfinite(frequency))
        or not np.isfinite(broadening)
        or broadening <= 0.0
    ):
        raise ValueError("retained JAX frequency grid and eta are invalid")

    operator = JaxRetainedEOMOperator.from_cpu(
        context,
        platform=platform,
        production=production,
    )
    device_sinks = operator.jax.device_put(
        np.vstack([item.sink for item in columns]),
        operator.device,
    )
    shape = (frequency.size, len(columns), len(columns))
    addition = np.empty(shape, dtype=np.complex128)
    removal = np.empty(shape, dtype=np.complex128)
    addition_residuals = np.empty(
        (len(columns), frequency.size), dtype=np.float64
    )
    removal_residuals = np.empty_like(addition_residuals)
    started = perf_counter()
    for source_index, source in enumerate(columns):
        for channel, values, residuals, host_source in (
            ("addition", addition, addition_residuals, source.addition),
            ("removal", removal, removal_residuals, source.removal),
        ):
            device_source = operator.jax.device_put(host_source, operator.device)
            previous = None
            projected = []
            for frequency_index, value in enumerate(frequency):
                solution, relative, info, _ = operator._solve_shifted_device(
                    complex(float(value), broadening),
                    device_source,
                    initial_solution=previous,
                    rtol=rtol,
                    restart=restart,
                    maxiter=maxiter,
                )
                if info != 0 or relative > 10.0 * rtol:
                    raise RuntimeError(
                        "retained JAX shifted solve did not converge: "
                        f"source={source.site}, channel={channel}, "
                        f"frequency_index={frequency_index}, omega={value:.16g}, "
                        f"info={info}, relative_residual={relative:.6e}, "
                        f"required<={10.0 * rtol:.6e}"
                    )
                previous = solution
                residuals[source_index, frequency_index] = relative
                projected.append(device_sinks @ solution)
            values[:, :, source_index] = np.asarray(
                operator.jnp.stack(projected, axis=0),
                dtype=np.complex128,
            )
    return JaxRetardedMatrixGreen(
        omega=frequency,
        eta=broadening,
        sites=tuple(item.site for item in columns),
        addition=addition,
        removal=removal,
        retarded=addition + removal,
        addition_relative_residuals=addition_residuals,
        removal_relative_residuals=removal_residuals,
        wall_seconds=float(perf_counter() - started),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        device_memory=operator.device_memory_record(),
    )


def component_retarded_matrix_green_arnoldi_jax(
    context: ComponentEOMContext,
    sources: Sequence[ComponentEOMSources],
    omega: ArrayLike,
    eta: float,
    *,
    platform: str = "gpu",
    production: bool = True,
    maximum_krylov_dimension: int = 320,
    residual_tolerance: float = 1.0e-8,
) -> JaxRetardedMatrixArnoldiGreen:
    """Project every shift and sink from one device basis per source/channel."""

    columns = tuple(sources)
    if not columns or len({item.site for item in columns}) != len(columns):
        raise ValueError("retained JAX sources must be nonempty and unique")
    if any(item.charge != context.problem.charge for item in columns):
        raise ValueError("retained JAX sources must match the context charge")
    dimension = context.problem.coordinates.dimension
    if any(
        item.addition.shape != (dimension,)
        or item.removal.shape != (dimension,)
        or item.sink.shape != (dimension,)
        for item in columns
    ):
        raise ValueError("retained JAX sources and sinks have incompatible shapes")
    frequency = np.asarray(omega, dtype=np.float64)
    broadening = float(eta)
    if (
        frequency.ndim != 1
        or frequency.size == 0
        or not np.all(np.isfinite(frequency))
        or not np.isfinite(broadening)
        or broadening <= 0.0
        or maximum_krylov_dimension <= 0
        or not np.isfinite(residual_tolerance)
        or residual_tolerance <= 0.0
    ):
        raise ValueError("retained JAX Arnoldi inputs are invalid")

    operator = JaxRetainedEOMOperator.from_cpu(
        context,
        platform=platform,
        production=production,
    )
    jax = operator.jax
    jnp = operator.jnp
    maximum_dimension = min(int(maximum_krylov_dimension), dimension)
    device_sinks = jax.device_put(
        np.vstack([item.sink for item in columns]), operator.device
    )
    device_shifts = jax.device_put(
        frequency.astype(np.complex128) + 1j * broadening,
        operator.device,
    )
    tiny = jnp.finfo(jnp.float64).tiny

    def arnoldi(source):
        source_norm = jnp.linalg.norm(source)
        basis = jnp.zeros(
            (dimension, maximum_dimension + 1), dtype=jnp.complex128
        )
        basis = basis.at[:, 0].set(source / jnp.maximum(source_norm, tiny))
        hessenberg = jnp.zeros(
            (maximum_dimension + 1, maximum_dimension),
            dtype=jnp.complex128,
        )

        def step(index, state):
            vectors, matrix = state
            work = operator._device_action(vectors[:, index])
            first = jnp.conj(vectors).T @ work
            work = work - vectors @ first
            second = jnp.conj(vectors).T @ work
            work = work - vectors @ second
            coefficients = first + second
            norm = jnp.linalg.norm(work)
            matrix = matrix.at[:, index].set(coefficients)
            matrix = matrix.at[index + 1, index].set(norm)
            vectors = vectors.at[:, index + 1].set(
                work / jnp.maximum(norm, tiny)
            )
            return vectors, matrix

        basis, hessenberg = jax.lax.fori_loop(
            0,
            maximum_dimension,
            step,
            (basis, hessenberg),
        )
        return basis, hessenberg, source_norm

    compiled_arnoldi = jax.jit(arnoldi)
    shape = (frequency.size, len(columns), len(columns))
    addition = np.empty(shape, dtype=np.complex128)
    removal = np.empty(shape, dtype=np.complex128)
    addition_residuals = np.empty(
        (len(columns), frequency.size), dtype=np.float64
    )
    removal_residuals = np.empty_like(addition_residuals)
    dimensions: list[int] = []
    started = perf_counter()
    for source_index, source in enumerate(columns):
        for channel, values, residuals, host_source in (
            ("addition", addition, addition_residuals, source.addition),
            ("removal", removal, removal_residuals, source.removal),
        ):
            device_source = jax.device_put(host_source, operator.device)
            basis, hessenberg, source_norm = compiled_arnoldi(device_source)
            basis.block_until_ready()
            host_hessenberg = np.asarray(hessenberg, dtype=np.complex128)
            subdiagonal = np.abs(
                host_hessenberg[
                    np.arange(1, maximum_dimension + 1),
                    np.arange(maximum_dimension),
                ]
            )
            breakdown = np.flatnonzero(subdiagonal <= 1.0e-13)
            m = (
                int(breakdown[0] + 1)
                if breakdown.size
                else maximum_dimension
            )
            dimensions.append(m)
            reduced = hessenberg[:m, :m]
            with jax.default_device(operator.device):
                rhs = jnp.zeros(m, dtype=jnp.complex128).at[0].set(source_norm)
                identity = jnp.eye(m, dtype=jnp.complex128)
            tail = jnp.abs(hessenberg[m, m - 1])
            projected_sinks = device_sinks @ basis[:, :m]

            def project(shift):
                coefficients = jnp.linalg.solve(
                    shift * identity - reduced, rhs
                )
                relative = tail * jnp.abs(coefficients[-1]) / jnp.maximum(
                    source_norm, tiny
                )
                return projected_sinks @ coefficients, relative

            device_values, device_residuals = jax.jit(jax.vmap(project))(
                device_shifts
            )
            device_values.block_until_ready()
            host_residuals = np.asarray(device_residuals, dtype=np.float64)
            maximum_residual = float(np.max(host_residuals))
            if maximum_residual > residual_tolerance:
                raise RuntimeError(
                    "retained JAX Arnoldi projection did not converge: "
                    f"source={source.site}, channel={channel}, m={m}, "
                    f"relative_residual={maximum_residual:.6e}, "
                    f"required<={residual_tolerance:.6e}"
                )
            values[:, :, source_index] = np.asarray(
                device_values, dtype=np.complex128
            )
            residuals[source_index] = host_residuals

    return JaxRetardedMatrixArnoldiGreen(
        omega=frequency,
        eta=broadening,
        sites=tuple(item.site for item in columns),
        addition=addition,
        removal=removal,
        retarded=addition + removal,
        addition_relative_residuals=addition_residuals,
        removal_relative_residuals=removal_residuals,
        krylov_dimensions=tuple(dimensions),
        wall_seconds=float(perf_counter() - started),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        device_memory=operator.device_memory_record(),
    )


def component_retarded_matrix_time_jax(
    context: ComponentEOMContext,
    sources: Sequence[ComponentEOMSources],
    time: ArrayLike,
    validation_shifts: ArrayLike,
    *,
    platform: str = "gpu",
    production: bool = True,
    maximum_krylov_dimension: int = 160,
    residual_tolerance: float = 1.0e-10,
) -> JaxRetardedMatrixTime:
    """Propagate combined source columns with a device-resident Arnoldi basis."""

    columns = tuple(sources)
    if not columns or len({item.site for item in columns}) != len(columns):
        raise ValueError("retained JAX sources must be nonempty and unique")
    if any(item.charge != context.problem.charge for item in columns):
        raise ValueError("retained JAX sources must match the context charge")
    dimension = context.problem.coordinates.dimension
    if any(
        item.addition.shape != (dimension,)
        or item.removal.shape != (dimension,)
        or item.sink.shape != (dimension,)
        for item in columns
    ):
        raise ValueError("retained JAX sources and sinks have incompatible shapes")
    t = np.asarray(time, dtype=np.float64)
    shifts = np.asarray(validation_shifts, dtype=np.complex128)
    if (
        t.ndim != 1
        or t.size == 0
        or np.any(t < 0.0)
        or not np.all(np.isfinite(t))
        or shifts.ndim != 1
        or shifts.size == 0
        or not np.all(np.isfinite(shifts))
        or maximum_krylov_dimension <= 0
        or residual_tolerance <= 0.0
    ):
        raise ValueError("retained JAX time/Arnoldi inputs are invalid")
    uniform_time_grid = t.size == 1 or np.allclose(
        np.diff(t), t[1] - t[0], rtol=1.0e-12, atol=1.0e-14
    )

    operator = JaxRetainedEOMOperator.from_cpu(
        context,
        platform=platform,
        production=production,
    )
    jax = operator.jax
    jnp = operator.jnp
    maximum_dimension = min(int(maximum_krylov_dimension), dimension)
    device_sinks = jax.device_put(
        np.vstack([item.sink for item in columns]), operator.device
    )
    tiny = jnp.finfo(jnp.float64).tiny

    def arnoldi(source):
        source_norm = jnp.linalg.norm(source)
        basis = jnp.zeros(
            (dimension, maximum_dimension + 1), dtype=jnp.complex128
        )
        basis = basis.at[:, 0].set(source / jnp.maximum(source_norm, tiny))
        hessenberg = jnp.zeros(
            (maximum_dimension + 1, maximum_dimension),
            dtype=jnp.complex128,
        )

        def step(index, state):
            vectors, matrix = state
            work = operator._device_action(vectors[:, index])
            first = jnp.conj(vectors).T @ work
            work = work - vectors @ first
            second = jnp.conj(vectors).T @ work
            work = work - vectors @ second
            coefficients = first + second
            norm = jnp.linalg.norm(work)
            matrix = matrix.at[:, index].set(coefficients)
            matrix = matrix.at[index + 1, index].set(norm)
            vectors = vectors.at[:, index + 1].set(
                work / jnp.maximum(norm, tiny)
            )
            return vectors, matrix

        basis, hessenberg = jax.lax.fori_loop(
            0,
            maximum_dimension,
            step,
            (basis, hessenberg),
        )
        return basis, hessenberg, source_norm

    compiled_arnoldi = jax.jit(arnoldi)
    shape = (t.size, len(columns), len(columns))
    retarded = np.empty(shape, dtype=np.complex128)
    residuals = np.empty((len(columns), shifts.size), dtype=np.float64)
    dimensions: list[int] = []
    orthogonality: list[float] = []
    started = perf_counter()
    for source_index, source in enumerate(columns):
        device_source = jax.device_put(
            source.addition + source.removal,
            operator.device,
        )
        basis, hessenberg, source_norm = compiled_arnoldi(device_source)
        basis.block_until_ready()
        host_hessenberg = np.asarray(hessenberg, dtype=np.complex128)
        host_norm = float(np.asarray(source_norm))
        subdiagonal = np.abs(
            host_hessenberg[
                np.arange(1, maximum_dimension + 1),
                np.arange(maximum_dimension),
            ]
        )
        breakdown = np.flatnonzero(subdiagonal <= 1.0e-13)
        m = (
            int(breakdown[0] + 1)
            if breakdown.size
            else maximum_dimension
        )
        dimensions.append(m)
        reduced = hessenberg[:m, :m]
        with jax.default_device(operator.device):
            rhs = jnp.zeros(m, dtype=jnp.complex128).at[0].set(source_norm)
            identity = jnp.eye(m, dtype=jnp.complex128)
        tail = jnp.abs(hessenberg[m, m - 1])

        def shifted_residual(shift):
            coefficients = jnp.linalg.solve(shift * identity - reduced, rhs)
            return tail * jnp.abs(coefficients[-1]) / jnp.maximum(
                source_norm, tiny
            )

        device_residuals = jax.jit(jax.vmap(shifted_residual))(
            jax.device_put(shifts, operator.device)
        )
        device_residuals.block_until_ready()
        residuals[source_index] = np.asarray(device_residuals, dtype=np.float64)
        active_basis = basis[:, :m]
        defect = jnp.linalg.norm(
            jnp.conj(active_basis).T @ active_basis
            - identity
        )
        defect.block_until_ready()
        orthogonality.append(float(np.asarray(defect)))
        projected_sinks = device_sinks @ active_basis

        if uniform_time_grid:
            initial_time = float(t[0])
            time_step = 0.0 if t.size == 1 else float(t[1] - t[0])

            def propagate_uniform():
                state = (
                    rhs
                    if initial_time == 0.0
                    else jax.scipy.linalg.expm(-1j * initial_time * reduced)
                    @ rhs
                )
                transition = jax.scipy.linalg.expm(-1j * time_step * reduced)

                def advance(current, _):
                    response = -1j * (projected_sinks @ current)
                    return transition @ current, response

                _, response = jax.lax.scan(
                    advance, state, None, length=t.size
                )
                return response

            device_response = jax.jit(propagate_uniform)()
        else:
            def propagate(value):
                state = jax.scipy.linalg.expm(-1j * value * reduced) @ rhs
                return -1j * (projected_sinks @ state)

            device_response = jax.jit(jax.vmap(propagate))(
                jax.device_put(t, operator.device)
            )
        device_response.block_until_ready()
        retarded[:, :, source_index] = np.asarray(
            device_response, dtype=np.complex128
        )
    if float(np.max(residuals)) > residual_tolerance:
        raise RuntimeError("retained JAX Arnoldi did not pass the shifted residual gate")
    return JaxRetardedMatrixTime(
        time=t,
        sites=tuple(item.site for item in columns),
        retarded=retarded,
        relative_residuals=residuals,
        krylov_dimensions=tuple(dimensions),
        orthogonality_defects=tuple(orthogonality),
        projected_backend=(
            "jax_uniform_step_expm"
            if uniform_time_grid
            else "jax_batched_expm"
        ),
        wall_seconds=float(perf_counter() - started),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        device_memory=operator.device_memory_record(),
    )


__all__ = [
    "JaxRetainedEOMOperator",
    "JaxRetardedMatrixArnoldiGreen",
    "JaxRetardedMatrixGreen",
    "JaxRetardedMatrixTime",
    "JaxShiftedSolution",
    "JaxStaticCCSDOperator",
    "JaxStaticProfile",
    "component_retarded_matrix_green_arnoldi_jax",
    "component_retarded_matrix_green_jax",
    "component_retarded_matrix_time_jax",
    "solve_component_ccsd_jax",
]

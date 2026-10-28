"""Disk-backed one-sided Arnoldi with exact interruption and resume."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import resource
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

from c3l.green.krylov import ArnoldiResolvent, MatrixFreeAction, _reduced_residuals


if TYPE_CHECKING:
    from c3l.cc.component_tensor import ComponentEOMContext


ComplexMatrix = NDArray[np.complex128]
CPU_ORTHOGONALIZATION_BACKEND = "cpu_scipy_blas_cgs2"
ORTHOGONALIZATION_PROFILE_FIELDS = (
    "work_upload_seconds",
    "basis_upload_seconds",
    "cache_fill_upload_seconds",
    "stream_upload_seconds",
    "kernel_seconds",
    "result_download_seconds",
    "panel_dispatches",
    "work_upload_bytes",
    "cache_fill_bytes",
    "stream_upload_bytes",
    "cache_hit_bytes",
    "transient_panel_reuse_bytes",
    "logical_basis_bytes",
    "result_download_bytes",
    "cached_basis_bytes",
    "wall_seconds",
)


@dataclass(slots=True)
class CheckpointedArnoldiResult:
    projection: ArnoldiResolvent
    sink_projections: ComplexMatrix
    orthogonalization_wall_seconds: NDArray[np.float64]
    checkpoint_dimensions: tuple[int, ...]
    directory: Path
    orthogonalization_backend: str = CPU_ORTHOGONALIZATION_BACKEND
    orthogonalization_profile: dict[str, NDArray[np.float64]] = field(
        default_factory=dict
    )
    orthogonalization_setup_wall_seconds: float = 0.0

    def transfer(
        self, shifts: ArrayLike, *, dimension: int | None = None
    ) -> ComplexMatrix:
        m = self.projection._dimension(dimension)
        y = self.projection.coefficients(shifts, dimension=m)
        return np.asarray(y @ self.sink_projections[:, :m].T, dtype=np.complex128)


def require_resources(
    *,
    required_memory_bytes: int,
    required_disk_bytes: int,
    available_memory_bytes: int,
    free_disk_bytes: int,
) -> None:
    """Reject a run when either declared resource is one byte short."""

    if available_memory_bytes < required_memory_bytes:
        raise MemoryError(
            f"insufficient memory: need {required_memory_bytes}, "
            f"have {available_memory_bytes} bytes"
        )
    if free_disk_bytes < required_disk_bytes:
        raise OSError(
            f"insufficient disk: need {required_disk_bytes}, "
            f"have {free_disk_bytes} bytes"
        )


def _update_array_hash(digest: "hashlib._Hash", array: np.ndarray) -> None:
    value = np.ascontiguousarray(array)
    digest.update(str(value.shape).encode())
    digest.update(value.dtype.str.encode())
    digest.update(value.view(np.uint8))


def _update_eom_array_hash(digest: "hashlib._Hash", array: np.ndarray) -> None:
    value = np.ascontiguousarray(array)
    digest.update(repr((value.shape, value.dtype.str)).encode())
    digest.update(value.view(np.uint8))


def eom_checkpoint_identity(context: ComponentEOMContext) -> str:
    """Identify the EOM context used to construct a checkpointed operator.

    Call once when preparing a checkpointed run and pass the result as the
    Arnoldi ``provenance``. Ordinary kernel construction does not need it.
    The byte encoding is unchanged so existing restart identities stay valid.
    Expanded EOM tensors and materialized lazy shells are never hashed.
    """

    from c3l.cc.component_tensor import ComponentCoordinateShell

    digest = hashlib.sha256(b"cclens-full-q5-inputs-v2")
    oracle = context.problem.static.oracle
    _update_eom_array_hash(digest, oracle.rapidities)
    for key, values in sorted(context.amplitudes.blocks.items()):
        digest.update(repr(key).encode())
        _update_eom_array_hash(digest, values)
    for record in oracle.products:
        factors = tuple(factor.components for factor in record.factors)
        digest.update(repr((record.coefficient, factors)).encode())
    for coordinates in (context.problem.coordinates, context.problem.s5_coordinates):
        digest.update(repr((coordinates.n, coordinates.dimension)).encode())
        if isinstance(coordinates, ComponentCoordinateShell):
            shell = (coordinates.response_spin, coordinates.charge, coordinates.creator_rank)
            digest.update(repr(shell).encode())
        else:
            for block in coordinates.blocks:
                digest.update(repr(block.key).encode())
                _update_eom_array_hash(digest, block.vector_positions)
                _update_eom_array_hash(digest, block.indices)
    return digest.hexdigest()


def _binding(
    b: np.ndarray,
    sinks: np.ndarray,
    z: np.ndarray,
    *,
    n: int,
    maximum_dimension: int,
    checkpoint_dimensions: tuple[int, ...],
    residual_tolerance: float,
    breakdown_tolerance: float,
    provenance: str,
) -> str:
    digest = hashlib.sha256()
    for value in (b, sinks, z):
        _update_array_hash(digest, value)
    values = (
        n,
        maximum_dimension,
        checkpoint_dimensions,
        residual_tolerance,
        breakdown_tolerance,
        provenance,
    )
    digest.update(repr(values).encode())
    return digest.hexdigest()


def _write_state(path: Path, **arrays: object) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def checkpointed_arnoldi_resolvent(
    operator: MatrixFreeAction,
    rhs: ArrayLike,
    sinks: ArrayLike,
    shifts: ArrayLike,
    directory: Path,
    *,
    maximum_dimension: int,
    checkpoint_dimensions: tuple[int, ...],
    residual_tolerance: float,
    breakdown_tolerance: float = 1.0e-14,
    maximum_new_steps: int | None = None,
    provenance: str = "",
    orthogonalizer: Any | None = None,
) -> CheckpointedArnoldiResult:
    r"""Build $K_m(A,b)$ in a mapped basis and resume its last checkpoint."""

    if not hasattr(operator, "dimension") or not hasattr(operator, "matvec"):
        raise TypeError("operator must expose dimension and matvec")
    n = int(operator.dimension)
    b = np.asarray(rhs, dtype=np.complex128)
    rows = np.asarray(sinks, dtype=np.complex128)
    z = np.asarray(shifts, dtype=np.complex128)
    if b.shape != (n,) or not np.all(np.isfinite(b)):
        raise ValueError("rhs must be a finite vector matching the operator")
    if rows.ndim != 2 or rows.shape[1] != n or not np.all(np.isfinite(rows)):
        raise ValueError("sinks must be a finite matrix with one row per sink")
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("shifts must be a finite nonempty vector")
    if (
        isinstance(maximum_dimension, bool)
        or not isinstance(maximum_dimension, int)
        or not 1 <= maximum_dimension <= n
    ):
        raise ValueError("maximum_dimension must lie in [1, operator.dimension]")
    checks = tuple(checkpoint_dimensions)
    if (
        not checks
        or checks != tuple(sorted(set(checks)))
        or any(isinstance(m, bool) or not isinstance(m, int) for m in checks)
        or checks[0] < 1
        or checks[-1] > maximum_dimension
    ):
        raise ValueError("checkpoint_dimensions must be sorted unique valid prefixes")
    tolerance = float(residual_tolerance)
    breakdown_threshold = float(breakdown_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if not np.isfinite(breakdown_threshold) or breakdown_threshold <= 0.0:
        raise ValueError("breakdown_tolerance must be finite and positive")
    if maximum_new_steps is not None and (
        isinstance(maximum_new_steps, bool)
        or not isinstance(maximum_new_steps, int)
        or maximum_new_steps < 0
    ):
        raise ValueError("maximum_new_steps must be a nonnegative integer")
    beta = float(np.linalg.norm(b))
    if beta == 0.0:
        raise ValueError("rhs must be nonzero")
    orthogonalization_backend = CPU_ORTHOGONALIZATION_BACKEND
    if orthogonalizer is not None:
        orthogonalization_backend = getattr(orthogonalizer, "backend", None)
        if (
            not isinstance(orthogonalization_backend, str)
            or not orthogonalization_backend
            or not callable(getattr(orthogonalizer, "orthogonalize", None))
        ):
            raise TypeError("orthogonalizer has an invalid Arnoldi interface")

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    basis_path = root / "basis.npy"
    state_path = root / "state.npz"
    key = _binding(
        b,
        rows,
        z,
        n=n,
        maximum_dimension=maximum_dimension,
        checkpoint_dimensions=checks,
        residual_tolerance=tolerance,
        breakdown_tolerance=breakdown_threshold,
        provenance=provenance,
    )
    H = np.zeros((maximum_dimension + 1, maximum_dimension), dtype=np.complex128)
    projected = np.zeros(
        (rows.shape[0], maximum_dimension + 1), dtype=np.complex128
    )
    action_seconds = np.zeros(maximum_dimension, dtype=np.float64)
    orthogonalization_seconds = np.zeros(maximum_dimension, dtype=np.float64)
    orthogonalization_profile = {
        name: np.zeros(maximum_dimension, dtype=np.float64)
        for name in ORTHOGONALIZATION_PROFILE_FIELDS
    }
    check_m: list[int] = []
    check_r: list[float] = []
    elapsed = 0.0
    breakdown = False

    if state_path.exists():
        if not basis_path.exists():
            raise ValueError("checkpoint basis is missing")
        state = np.load(state_path, allow_pickle=False)
        if str(state["binding"].item()) != key:
            raise ValueError("checkpoint inputs do not match the requested problem")
        actual = int(state["dimension"].item())
        H[: actual + 1, :actual] = state["hessenberg"]
        projected[:, : actual + 1] = state["sink_projections"]
        action_seconds[:actual] = state["action_wall_seconds"]
        orthogonalization_seconds[:actual] = state[
            "orthogonalization_wall_seconds"
        ]
        saved_backend = (
            str(state["orthogonalization_backend"].item())
            if "orthogonalization_backend" in state
            else CPU_ORTHOGONALIZATION_BACKEND
        )
        if saved_backend != orthogonalization_backend:
            raise ValueError("checkpoint orthogonalization backend does not match")
        for name, values in orthogonalization_profile.items():
            saved_name = f"orthogonalization_profile_{name}"
            if saved_name in state:
                values[:actual] = state[saved_name]
        check_m = [int(value) for value in state["check_dimensions"]]
        check_r = [float(value) for value in state["check_residuals"]]
        elapsed = float(state["construction_wall_seconds"].item())
        breakdown = bool(state["breakdown"].item())
        V = np.lib.format.open_memmap(basis_path, mode="r+")
        if V.shape != (n, maximum_dimension + 1) or V.dtype != np.complex128:
            raise ValueError("checkpoint basis shape or dtype does not match")
    else:
        if basis_path.exists():
            raise ValueError("checkpoint state is missing")
        if maximum_new_steps == 0:
            raise ValueError("zero new steps requires an existing checkpoint")
        actual = 0
        V = np.lib.format.open_memmap(
            basis_path,
            mode="w+",
            dtype=np.complex128,
            shape=(n, maximum_dimension + 1),
            fortran_order=True,
        )
        V[:, 0] = b / beta
        projected[:, 0] = rows @ V[:, 0]
        V.flush()

    call_started = perf_counter()
    start_dimension = actual

    def save() -> None:
        V.flush()
        arrays: dict[str, object] = {
            f"orthogonalization_profile_{name}": values[:actual]
            for name, values in orthogonalization_profile.items()
        }
        arrays.update(
            {
                "binding": np.asarray(key),
                "dimension": np.asarray(actual, dtype=np.int64),
                "hessenberg": H[: actual + 1, :actual],
                "sink_projections": projected[:, : actual + 1],
                "action_wall_seconds": action_seconds[:actual],
                "orthogonalization_wall_seconds": (
                    orthogonalization_seconds[:actual]
                ),
                "check_dimensions": np.asarray(check_m, dtype=np.int64),
                "check_residuals": np.asarray(check_r, dtype=np.float64),
                "construction_wall_seconds": np.asarray(
                    elapsed + perf_counter() - call_started,
                    dtype=np.float64,
                ),
                "breakdown": np.asarray(breakdown),
                "orthogonalization_backend": np.asarray(
                    orthogonalization_backend
                ),
            }
        )
        _write_state(state_path, **arrays)

    while (
        actual < maximum_dimension
        and not breakdown
        and (
            maximum_new_steps is None
            or actual - start_dimension < maximum_new_steps
        )
    ):
        j = actual
        action_started = perf_counter()
        w = np.asarray(operator.matvec(V[:, j]), dtype=np.complex128)
        action_seconds[j] = perf_counter() - action_started
        if w.shape != (n,) or not np.all(np.isfinite(w)):
            raise ValueError("operator matvec returned an invalid vector")
        orthogonalization_started = perf_counter()
        if orthogonalizer is None:
            for _ in range(2):
                correction = scipy.linalg.blas.zgemv(
                    1.0, V[:, : j + 1], w, trans=2, overwrite_y=0
                )
                H[: j + 1, j] += correction
                w -= V[:, : j + 1] @ correction
        else:
            step = orthogonalizer.orthogonalize(V[:, : j + 1], w)
            correction = np.asarray(step.correction)
            remainder = np.asarray(step.remainder)
            if (
                correction.shape != (j + 1,)
                or correction.dtype != np.complex128
                or remainder.shape != (n,)
                or remainder.dtype != np.complex128
                or not np.all(np.isfinite(correction))
                or not np.all(np.isfinite(remainder))
            ):
                raise ValueError("orthogonalizer returned invalid Arnoldi data")
            H[: j + 1, j] += correction
            w = np.array(remainder, dtype=np.complex128, copy=True)
            profile = getattr(step, "profile", None)
            if not isinstance(profile, dict):
                raise ValueError("orthogonalizer returned invalid timing data")
            for name, values in orthogonalization_profile.items():
                if name not in profile:
                    raise ValueError(f"orthogonalizer profile misses {name}")
                value = float(profile[name])
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError("orthogonalizer profile must be nonnegative")
                values[j] = value
        H[j + 1, j] = np.linalg.norm(w)
        orthogonalization_seconds[j] = perf_counter() - orthogonalization_started
        actual = j + 1
        if abs(H[actual, j]) <= breakdown_threshold:
            breakdown = True
        else:
            V[:, actual] = w / H[actual, j]
            projected[:, actual] = rows @ V[:, actual]

        stop_for_limit = (
            maximum_new_steps is not None
            and actual - start_dimension >= maximum_new_steps
        )
        inspect = actual in checks or actual == maximum_dimension or breakdown
        if inspect or stop_for_limit:
            residual = float(
                np.max(_reduced_residuals(H[: actual + 1, :actual], beta, z))
            )
            if check_m and check_m[-1] == actual:
                check_r[-1] = residual
            else:
                check_m.append(actual)
                check_r.append(residual)
            save()
            if residual <= tolerance or stop_for_limit:
                break

    if not check_m or check_m[-1] != actual:
        residual = float(
            np.max(_reduced_residuals(H[: actual + 1, :actual], beta, z))
        )
        check_m.append(actual)
        check_r.append(residual)
        save()

    basis = V[:, : actual + 1]
    hessenberg = np.ascontiguousarray(H[: actual + 1, :actual])
    projection = ArnoldiResolvent(
        basis=basis,
        hessenberg=hessenberg,
        source_norm=np.float64(beta),
        action_wall_seconds=np.asarray(action_seconds[:actual]),
        orthogonality_defect=np.float64(np.nan),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        construction_wall_seconds=np.float64(
            elapsed + perf_counter() - call_started
        ),
        breakdown=breakdown,
        check_dimensions=tuple(check_m),
        check_maximum_relative_residuals=np.asarray(check_r, dtype=np.float64),
        residual_tolerance=np.float64(tolerance),
        converged=bool(check_r[-1] <= tolerance),
    )
    return CheckpointedArnoldiResult(
        projection=projection,
        sink_projections=np.asarray(projected[:, :actual]),
        orthogonalization_wall_seconds=np.asarray(
            orthogonalization_seconds[:actual]
        ),
        checkpoint_dimensions=checks,
        directory=root,
        orthogonalization_backend=orthogonalization_backend,
        orthogonalization_profile={
            name: np.asarray(values[:actual])
            for name, values in orthogonalization_profile.items()
        },
        orthogonalization_setup_wall_seconds=float(
            getattr(orthogonalizer, "setup_wall_seconds", 0.0)
        ),
    )


__all__ = [
    "CheckpointedArnoldiResult",
    "CPU_ORTHOGONALIZATION_BACKEND",
    "ORTHOGONALIZATION_PROFILE_FIELDS",
    "checkpointed_arnoldi_resolvent",
    "eom_checkpoint_identity",
    "require_resources",
]

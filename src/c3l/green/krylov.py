"""One-sided Arnoldi resolvents for many frequencies and one physical source."""

from __future__ import annotations

from dataclasses import dataclass
import resource
from time import perf_counter
from typing import Protocol

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from numpy.typing import ArrayLike, NDArray


ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
RealVector = NDArray[np.float64]


class MatrixFreeAction(Protocol):
    @property
    def dimension(self) -> int: ...

    def matvec(self, vector: ArrayLike) -> ComplexVector: ...


@dataclass(slots=True)
class ArnoldiResolvent:
    r"""Projection satisfying $K V_m=V_m H_m+h v_{m+1}e_m^T$."""

    basis: ComplexMatrix
    hessenberg: ComplexMatrix
    source_norm: np.float64
    action_wall_seconds: RealVector
    orthogonality_defect: np.float64
    peak_rss_kib: int
    construction_wall_seconds: np.float64
    breakdown: bool
    check_dimensions: tuple[int, ...]
    check_maximum_relative_residuals: RealVector
    residual_tolerance: np.float64
    converged: bool

    @property
    def dimension(self) -> int:
        return int(self.hessenberg.shape[1])

    @property
    def vector_dimension(self) -> int:
        return int(self.basis.shape[0])

    @property
    def terminal_coupling(self) -> complex:
        m = self.dimension
        return complex(self.hessenberg[m, m - 1])

    def _dimension(self, dimension: int | None) -> int:
        m = self.dimension if dimension is None else dimension
        if isinstance(m, bool) or not isinstance(m, int):
            raise TypeError("dimension must be an integer")
        if not 1 <= m <= self.dimension:
            raise ValueError("dimension must lie in [1, krylov_dimension]")
        return m

    def coefficients(
        self,
        shifts: ArrayLike,
        *,
        dimension: int | None = None,
    ) -> ComplexMatrix:
        """Return reduced coordinates, optionally at a saved Arnoldi prefix."""

        z = np.asarray(shifts, dtype=np.complex128)
        if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
            raise ValueError("shifts must be a finite nonempty vector")
        m = self._dimension(dimension)
        H = self.hessenberg[:m, :m]
        rhs = np.zeros(m, dtype=np.complex128)
        rhs[0] = self.source_norm
        T, Q = scipy.linalg.schur(H, output="complex", check_finite=False)
        reduced_rhs = Q.conj().T @ rhs
        identity = np.eye(m, dtype=np.complex128)
        reduced = np.empty((z.size, m), dtype=np.complex128)
        for p, shift in enumerate(z):
            coordinates = scipy.linalg.solve_triangular(
                shift * identity - T,
                reduced_rhs,
                lower=False,
                check_finite=False,
            )
            reduced[p] = Q @ coordinates
        return reduced

    def residual_norms(
        self,
        shifts: ArrayLike,
        *,
        dimension: int | None = None,
    ) -> RealVector:
        """Exact Arnoldi-relation residual norms at one Arnoldi prefix."""

        m = self._dimension(dimension)
        reduced = self.coefficients(shifts, dimension=m)
        residual = np.abs(self.hessenberg[m, m - 1] * reduced[:, -1])
        return np.asarray(residual, dtype=np.float64)

    def relative_residuals(
        self,
        shifts: ArrayLike,
        *,
        dimension: int | None = None,
    ) -> RealVector:
        return np.asarray(
            self.residual_norms(shifts, dimension=dimension)
            / max(float(self.source_norm), np.finfo(np.float64).tiny),
            dtype=np.float64,
        )

    def solutions(
        self,
        shifts: ArrayLike,
        *,
        dimension: int | None = None,
    ) -> ComplexMatrix:
        """Reconstruct vectors at the full projection or one saved prefix."""

        m = self._dimension(dimension)
        reduced = self.coefficients(shifts, dimension=m)
        return np.asarray(
            reduced @ self.basis[:, :m].T, dtype=np.complex128
        )

    def time_coefficients(
        self,
        time: ArrayLike,
        *,
        dimension: int | None = None,
    ) -> ComplexMatrix:
        r"""Return $\exp(-iH_m t)\,\|b\|e_1$ on a finite time grid."""

        t = np.asarray(time, dtype=np.float64)
        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
            raise ValueError("time must be a finite nonempty vector")
        m = self._dimension(dimension)
        H = self.hessenberg[:m, :m]
        rhs = np.zeros(m, dtype=np.complex128)
        rhs[0] = self.source_norm
        if t.size == 1:
            reduced = scipy.sparse.linalg.expm_multiply(
                -1j * float(t[0]) * H, rhs
            )[None, :]
        elif np.allclose(
            np.diff(t), t[1] - t[0], atol=1.0e-14, rtol=1.0e-14
        ):
            reduced = scipy.sparse.linalg.expm_multiply(
                -1j * H,
                rhs,
                start=float(t[0]),
                stop=float(t[-1]),
                num=t.size,
                endpoint=True,
            )
        else:
            reduced = np.asarray(
                [
                    scipy.sparse.linalg.expm_multiply(-1j * float(value) * H, rhs)
                    for value in t
                ],
                dtype=np.complex128,
            )
        return np.asarray(reduced, dtype=np.complex128)

    def time_residual_norms(
        self,
        time: ArrayLike,
        *,
        dimension: int | None = None,
    ) -> RealVector:
        r"""Return the exact Arnoldi differential-residual norm at each time.

        If $y_m(t)=V_m\exp(-iH_m t)\|b\|e_1$ and
        $A V_m=V_mH_m+h_{m+1,m}v_{m+1}e_m^T$, then

        $$
        \|\dot y_m+iAy_m\|_2
        =|h_{m+1,m}e_m^T\exp(-iH_m t)\|b\|e_1|.
        $$

        This diagnoses the time propagator itself; it is distinct from the
        shifted-resolvent residual used to stop construction of the basis.
        """

        m = self._dimension(dimension)
        reduced = self.time_coefficients(time, dimension=m)
        return np.asarray(
            np.abs(self.hessenberg[m, m - 1] * reduced[:, -1]),
            dtype=np.float64,
        )

    def relative_time_residuals(
        self,
        time: ArrayLike,
        *,
        dimension: int | None = None,
    ) -> RealVector:
        """Normalize :meth:`time_residual_norms` by the source norm."""

        return np.asarray(
            self.time_residual_norms(time, dimension=dimension)
            / max(float(self.source_norm), np.finfo(np.float64).tiny),
            dtype=np.float64,
        )

    def transfer(
        self,
        shifts: ArrayLike,
        sink: ArrayLike,
        *,
        dimension: int | None = None,
    ) -> ComplexVector:
        """Evaluate a sink row at the full projection or one saved prefix."""

        row = np.asarray(sink, dtype=np.complex128)
        if row.shape != (self.vector_dimension,) or not np.all(np.isfinite(row)):
            raise ValueError("sink must match the full vector dimension")
        m = self._dimension(dimension)
        reduced = self.coefficients(shifts, dimension=m)
        projected_sink = row @ self.basis[:, :m]
        return np.asarray(reduced @ projected_sink, dtype=np.complex128)

    def to_dict(self) -> dict[str, object]:
        return {
            "krylov_dimension": self.dimension,
            "full_vector_dimension": self.vector_dimension,
            "source_norm": float(self.source_norm),
            "terminal_coupling_absolute": float(abs(self.terminal_coupling)),
            "orthogonality_defect": float(self.orthogonality_defect),
            "action_wall_seconds": [float(value) for value in self.action_wall_seconds],
            "action_wall_seconds_total": float(np.sum(self.action_wall_seconds)),
            "construction_wall_seconds": float(self.construction_wall_seconds),
            "peak_rss_kib": self.peak_rss_kib,
            "breakdown": self.breakdown,
            "check_dimensions": list(self.check_dimensions),
            "check_maximum_relative_residuals": [
                float(value) for value in self.check_maximum_relative_residuals
            ],
            "maximum_relative_residual": float(
                self.check_maximum_relative_residuals[-1]
            ),
            "residual_tolerance": float(self.residual_tolerance),
            "converged": self.converged,
            "method": "two_pass_one_sided_Arnoldi_shift_invariant_resolvent",
        }


def _reduced_residuals(
    H: ComplexMatrix,
    source_norm: float,
    shifts: ComplexVector,
) -> RealVector:
    m = H.shape[1]
    rhs = np.zeros(m, dtype=np.complex128)
    rhs[0] = source_norm
    T, Q = scipy.linalg.schur(H[:m, :m], output="complex", check_finite=False)
    reduced_rhs = Q.conj().T @ rhs
    identity = np.eye(m, dtype=np.complex128)
    residual = np.empty(shifts.size, dtype=np.float64)
    terminal = H[m, m - 1]
    for p, shift in enumerate(shifts):
        coordinates = scipy.linalg.solve_triangular(
            shift * identity - T,
            reduced_rhs,
            lower=False,
            check_finite=False,
        )
        y = Q @ coordinates
        residual[p] = abs(terminal * y[-1]) / max(
            source_norm, np.finfo(np.float64).tiny
        )
    return residual


def arnoldi_resolvent(
    operator: MatrixFreeAction,
    rhs: ArrayLike,
    shifts: ArrayLike,
    *,
    maximum_dimension: int,
    residual_tolerance: float,
    minimum_dimension: int = 16,
    check_interval: int = 8,
    breakdown_tolerance: float = 1.0e-14,
) -> ArnoldiResolvent:
    r"""Build one shift-invariant Krylov space for $(zI-K)^{-1}b$.

    Two-pass classical Gram--Schmidt keeps the implementation compact and
    makes each orthogonalization pass a pair of dense BLAS operations.  The
    stopping residual follows directly from the Arnoldi relation and is exact
    for every declared shift up to floating-point roundoff.
    """

    if not hasattr(operator, "dimension") or not hasattr(operator, "matvec"):
        raise TypeError("operator must expose dimension and matvec")
    n = int(operator.dimension)
    b = np.asarray(rhs, dtype=np.complex128)
    z = np.asarray(shifts, dtype=np.complex128)
    if b.shape != (n,) or not np.all(np.isfinite(b)):
        raise ValueError("rhs must be a finite vector matching the operator")
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("shifts must be a finite nonempty vector")
    if (
        isinstance(maximum_dimension, bool)
        or not isinstance(maximum_dimension, int)
        or maximum_dimension <= 0
        or maximum_dimension > n
    ):
        raise ValueError("maximum_dimension must lie in [1, operator.dimension]")
    if (
        isinstance(minimum_dimension, bool)
        or not isinstance(minimum_dimension, int)
        or not 1 <= minimum_dimension <= maximum_dimension
    ):
        raise ValueError("minimum_dimension must lie in [1, maximum_dimension]")
    if (
        isinstance(check_interval, bool)
        or not isinstance(check_interval, int)
        or check_interval <= 0
    ):
        raise ValueError("check_interval must be a positive integer")
    tolerance = float(residual_tolerance)
    breakdown_threshold = float(breakdown_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if not np.isfinite(breakdown_threshold) or breakdown_threshold <= 0.0:
        raise ValueError("breakdown_tolerance must be finite and positive")
    beta = float(np.linalg.norm(b))
    if beta == 0.0:
        raise ValueError("rhs must be nonzero")

    started = perf_counter()
    # Column-major storage makes each action vector contiguous and lets BLAS
    # evaluate V^dagger w without materializing a conjugated Krylov basis.
    V = np.zeros((n, maximum_dimension + 1), dtype=np.complex128, order="F")
    H = np.zeros(
        (maximum_dimension + 1, maximum_dimension), dtype=np.complex128
    )
    V[:, 0] = b / beta
    action_seconds: list[float] = []
    check_dimensions: list[int] = []
    check_residuals: list[float] = []
    actual = maximum_dimension
    breakdown = False
    for j in range(maximum_dimension):
        action_started = perf_counter()
        w = np.asarray(operator.matvec(V[:, j]), dtype=np.complex128)
        action_seconds.append(perf_counter() - action_started)
        if w.shape != (n,) or not np.all(np.isfinite(w)):
            raise ValueError("operator matvec returned an invalid vector")
        # Projection coefficients use the Euclidean conjugate inner product.
        for _ in range(2):
            correction = scipy.linalg.blas.zgemv(
                1.0,
                V[:, : j + 1],
                w,
                trans=2,
                overwrite_y=0,
            )
            H[: j + 1, j] += correction
            w -= V[:, : j + 1] @ correction
        H[j + 1, j] = np.linalg.norm(w)
        actual = j + 1
        if abs(H[j + 1, j]) <= breakdown_threshold:
            breakdown = True
            break
        V[:, j + 1] = w / H[j + 1, j]
        if actual >= minimum_dimension and (
            actual % check_interval == 0 or actual == maximum_dimension
        ):
            residual = _reduced_residuals(H[: actual + 1, :actual], beta, z)
            check_dimensions.append(actual)
            check_residuals.append(float(np.max(residual)))
            if float(np.max(residual)) <= tolerance:
                break

    if not check_dimensions or check_dimensions[-1] != actual:
        residual = _reduced_residuals(H[: actual + 1, :actual], beta, z)
        check_dimensions.append(actual)
        check_residuals.append(float(np.max(residual)))

    basis = np.asfortranarray(V[:, : actual + 1])
    hessenberg = np.ascontiguousarray(H[: actual + 1, :actual])
    gram = scipy.linalg.blas.zgemm(
        1.0,
        basis[:, :actual],
        basis[:, :actual],
        trans_a=2,
    )
    orthogonality = np.linalg.norm(
        gram - np.eye(actual, dtype=np.complex128), ord="fro"
    ) / max(float(actual), 1.0)
    return ArnoldiResolvent(
        basis=basis,
        hessenberg=hessenberg,
        source_norm=np.float64(beta),
        action_wall_seconds=np.asarray(action_seconds, dtype=np.float64),
        orthogonality_defect=np.float64(orthogonality),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        construction_wall_seconds=np.float64(perf_counter() - started),
        breakdown=breakdown,
        check_dimensions=tuple(check_dimensions),
        check_maximum_relative_residuals=np.asarray(
            check_residuals, dtype=np.float64
        ),
        residual_tolerance=np.float64(tolerance),
        converged=bool(check_residuals[-1] <= tolerance),
    )


__all__ = ["ArnoldiResolvent", "MatrixFreeAction", "arnoldi_resolvent"]

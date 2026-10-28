"""Auxiliary self-energy and physical Dyson reconstruction for C3L spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from c3l.models import FermionicLindbladModel


ComplexVector = NDArray[np.complex128]
ComplexMatrixGrid = NDArray[np.complex128]
RealVector = NDArray[np.float64]
BoolVector = NDArray[np.bool_]


def _grid(value: ArrayLike, name: str) -> RealVector:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite nonempty one-dimensional array")
    return array


def _complex_vector(value: ArrayLike, shape: tuple[int, ...], name: str) -> ComplexVector:
    array = np.asarray(value, dtype=np.complex128)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite complex vector on omega")
    return array


def _complex_matrix_grid(value: ArrayLike, name: str) -> ComplexMatrixGrid:
    array = np.asarray(value, dtype=np.complex128)
    if (
        array.ndim != 3
        or array.shape[0] == 0
        or array.shape[1] == 0
        or array.shape[1] != array.shape[2]
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must have finite shape (n_omega,n,n)")
    return array


@dataclass(frozen=True, slots=True)
class MatrixResolventDiagnostics:
    """Condition numbers and two-sided backward residuals for ``A G = I``."""

    condition_number: RealVector
    right_backward_residual: RealVector
    left_backward_residual: RealVector
    conditioning_limited: BoolVector
    conditioning_limit: float


@dataclass(frozen=True, slots=True)
class MatrixSpectralDiagnostics:
    """Hermitian matrix spectrum and positivity diagnostics at every frequency."""

    spectral: ComplexMatrixGrid
    hermiticity_defect: RealVector
    minimum_eigenvalue: RealVector
    maximum_eigenvalue: RealVector
    negative_ratio: RealVector


@dataclass(frozen=True, slots=True)
class MatrixSelfEnergyResult:
    """Matrix retarded self-energy with Green-zero conditioning metadata."""

    retarded: ComplexMatrixGrid
    G0_condition_number: RealVector
    G_condition_number: RealVector
    conditioning_limited: BoolVector
    dyson: MatrixResolventDiagnostics


@dataclass(frozen=True, slots=True)
class MatrixDysonResult:
    """Matrix retarded Green function from a supplied Dyson denominator."""

    inverse_retarded: ComplexMatrixGrid
    retarded: ComplexMatrixGrid
    resolvent: MatrixResolventDiagnostics
    spectral: MatrixSpectralDiagnostics


@dataclass(frozen=True, slots=True)
class MatrixOrderedReconstruction:
    """PM-to-continuous-bath reconstruction in both resolvent orderings."""

    retarded: ComplexMatrixGrid
    retarded_from_right_order: ComplexMatrixGrid
    left_ordered_residual: RealVector
    right_ordered_residual: RealVector
    left_resolvent_identity_residual: RealVector
    right_resolvent_identity_residual: RealVector
    ordering_difference: RealVector
    left_operator_condition_number: RealVector
    right_operator_condition_number: RealVector
    conditioning_limited: BoolVector
    spectral: MatrixSpectralDiagnostics


def matrix_resolvent_diagnostics(
    inverse_green: ArrayLike,
    green: ArrayLike,
    *,
    conditioning_limit: float = 1.0e10,
) -> MatrixResolventDiagnostics:
    r"""Evaluate the frozen left/right backward residuals for $A G=I$."""

    A = _complex_matrix_grid(inverse_green, "inverse_green")
    G = _complex_matrix_grid(green, "green")
    if G.shape != A.shape:
        raise ValueError("inverse_green and green must have the same shape")
    limit = float(conditioning_limit)
    if not np.isfinite(limit) or limit <= 1.0:
        raise ValueError("conditioning_limit must be finite and larger than one")
    n = A.shape[1]
    identity = np.eye(n, dtype=np.complex128)
    identity_norm = float(np.linalg.norm(identity, ord="fro"))
    condition = np.empty(A.shape[0], dtype=np.float64)
    right = np.empty_like(condition)
    left = np.empty_like(condition)
    for k, (a, g) in enumerate(zip(A, G, strict=True)):
        condition[k] = float(np.linalg.cond(a, p=2))
        denominator = (
            float(np.linalg.norm(a, ord=2))
            * float(np.linalg.norm(g, ord="fro"))
            + identity_norm
        )
        right[k] = float(np.linalg.norm(a @ g - identity, ord="fro") / denominator)
        left[k] = float(np.linalg.norm(g @ a - identity, ord="fro") / denominator)
    return MatrixResolventDiagnostics(
        condition_number=condition,
        right_backward_residual=right,
        left_backward_residual=left,
        conditioning_limited=np.asarray(condition > limit, dtype=np.bool_),
        conditioning_limit=limit,
    )


def matrix_spectral_diagnostics(G_R: ArrayLike) -> MatrixSpectralDiagnostics:
    r"""Return $A=i(G^R-G^{R\dagger})/(2\pi)$ and matrix positivity defects."""

    G = _complex_matrix_grid(G_R, "G_R")
    adjoint = G.conj().swapaxes(1, 2)
    spectral = 1j * (G - adjoint) / (2.0 * np.pi)
    hermiticity = np.empty(G.shape[0], dtype=np.float64)
    minimum = np.empty_like(hermiticity)
    maximum = np.empty_like(hermiticity)
    negative_ratio = np.empty_like(hermiticity)
    for k, value in enumerate(spectral):
        scale = max(float(np.linalg.norm(value, ord="fro")), np.finfo(float).tiny)
        hermiticity[k] = float(
            np.linalg.norm(value - value.conj().T, ord="fro") / scale
        )
        hermitian = (value + value.conj().T) / 2.0
        eigenvalues = np.linalg.eigvalsh(hermitian)
        minimum[k] = float(eigenvalues[0])
        maximum[k] = float(eigenvalues[-1])
        negative_ratio[k] = max(0.0, -minimum[k]) / max(maximum[k], 1.0e-14)
    return MatrixSpectralDiagnostics(
        spectral=np.asarray(spectral, dtype=np.complex128),
        hermiticity_defect=hermiticity,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        negative_ratio=negative_ratio,
    )


def matrix_auxiliary_self_energy(
    G0_R: ArrayLike,
    G_R: ArrayLike,
    *,
    conditioning_limit: float = 1.0e10,
) -> MatrixSelfEnergyResult:
    r"""Evaluate $\Sigma^R=G_0^{-1}-G^{-1}$ by ordered linear solves."""

    G0 = _complex_matrix_grid(G0_R, "G0_R")
    G = _complex_matrix_grid(G_R, "G_R")
    if G.shape != G0.shape:
        raise ValueError("G0_R and G_R must have the same shape")
    identity = np.eye(G.shape[1], dtype=np.complex128)
    inverse0 = np.empty_like(G0)
    inverse = np.empty_like(G)
    condition0 = np.empty(G.shape[0], dtype=np.float64)
    condition = np.empty_like(condition0)
    for k, (g0, g) in enumerate(zip(G0, G, strict=True)):
        condition0[k] = float(np.linalg.cond(g0, p=2))
        condition[k] = float(np.linalg.cond(g, p=2))
        inverse0[k] = np.linalg.solve(g0, identity)
        inverse[k] = np.linalg.solve(g, identity)
    sigma = inverse0 - inverse
    dyson_inverse = inverse0 - sigma
    diagnostics = matrix_resolvent_diagnostics(
        dyson_inverse,
        G,
        conditioning_limit=conditioning_limit,
    )
    return MatrixSelfEnergyResult(
        retarded=np.asarray(sigma, dtype=np.complex128),
        G0_condition_number=condition0,
        G_condition_number=condition,
        conditioning_limited=np.asarray(
            (condition0 > conditioning_limit) | (condition > conditioning_limit),
            dtype=np.bool_,
        ),
        dyson=diagnostics,
    )


def matrix_dyson_green(
    z: ArrayLike,
    h: ArrayLike,
    Delta_R: ArrayLike,
    Sigma_R: ArrayLike,
    *,
    conditioning_limit: float = 1.0e10,
) -> MatrixDysonResult:
    r"""Solve $G^R=[zI-h-\Delta^R-\Sigma^R]^{-1}$ on a frequency grid."""

    shifts = np.asarray(z, dtype=np.complex128)
    delta = _complex_matrix_grid(Delta_R, "Delta_R")
    sigma = _complex_matrix_grid(Sigma_R, "Sigma_R")
    one_body = np.asarray(h, dtype=np.complex128)
    n = delta.shape[1]
    if (
        shifts.shape != (delta.shape[0],)
        or not np.all(np.isfinite(shifts))
        or sigma.shape != delta.shape
        or one_body.shape != (n, n)
        or not np.all(np.isfinite(one_body))
    ):
        raise ValueError("z, h, Delta_R, and Sigma_R have incompatible shapes")
    identity = np.eye(n, dtype=np.complex128)
    inverse = (
        shifts[:, None, None] * identity[None, :, :]
        - one_body[None, :, :]
        - delta
        - sigma
    )
    green = np.empty_like(inverse)
    for k, value in enumerate(inverse):
        green[k] = np.linalg.solve(value, identity)
    return MatrixDysonResult(
        inverse_retarded=np.asarray(inverse, dtype=np.complex128),
        retarded=np.asarray(green, dtype=np.complex128),
        resolvent=matrix_resolvent_diagnostics(
            inverse,
            green,
            conditioning_limit=conditioning_limit,
        ),
        spectral=matrix_spectral_diagnostics(green),
    )


def matrix_ordered_dyson_reconstruction(
    G_PM_R: ArrayLike,
    Delta_PM_R: ArrayLike,
    Delta_physical_R: ArrayLike,
    *,
    conditioning_limit: float = 1.0e10,
) -> MatrixOrderedReconstruction:
    r"""Replace the PM bath by the physical bath using both ordered solves.

    With ``D = Delta_physical_R - Delta_PM_R``, the two equivalent equations
    are ``(I-G_PM D) G = G_PM`` and ``G (I-D G_PM) = G_PM``.  Both are solved
    and their two resolvent identities are reapplied without forming
    ``G_PM^{-1}``.
    """

    G_pm = _complex_matrix_grid(G_PM_R, "G_PM_R")
    delta_pm = _complex_matrix_grid(Delta_PM_R, "Delta_PM_R")
    delta_physical = _complex_matrix_grid(Delta_physical_R, "Delta_physical_R")
    if delta_pm.shape != G_pm.shape or delta_physical.shape != G_pm.shape:
        raise ValueError("all reconstruction arrays must have the same shape")
    n = G_pm.shape[1]
    identity = np.eye(n, dtype=np.complex128)
    D = delta_physical - delta_pm
    left_solution = np.empty_like(G_pm)
    right_solution = np.empty_like(G_pm)
    left_residual = np.empty(G_pm.shape[0], dtype=np.float64)
    right_residual = np.empty_like(left_residual)
    left_identity = np.empty_like(left_residual)
    right_identity = np.empty_like(left_residual)
    difference = np.empty_like(left_residual)
    left_condition = np.empty_like(left_residual)
    right_condition = np.empty_like(left_residual)
    for k, (g_pm, d) in enumerate(zip(G_pm, D, strict=True)):
        left_operator = identity - g_pm @ d
        right_operator = identity - d @ g_pm
        left_solution[k] = np.linalg.solve(left_operator, g_pm)
        right_solution[k] = np.linalg.solve(right_operator.T, g_pm.T).T
        left_condition[k] = float(np.linalg.cond(left_operator, p=2))
        right_condition[k] = float(np.linalg.cond(right_operator, p=2))
        left_denominator = (
            float(np.linalg.norm(left_operator, ord=2))
            * float(np.linalg.norm(left_solution[k], ord="fro"))
            + float(np.linalg.norm(g_pm, ord="fro"))
        )
        right_denominator = (
            float(np.linalg.norm(right_operator, ord=2))
            * float(np.linalg.norm(right_solution[k], ord="fro"))
            + float(np.linalg.norm(g_pm, ord="fro"))
        )
        left_residual[k] = float(
            np.linalg.norm(left_operator @ left_solution[k] - g_pm, ord="fro")
            / left_denominator
        )
        right_residual[k] = float(
            np.linalg.norm(right_solution[k] @ right_operator - g_pm, ord="fro")
            / right_denominator
        )
        change = left_solution[k] - g_pm
        left_product = g_pm @ d @ left_solution[k]
        right_product = left_solution[k] @ d @ g_pm
        left_identity[k] = float(
            np.linalg.norm(change - left_product, ord="fro")
            / max(
                float(np.linalg.norm(change, ord="fro"))
                + float(np.linalg.norm(left_product, ord="fro")),
                np.finfo(float).tiny,
            )
        )
        right_identity[k] = float(
            np.linalg.norm(change - right_product, ord="fro")
            / max(
                float(np.linalg.norm(change, ord="fro"))
                + float(np.linalg.norm(right_product, ord="fro")),
                np.finfo(float).tiny,
            )
        )
        difference[k] = float(
            np.linalg.norm(left_solution[k] - right_solution[k], ord="fro")
            / max(float(np.linalg.norm(left_solution[k], ord="fro")), np.finfo(float).tiny)
        )
    return MatrixOrderedReconstruction(
        retarded=left_solution,
        retarded_from_right_order=right_solution,
        left_ordered_residual=left_residual,
        right_ordered_residual=right_residual,
        left_resolvent_identity_residual=left_identity,
        right_resolvent_identity_residual=right_identity,
        ordering_difference=difference,
        left_operator_condition_number=left_condition,
        right_operator_condition_number=right_condition,
        conditioning_limited=np.asarray(
            (left_condition > conditioning_limit)
            | (right_condition > conditioning_limit),
            dtype=np.bool_,
        ),
        spectral=matrix_spectral_diagnostics(left_solution),
    )


def noninteracting_auxiliary_green(
    model: FermionicLindbladModel,
    omega: ArrayLike,
    *,
    eta: float = 0.0,
) -> tuple[ComplexVector, ComplexVector]:
    r"""Return impurity $G_{0,\mathrm{aux}}^R$ and $G_{0,\mathrm{aux}}^K$.

    The same nonnegative numerical broadening used in the interacting
    resolvent may be supplied.  Using it in both Green functions makes the
    noninteracting self-energy regression exactly zero at finite broadening.
    """

    if not isinstance(model, FermionicLindbladModel):
        raise TypeError("model must be a FermionicLindbladModel")
    w = _grid(omega, "omega")
    broadening = float(eta)
    if not np.isfinite(broadening) or broadening < 0.0:
        raise ValueError("eta must be finite and nonnegative")
    n = model.n_sites
    identity = np.eye(n, dtype=np.complex128)
    Lambda = model.gamma_loss + model.gamma_gain
    Omega = model.gamma_gain - model.gamma_loss
    inverse = (
        (w + 1j * broadening)[:, None, None] * identity[None, :, :]
        - model.E[None, :, :]
        + 1j * Lambda[None, :, :]
    )
    GR = np.linalg.inv(inverse)
    GA = GR.conj().swapaxes(1, 2)
    GK = 2j * GR @ Omega[None, :, :] @ GA
    f = model.impurity_site
    return (
        np.asarray(GR[:, f, f], dtype=np.complex128),
        np.asarray(GK[:, f, f], dtype=np.complex128),
    )


def auxiliary_self_energy(
    G0_R: ArrayLike,
    G0_K: ArrayLike,
    G_R: ArrayLike,
    G_K: ArrayLike,
) -> tuple[ComplexVector, ComplexVector]:
    r"""Evaluate the frozen scalar $\Sigma_{\mathrm{aux}}^{R,K}$ equations."""

    g0r = np.asarray(G0_R, dtype=np.complex128)
    shape = g0r.shape
    if g0r.ndim != 1 or g0r.size == 0:
        raise ValueError("G0_R must be a nonempty vector")
    g0k = _complex_vector(G0_K, shape, "G0_K")
    gr = _complex_vector(G_R, shape, "G_R")
    gk = _complex_vector(G_K, shape, "G_K")
    if not np.all(np.isfinite(g0r)):
        raise ValueError("G0_R must be finite")
    floor = np.finfo(np.float64).tiny
    if np.any(np.abs(g0r) <= floor) or np.any(np.abs(gr) <= floor):
        raise ValueError("Green function has a zero that makes Sigma undefined")
    Sigma_R = 1.0 / g0r - 1.0 / gr
    Sigma_K = -g0k / np.abs(g0r) ** 2 + gk / np.abs(gr) ** 2
    return (
        np.asarray(Sigma_R, dtype=np.complex128),
        np.asarray(Sigma_K, dtype=np.complex128),
    )


@dataclass(frozen=True, slots=True)
class PhysicalDysonResult:
    """Scalar physical impurity Green function reconstructed from auxiliary Sigma."""

    omega: RealVector
    retarded: ComplexVector
    keldysh: ComplexVector
    spectral: RealVector

    def __post_init__(self) -> None:
        w = _grid(self.omega, "omega")
        shape = w.shape
        retarded = _complex_vector(self.retarded, shape, "retarded")
        keldysh = _complex_vector(self.keldysh, shape, "keldysh")
        spectral = np.asarray(self.spectral, dtype=np.float64)
        if spectral.shape != shape or not np.all(np.isfinite(spectral)):
            raise ValueError("spectral must be a finite real vector on omega")
        object.__setattr__(self, "omega", w)
        object.__setattr__(self, "retarded", retarded)
        object.__setattr__(self, "keldysh", keldysh)
        object.__setattr__(self, "spectral", spectral)


def physical_dyson_reconstruction(
    omega: ArrayLike,
    epsilon_imp: float,
    Delta_R: ArrayLike,
    Delta_K: ArrayLike,
    Sigma_R: ArrayLike,
    Sigma_K: ArrayLike,
) -> PhysicalDysonResult:
    r"""Insert $\Sigma_{\mathrm{aux}}$ into the physical target bath."""

    w = _grid(omega, "omega")
    shape = w.shape
    epsilon = float(epsilon_imp)
    if not np.isfinite(epsilon):
        raise ValueError("epsilon_imp must be finite")
    delta_r = _complex_vector(Delta_R, shape, "Delta_R")
    delta_k = _complex_vector(Delta_K, shape, "Delta_K")
    sigma_r = _complex_vector(Sigma_R, shape, "Sigma_R")
    sigma_k = _complex_vector(Sigma_K, shape, "Sigma_K")
    denominator = w - epsilon - delta_r - sigma_r
    if np.any(np.abs(denominator) <= np.finfo(np.float64).tiny):
        raise ValueError("physical Dyson denominator vanishes on the grid")
    retarded = 1.0 / denominator
    keldysh = np.abs(retarded) ** 2 * (delta_k + sigma_k)
    spectral = -retarded.imag / np.pi
    return PhysicalDysonResult(w, retarded, keldysh, spectral)


def equilibrium_fdt_defect(
    omega: ArrayLike,
    G_R: ArrayLike,
    G_K: ArrayLike,
    temperature: float,
) -> float:
    r"""Relative equilibrium defect of $G^K=(G^R-G^A)\tanh(\omega/2T)$."""

    w = _grid(omega, "omega")
    gr = _complex_vector(G_R, w.shape, "G_R")
    gk = _complex_vector(G_K, w.shape, "G_K")
    T = float(temperature)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("temperature must be finite and positive")
    expected = (gr - gr.conj()) * np.tanh(w / (2.0 * T))
    scale = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(gk - expected) / scale)


def particle_hole_spectral_defect(omega: ArrayLike, spectral: ArrayLike) -> float:
    r"""Maximum normalized $A(\omega)-A(-\omega)$ defect on a symmetric grid."""

    w = _grid(omega, "omega")
    A = np.asarray(spectral, dtype=np.float64)
    if A.shape != w.shape or not np.all(np.isfinite(A)):
        raise ValueError("spectral must be a finite vector on omega")
    if not np.allclose(w, -w[::-1], atol=2.0e-14, rtol=2.0e-14):
        raise ValueError("particle-hole diagnostic requires a symmetric grid")
    scale = max(float(np.max(np.abs(A))), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(A - A[::-1])) / scale)


def spectral_grid_diagnostics(
    omega: ArrayLike,
    G_R: ArrayLike,
) -> dict[str, float]:
    """Causality, finite-grid sum rule, and high-frequency moment diagnostics."""

    w = _grid(omega, "omega")
    gr = _complex_vector(G_R, w.shape, "G_R")
    spectral = -gr.imag / np.pi
    order = np.argsort(w)
    sorted_w = w[order]
    sorted_A = spectral[order]
    edge = max(1, min(4, w.size // 4))
    high_indices = np.argsort(np.abs(w))[-edge:]
    return {
        "minimum_spectral_weight": float(np.min(spectral)),
        "causality_violation": float(max(0.0, np.max(gr.imag))),
        "finite_grid_sum": float(np.trapezoid(sorted_A, x=sorted_w)),
        "finite_grid_sum_rule_defect": float(
            abs(np.trapezoid(sorted_A, x=sorted_w) - 1.0)
        ),
        "high_frequency_moment_defect": float(
            np.max(np.abs(w[high_indices] * gr[high_indices] - 1.0))
        ),
    }


__all__ = [
    "MatrixDysonResult",
    "MatrixOrderedReconstruction",
    "MatrixResolventDiagnostics",
    "MatrixSelfEnergyResult",
    "MatrixSpectralDiagnostics",
    "PhysicalDysonResult",
    "auxiliary_self_energy",
    "equilibrium_fdt_defect",
    "matrix_auxiliary_self_energy",
    "matrix_dyson_green",
    "matrix_ordered_dyson_reconstruction",
    "matrix_resolvent_diagnostics",
    "matrix_spectral_diagnostics",
    "noninteracting_auxiliary_green",
    "particle_hole_spectral_defect",
    "physical_dyson_reconstruction",
    "spectral_grid_diagnostics",
]

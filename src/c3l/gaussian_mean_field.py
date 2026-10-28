"""Number-conserving Gaussian mean field for sparse two-body models.

The one-particle density matrix convention is

``gamma[p, q] = <c_q^dagger c_p>``.

Only the molecular spin orbitals interact.  Pseudomode orbitals enter the
quadratic Hamiltonian, damping matrices, covariance, and response without an
interaction vertex.  No anomalous density or pairing field is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .interactions import SparseTwoBodyInteraction
from .quadratic_ness import finite_pm_covariance

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]


def _square(value: ArrayLike, name: str) -> ComplexArray:
    array = np.asarray(value, dtype=np.complex128)
    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite square matrix")
    return array


def _interaction(value: ArrayLike) -> RealArray:
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.shape[0] != raw.shape[1]
        or not np.all(np.isfinite(raw))
        or (np.iscomplexobj(raw) and np.any(raw.imag != 0.0))
    ):
        raise ValueError("U must be a finite real square matrix")
    U = np.asarray(raw.real, dtype=np.float64)
    if np.linalg.norm(U - U.T, ord="fro") > 1.0e-12 * max(
        np.linalg.norm(U, ord="fro"), 1.0
    ):
        raise ValueError("U must be symmetric")
    return U


def _require_hermitian(value: ComplexArray, name: str) -> None:
    defect = np.linalg.norm(value - value.conj().T, ord="fro")
    scale = max(np.linalg.norm(value, ord="fro"), 1.0)
    if defect > 1.0e-12 * scale:
        raise ValueError(f"{name} must be Hermitian")


def _spin_restriction_defect(value: ComplexArray) -> float:
    """Return the relative defect from site-major restricted-spin form."""

    if value.shape[0] % 2:
        return float("inf")
    up = value[0::2, 0::2]
    down = value[1::2, 1::2]
    mixing = np.concatenate(
        (value[0::2, 1::2].ravel(), value[1::2, 0::2].ravel())
    )
    numerator = np.sqrt(
        np.linalg.norm(up - down, ord="fro") ** 2
        + np.linalg.norm(mixing) ** 2
    )
    return float(numerator / max(np.linalg.norm(value, ord="fro"), 1.0))


def restrict_spin_density(gamma: ArrayLike) -> ComplexArray:
    r"""Project a covariance onto the number-conserving restricted-spin form.

    Orbitals are ordered site-major and spin-minor,
    ``(0 up, 0 down, 1 up, 1 down, ...)``.  The projection is

    .. math::

       \gamma_{\uparrow\uparrow}
       =\gamma_{\downarrow\downarrow}
       =\tfrac12(\gamma_{\uparrow\uparrow}
                 +\gamma_{\downarrow\downarrow}),\qquad
       \gamma_{\uparrow\downarrow}
       =\gamma_{\downarrow\uparrow}=0.

    The common spatial block is not diagonalized, clipped, or made
    idempotent.
    """

    density = _square(gamma, "gamma")
    if density.shape[0] % 2:
        raise ValueError("restricted spin requires an even orbital count")
    spatial = 0.5 * (
        density[0::2, 0::2] + density[1::2, 1::2]
    )
    result = np.zeros_like(density)
    result[0::2, 0::2] = spatial
    result[1::2, 1::2] = spatial
    return result


def _require_spin_restricted(value: ComplexArray, name: str) -> None:
    defect = _spin_restriction_defect(value)
    if defect > 1.0e-12:
        raise ValueError(
            f"{name} must have identical up/down blocks and zero spin mixing"
        )


def _orbitals(
    value: ArrayLike | None, n_sites: int, n_orbitals: int
) -> IndexArray:
    if value is None:
        result = np.arange(2 * n_sites, dtype=np.int64).reshape(n_sites, 2)
    else:
        raw = np.asarray(value)
        if raw.shape != (n_sites, 2) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("impurity_orbitals must have integer shape (n_sites, 2)")
        result = np.asarray(raw, dtype=np.int64)
    if (
        np.any(result < 0)
        or np.any(result >= n_orbitals)
        or np.unique(result).size != result.size
    ):
        raise ValueError("impurity_orbitals must be distinct valid orbital indices")
    return result


def density_vertex(
    U: ArrayLike,
    n_orbitals: int,
    impurity_orbitals: ArrayLike | None = None,
) -> ComplexArray:
    """Return ``V[p,q]`` for ``1/2 sum_(p!=q) V[p,q] n_p n_q``.

    ``U[i,j]`` couples the spin-summed densities on molecular sites ``i,j``.
    Consequently opposite spins on the same site couple through ``U[i,i]``;
    an orbital never interacts with itself, so ``V[p,p]=0``.
    """

    interaction = _interaction(U)
    if isinstance(n_orbitals, bool) or not isinstance(n_orbitals, int):
        raise TypeError("n_orbitals must be an integer")
    if n_orbitals < 2 * interaction.shape[0]:
        raise ValueError("n_orbitals is too small for the molecular spin orbitals")
    orbitals = _orbitals(
        impurity_orbitals, interaction.shape[0], n_orbitals
    )
    V = np.zeros((n_orbitals, n_orbitals), dtype=np.complex128)
    for i in range(interaction.shape[0]):
        for spin in (0, 1):
            p = int(orbitals[i, spin])
            for j in range(interaction.shape[0]):
                for other_spin in (0, 1):
                    q = int(orbitals[j, other_spin])
                    if p != q:
                        V[p, q] = interaction[i, j]
    return V


def centered_one_body(
    h_bare: ArrayLike,
    U: ArrayLike,
    impurity_orbitals: ArrayLike | None = None,
) -> ComplexArray:
    """Expand the charge-centered PPP interaction's one-body term.

    For each molecular site,

    ``v_center[i] = -U[i,i]/2 - sum_(j!=i) U[i,j]``.
    """

    h = _square(h_bare, "h_bare")
    interaction = _interaction(U)
    orbitals = _orbitals(
        impurity_orbitals, interaction.shape[0], h.shape[0]
    )
    result = np.array(h, copy=True)
    v_center = -0.5 * np.diag(interaction) - (
        np.sum(interaction, axis=1) - np.diag(interaction)
    )
    for i in range(interaction.shape[0]):
        for spin in (0, 1):
            p = int(orbitals[i, spin])
            result[p, p] += v_center[i]
    return result


def wick_fock(
    h_solver: ArrayLike,
    V: ArrayLike | SparseTwoBodyInteraction,
    gamma: ArrayLike,
) -> ComplexArray:
    r"""Return the number-conserving Wick/HF one-body matrix.

    For a sparse antisymmetrized interaction this evaluates directly

    .. math::

       F_{pq}=h_{pq}+\sum_{rs}\bar v_{prqs}\gamma_{sr}.

    The historical density representation uses ``V[p,p]=0`` and

    .. math::

       \bar v_{prqs}=\delta_{pq}\delta_{rs}V_{pr}
                    -\delta_{ps}\delta_{rq}V_{pr},

    the contraction

    .. math::

       F_{pq}=h_{pq}+\sum_{rs}\bar v_{prqs}\gamma_{sr}

    is evaluated as

    ``F = h + diag(V @ diag(gamma)) - V * gamma``.
    """

    h = _square(h_solver, "h_solver")
    density = _square(gamma, "gamma")
    if h.shape != density.shape:
        raise ValueError("h_solver and gamma must have identical shapes")
    if isinstance(V, SparseTwoBodyInteraction):
        if h.shape[0] != V.n_spin_orbitals:
            raise ValueError("the sparse interaction does not match h_solver")
        result = np.array(h, copy=True)
        for p, r, q, s, value in V.terms:
            result[p, q] += value * density[s, r]
            result[r, q] -= value * density[s, p]
            result[p, s] -= value * density[q, r]
            result[r, s] += value * density[q, p]
        return result

    vertex = _square(V, "V")
    if h.shape != vertex.shape:
        raise ValueError("h_solver, V, and gamma must have identical shapes")
    if np.max(abs(np.diag(vertex))) > 1.0e-14:
        raise ValueError("V must have a zero diagonal")
    return (
        h
        + np.diag(vertex @ np.diag(density))
        - vertex * density
    )


def charge_centered_fock(
    h_bare: ArrayLike,
    U: ArrayLike,
    gamma: ArrayLike,
    impurity_orbitals: ArrayLike | None = None,
) -> ComplexArray:
    r"""Return the GHF matrix for ``1/2 sum_ij U_ij (n_i-1)(n_j-1)``.

    Written directly, its molecular entries are

    .. math::

       F_{i\sigma,i\sigma}
       =h_{i\sigma,i\sigma}
        +U_{ii}(\gamma_{i\bar\sigma,i\bar\sigma}-1/2)
        +\sum_{j\ne i}U_{ij}(n_j-1),

    and, for distinct molecular spin orbitals,

    .. math::

       F_{i\sigma,j\sigma'}=h_{i\sigma,j\sigma'}
                            -U_{ij}\gamma_{i\sigma,j\sigma'}.

    This is exactly ``wick_fock(centered_one_body(...), V, gamma)``.
    """

    h = _square(h_bare, "h_bare")
    interaction = _interaction(U)
    density = _square(gamma, "gamma")
    if density.shape != h.shape:
        raise ValueError("h_bare and gamma must have identical shapes")
    orbitals = _orbitals(
        impurity_orbitals, interaction.shape[0], h.shape[0]
    )
    result = np.array(h, copy=True)
    site_density = np.sum(
        np.diag(density)[orbitals], axis=1
    )
    for i in range(interaction.shape[0]):
        for spin in (0, 1):
            p = int(orbitals[i, spin])
            other = int(orbitals[i, 1 - spin])
            result[p, p] += interaction[i, i] * (
                density[other, other] - 0.5
            )
            result[p, p] += np.sum(
                np.delete(interaction[i], i)
                * np.delete(site_density - 1.0, i)
            )
            for j in range(interaction.shape[0]):
                for other_spin in (0, 1):
                    q = int(orbitals[j, other_spin])
                    if q != p:
                        result[p, q] -= interaction[i, j] * density[p, q]
    return result


def charge_centered_wick_energy(
    h_bare: ArrayLike,
    U: ArrayLike,
    gamma: ArrayLike,
    impurity_orbitals: ArrayLike | None = None,
) -> complex:
    """Return the Gaussian expectation of the charge-centered Hamiltonian."""

    h = _square(h_bare, "h_bare")
    interaction = _interaction(U)
    density = _square(gamma, "gamma")
    if density.shape != h.shape:
        raise ValueError("h_bare and gamma must have identical shapes")
    orbitals = _orbitals(
        impurity_orbitals, interaction.shape[0], h.shape[0]
    )
    V = density_vertex(interaction, h.shape[0], orbitals)
    solver = centered_one_body(h, interaction, orbitals)
    direct = np.diag(density)
    pair = 0.5 * np.sum(
        V * (
            direct[:, None] * direct[None, :]
            - density * density.T
        )
    )
    constant = 0.5 * np.sum(interaction)
    return complex(np.trace(solver @ density) + pair + constant)


def _lyapunov_residual(
    F: ComplexArray,
    loss: ComplexArray,
    gain: ComplexArray,
    gamma: ComplexArray,
) -> float:
    A = -1j * F - loss - gain
    residual = A @ gamma + gamma @ A.conj().T + 2.0 * gain
    scale = max(
        np.linalg.norm(A, ord=2) * np.linalg.norm(gamma, ord="fro"),
        np.linalg.norm(2.0 * gain, ord="fro"),
        1.0,
    )
    return float(np.linalg.norm(residual, ord="fro") / scale)


@dataclass(frozen=True, slots=True)
class GHFIteration:
    """One deterministic fixed-point/DIIS iteration."""

    iteration: int
    self_consistency_residual: float
    target_lyapunov_residual: float
    update_norm: float
    fock_norm: float
    density_minimum_eigenvalue: float
    density_maximum_eigenvalue: float
    diis_vectors: int


@dataclass(frozen=True, slots=True)
class GHFResult:
    """A converged or exhausted standalone Gaussian mean-field calculation."""

    converged: bool
    gamma: ComplexArray
    fock: ComplexArray
    iterations: tuple[GHFIteration, ...]
    self_consistency_residual: float
    lyapunov_residual: float
    hermiticity_defect: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    physicality_defect: float


def _solve_mean_field(
    h_bare: ArrayLike,
    U: ArrayLike | SparseTwoBodyInteraction,
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    *,
    impurity_orbitals: ArrayLike | None = None,
    gamma0: ArrayLike | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 200,
    damping: float = 0.5,
    diis_start: int = 3,
    diis_size: int = 6,
    restricted: bool,
    charge_centered: bool,
    terminal_least_squares: bool = False,
) -> GHFResult:
    h = _square(h_bare, "h_bare")
    sparse_interaction = U if isinstance(U, SparseTwoBodyInteraction) else None
    interaction = None if sparse_interaction is not None else _interaction(U)
    loss = _square(gamma_loss, "gamma_loss")
    gain = _square(gamma_gain, "gamma_gain")
    if h.shape != loss.shape or h.shape != gain.shape:
        raise ValueError("h_bare, gamma_loss, and gamma_gain must match")
    _require_hermitian(h, "h_bare")
    _require_hermitian(loss, "gamma_loss")
    _require_hermitian(gain, "gamma_gain")
    if np.linalg.eigvalsh(loss)[0] < -1.0e-12 or np.linalg.eigvalsh(gain)[0] < -1.0e-12:
        raise ValueError("gamma_loss and gamma_gain must be positive semidefinite")
    if sparse_interaction is not None:
        if sparse_interaction.n_spin_orbitals != h.shape[0]:
            raise ValueError("the sparse interaction does not match h_bare")
        orbitals = None
        vertex = sparse_interaction
    else:
        assert interaction is not None
        orbitals = _orbitals(
            impurity_orbitals, interaction.shape[0], h.shape[0]
        )
        vertex = None if charge_centered else density_vertex(
            interaction, h.shape[0], orbitals
        )

    def fock(density: ComplexArray) -> ComplexArray:
        if sparse_interaction is not None:
            return wick_fock(h, sparse_interaction, density)
        if vertex is None:
            assert interaction is not None
            return charge_centered_fock(h, interaction, density, orbitals)
        return wick_fock(h, vertex, density)

    def covariance(F: ComplexArray) -> tuple[ComplexArray, float]:
        if not restricted:
            return finite_pm_covariance(F, loss, gain)
        spatial, residual = finite_pm_covariance(
            F[0::2, 0::2],
            loss[0::2, 0::2],
            gain[0::2, 0::2],
        )
        lifted = np.zeros_like(F)
        lifted[0::2, 0::2] = lifted[1::2, 1::2] = spatial
        return lifted, residual

    if restricted:
        _require_spin_restricted(h, "h_bare")
        _require_spin_restricted(loss, "gamma_loss")
        _require_spin_restricted(gain, "gamma_gain")
        if orbitals is not None and (
            np.any(orbitals[:, 0] % 2)
            or np.any(orbitals[:, 1] != orbitals[:, 0] + 1)
        ):
            raise ValueError(
                "restricted impurity_orbitals must be site-major spin pairs"
            )
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    if isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0,1]")
    if diis_start < 1 or diis_size < 1:
        raise ValueError("diis_start and diis_size must be positive")
    if not isinstance(terminal_least_squares, (bool, np.bool_)):
        raise TypeError("terminal_least_squares must be boolean")
    if terminal_least_squares and not restricted:
        raise ValueError("terminal least-squares refinement requires spin restriction")

    if gamma0 is None:
        gamma, _ = covariance(h)
    else:
        gamma = _square(gamma0, "gamma0")
        if gamma.shape != h.shape:
            raise ValueError("gamma0 must match h_bare")
        gamma = np.array(gamma, copy=True)
    if restricted:
        gamma = restrict_spin_density(gamma)

    targets: list[ComplexArray] = []
    residuals: list[ComplexArray] = []
    history: list[GHFIteration] = []
    converged = False
    F = fock(gamma)
    target = gamma

    for iteration in range(max_iterations):
        F = fock(gamma)
        if restricted:
            _require_spin_restricted(F, "F[gamma]")
        raw_target, target_lyapunov = covariance(F)
        target = (
            restrict_spin_density(raw_target)
            if restricted
            else raw_target
        )
        residual = target - gamma
        residual_norm = float(
            np.linalg.norm(residual, ord="fro")
            / max(np.linalg.norm(target, ord="fro"), 1.0)
        )
        eigenvalues = np.linalg.eigvalsh(
            0.5 * (target + target.conj().T)
        )
        targets.append(np.asarray(target))
        residuals.append(np.asarray(residual))
        if len(targets) > diis_size:
            targets.pop(0)
            residuals.pop(0)

        if iteration + 1 >= diis_start and len(targets) > 1:
            m = len(targets)
            B = np.empty((m + 1, m + 1), dtype=np.float64)
            for i in range(m):
                for j in range(m):
                    B[i, j] = np.vdot(residuals[i], residuals[j]).real
            B[:m, m] = 1.0
            B[m, :m] = 1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1, dtype=np.float64)
            rhs[m] = 1.0
            coefficients = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            mixed_target = sum(
                coefficient * value
                for coefficient, value in zip(coefficients, targets, strict=True)
            )
        else:
            mixed_target = target
        if restricted:
            mixed_target = restrict_spin_density(mixed_target)
        update = damping * (mixed_target - gamma)
        history.append(
            GHFIteration(
                iteration=iteration,
                self_consistency_residual=residual_norm,
                target_lyapunov_residual=float(target_lyapunov),
                update_norm=float(np.linalg.norm(update, ord="fro")),
                fock_norm=float(np.linalg.norm(F, ord="fro")),
                density_minimum_eigenvalue=float(eigenvalues[0]),
                density_maximum_eigenvalue=float(eigenvalues[-1]),
                diis_vectors=len(targets),
            )
        )

        if residual_norm <= tolerance:
            trial_gamma = np.asarray(target)
            trial_fock = fock(trial_gamma)
            raw_trial_target, _ = covariance(trial_fock)
            trial_target = (
                restrict_spin_density(raw_trial_target)
                if restricted
                else raw_trial_target
            )
            trial_residual = float(
                np.linalg.norm(trial_target - trial_gamma, ord="fro")
                / max(np.linalg.norm(trial_target, ord="fro"), 1.0)
            )
            true_residual = float(
                np.linalg.norm(raw_trial_target - trial_gamma, ord="fro")
                / max(np.linalg.norm(raw_trial_target, ord="fro"), 1.0)
            )
            if trial_residual <= tolerance and true_residual <= tolerance:
                gamma = trial_gamma
                F = trial_fock
                target = raw_trial_target
                converged = True
                break
        gamma = gamma + update
        if restricted:
            gamma = restrict_spin_density(gamma)

    if not converged and terminal_least_squares:
        n = h.shape[0] // 2

        def pack_restricted(value: ComplexArray) -> RealArray:
            spatial = value[0::2, 0::2]
            packed = [*spatial.diagonal().real]
            for i in range(n):
                for j in range(i + 1, n):
                    packed.extend((spatial[i, j].real, spatial[i, j].imag))
            return np.asarray(packed, dtype=np.float64)

        def unpack_restricted(values: RealArray) -> ComplexArray:
            spatial = np.zeros((n, n), dtype=np.complex128)
            spatial[np.diag_indices(n)] = values[:n]
            cursor = n
            for i in range(n):
                for j in range(i + 1, n):
                    spatial[i, j] = values[cursor] + 1j * values[cursor + 1]
                    spatial[j, i] = spatial[i, j].conjugate()
                    cursor += 2
            full = np.zeros_like(h)
            full[0::2, 0::2] = full[1::2, 1::2] = spatial
            return full

        def fixed_point_residual(values: RealArray) -> RealArray:
            trial_gamma = unpack_restricted(values)
            trial_fock = fock(trial_gamma)
            trial_target, _ = covariance(trial_fock)
            return pack_restricted(trial_target - trial_gamma)

        previous_gamma = np.asarray(gamma)
        refinement = least_squares(
            fixed_point_residual,
            pack_restricted(previous_gamma),
            jac="3-point",
            method="trf",
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=100,
        )
        trial_gamma = unpack_restricted(refinement.x)
        trial_fock = fock(trial_gamma)
        trial_target, target_lyapunov = covariance(trial_fock)
        trial_residual = float(
            np.linalg.norm(trial_target - trial_gamma, ord="fro")
            / max(np.linalg.norm(trial_target, ord="fro"), 1.0)
        )
        trial_eigenvalues = np.linalg.eigvalsh(trial_target[0::2, 0::2])
        history.append(
            GHFIteration(
                iteration=len(history),
                self_consistency_residual=trial_residual,
                target_lyapunov_residual=float(target_lyapunov),
                update_norm=float(np.linalg.norm(trial_gamma - previous_gamma, ord="fro")),
                fock_norm=float(np.linalg.norm(trial_fock, ord="fro")),
                density_minimum_eigenvalue=float(trial_eigenvalues[0]),
                density_maximum_eigenvalue=float(trial_eigenvalues[-1]),
                diis_vectors=0,
            )
        )
        if trial_residual <= tolerance:
            gamma = trial_gamma
            F = trial_fock
            target = trial_target
            converged = True

    if not converged:
        F = fock(gamma)
        target, _ = covariance(F)
    self_consistency = float(
        np.linalg.norm(target - gamma, ord="fro")
        / max(np.linalg.norm(target, ord="fro"), 1.0)
    )
    hermiticity = float(
        np.linalg.norm(gamma - gamma.conj().T, ord="fro")
        / max(np.linalg.norm(gamma, ord="fro"), 1.0)
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (gamma + gamma.conj().T))
    physicality = max(
        0.0, float(-eigenvalues[0]), float(eigenvalues[-1] - 1.0)
    )
    return GHFResult(
        converged=converged,
        gamma=np.asarray(gamma),
        fock=np.asarray(F),
        iterations=tuple(history),
        self_consistency_residual=self_consistency,
        lyapunov_residual=_lyapunov_residual(F, loss, gain, gamma),
        hermiticity_defect=hermiticity,
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        physicality_defect=physicality,
    )


def solve_ghf(
    h_bare: ArrayLike,
    U: ArrayLike | SparseTwoBodyInteraction,
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    *,
    impurity_orbitals: ArrayLike | None = None,
    gamma0: ArrayLike | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 200,
    damping: float = 0.5,
    diis_start: int = 3,
    diis_size: int = 6,
    restricted: bool = False,
    terminal_least_squares: bool = False,
) -> GHFResult:
    """Solve ``gamma = C[F[gamma]]`` by damped deterministic Pulay mixing.

    ``C[F]`` is the exact quadratic Lindblad covariance.  Pulay coefficients
    minimize the Frobenius norm of a linear combination of fixed-point
    residuals subject to coefficients summing to one.  The update remains
    damped; density eigenvalues are diagnosed but never clipped.
    """

    return _solve_mean_field(
        h_bare,
        U,
        gamma_loss,
        gamma_gain,
        impurity_orbitals=impurity_orbitals,
        gamma0=gamma0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        damping=damping,
        diis_start=diis_start,
        diis_size=diis_size,
        restricted=bool(restricted),
        charge_centered=True,
        terminal_least_squares=terminal_least_squares,
    )


def solve_rhf(
    h_bare: ArrayLike,
    U: ArrayLike,
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    *,
    impurity_orbitals: ArrayLike | None = None,
    gamma0: ArrayLike | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 200,
    damping: float = 0.5,
    diis_start: int = 3,
    diis_size: int = 6,
) -> GHFResult:
    r"""Solve the restricted-spin open-system Gaussian fixed point.

    The auxiliary orbitals must use site-major, spin-minor ordering.  At every
    fixed-point and DIIS iteration this solver imposes

    .. math::

       \gamma_{\uparrow\uparrow}=\gamma_{\downarrow\downarrow},\qquad
       \gamma_{\uparrow\downarrow}=\gamma_{\downarrow\uparrow}=0.

    The common spatial covariance remains complex, off diagonal, and
    generally non-idempotent.  The final residual is evaluated independently
    against the unprojected quadratic Lindblad covariance.
    """

    return _solve_mean_field(
        h_bare,
        U,
        gamma_loss,
        gamma_gain,
        impurity_orbitals=impurity_orbitals,
        gamma0=gamma0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        damping=damping,
        diis_start=diis_start,
        diis_size=diis_size,
        restricted=True,
        charge_centered=True,
    )


def solve_density_rhf(
    h: ArrayLike,
    U: ArrayLike,
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    *,
    impurity_orbitals: ArrayLike | None = None,
    gamma0: ArrayLike | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 200,
    damping: float = 0.5,
    diis_start: int = 3,
    diis_size: int = 6,
) -> GHFResult:
    r"""Solve restricted full HF for ``1/2 sum_(p!=q) V[p,q] n_p n_q``.

    The imposed spin restriction is identical to :func:`solve_rhf`, while
    the Fock matrix is the uncentered Wick contraction

    ``F = h + diag(V @ diag(gamma)) - V * gamma``.
    """

    return _solve_mean_field(
        h,
        U,
        gamma_loss,
        gamma_gain,
        impurity_orbitals=impurity_orbitals,
        gamma0=gamma0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        damping=damping,
        diis_start=diis_start,
        diis_size=diis_size,
        restricted=True,
        charge_centered=False,
    )


@dataclass(frozen=True, slots=True)
class QuadraticResponse:
    """Addition, removal, and retarded matrices on one frequency/time grid."""

    addition: ComplexArray
    removal: ComplexArray
    retarded: ComplexArray
    reconstruction_defect: float


def _selected(value: ArrayLike | None, n: int) -> IndexArray:
    if value is None:
        return np.arange(n, dtype=np.int64)
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.any(raw < 0)
        or np.any(raw >= n)
        or np.unique(raw).size != raw.size
    ):
        raise ValueError("orbitals must be distinct valid integer indices")
    return np.asarray(raw, dtype=np.int64)


def quadratic_frequency_response(
    F: ArrayLike,
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    gamma: ArrayLike,
    z: ArrayLike,
    *,
    orbitals: ArrayLike | None = None,
) -> QuadraticResponse:
    r"""Return Gaussian QRT response at complex shifts ``z``.

    The complete resolvent is

    ``G^R(z) = [z I-F+i(gamma_loss+gamma_gain)]^{-1}``.

    Its two QRT terms are ``G^R(I-gamma)`` and ``G^R gamma`` before projection
    to the requested sink/source orbitals.
    """

    fock = _square(F, "F")
    loss = _square(gamma_loss, "gamma_loss")
    gain = _square(gamma_gain, "gamma_gain")
    density = _square(gamma, "gamma")
    if fock.shape != loss.shape or fock.shape != gain.shape or fock.shape != density.shape:
        raise ValueError("quadratic response matrices must have identical shapes")
    shifts = np.asarray(z, dtype=np.complex128)
    if shifts.ndim != 1 or shifts.size == 0 or not np.all(np.isfinite(shifts)):
        raise ValueError("z must be a finite nonempty one-dimensional grid")
    selected = _selected(orbitals, fock.shape[0])
    identity = np.eye(fock.shape[0], dtype=np.complex128)
    addition = np.empty((shifts.size, selected.size, selected.size), dtype=np.complex128)
    removal = np.empty_like(addition)
    retarded = np.empty_like(addition)
    complement = identity - density
    for k, shift in enumerate(shifts):
        inverse = (
            shift * identity
            - fock
            + 1j * (loss + gain)
        )
        full = np.linalg.solve(inverse, identity)
        addition[k] = (full @ complement)[np.ix_(selected, selected)]
        removal[k] = (full @ density)[np.ix_(selected, selected)]
        retarded[k] = full[np.ix_(selected, selected)]
    defect = float(np.max(abs(retarded - addition - removal)))
    return QuadraticResponse(addition, removal, retarded, defect)


def quadratic_time_response(
    F: ArrayLike,
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    gamma: ArrayLike,
    t: ArrayLike,
    *,
    orbitals: ArrayLike | None = None,
) -> QuadraticResponse:
    r"""Return ``-i exp[(-iF-Lambda)t](I-gamma)``, ``-i exp[...]gamma``."""

    fock = _square(F, "F")
    loss = _square(gamma_loss, "gamma_loss")
    gain = _square(gamma_gain, "gamma_gain")
    density = _square(gamma, "gamma")
    if fock.shape != loss.shape or fock.shape != gain.shape or fock.shape != density.shape:
        raise ValueError("quadratic response matrices must have identical shapes")
    time = np.asarray(t, dtype=np.float64)
    if (
        time.ndim != 1
        or time.size == 0
        or not np.all(np.isfinite(time))
        or np.any(time < 0.0)
        or np.any(np.diff(time) < 0.0)
    ):
        raise ValueError("t must be a finite nondecreasing nonnegative grid")
    selected = _selected(orbitals, fock.shape[0])
    identity = np.eye(fock.shape[0], dtype=np.complex128)
    generator = -1j * fock - loss - gain
    complement = identity - density
    addition = np.empty((time.size, selected.size, selected.size), dtype=np.complex128)
    removal = np.empty_like(addition)
    retarded = np.empty_like(addition)
    for k, value in enumerate(time):
        propagator = -1j * scipy.linalg.expm(generator * value)
        addition[k] = (propagator @ complement)[np.ix_(selected, selected)]
        removal[k] = (propagator @ density)[np.ix_(selected, selected)]
        retarded[k] = propagator[np.ix_(selected, selected)]
    defect = float(np.max(abs(retarded - addition - removal)))
    return QuadraticResponse(addition, removal, retarded, defect)


@dataclass(frozen=True, slots=True)
class MolecularGHFObservables:
    """Molecular-site Wick observables from a Gaussian covariance."""

    one_body_density: ComplexArray
    n_up: RealArray
    n_down: RealArray
    occupations: RealArray
    double_occupancies: RealArray
    local_moments: RealArray
    bond_orders: RealArray
    particle_number: float
    particle_number_variance: float


def molecular_observables(
    gamma: ArrayLike,
    *,
    impurity_orbitals: ArrayLike,
    bonds: tuple[tuple[int, int], ...],
) -> MolecularGHFObservables:
    """Evaluate molecular densities, Wick doublons, moments, and bond orders."""

    density = _square(gamma, "gamma")
    raw_orbitals = np.asarray(impurity_orbitals)
    if raw_orbitals.ndim != 2 or raw_orbitals.shape[1] != 2:
        raise ValueError("impurity_orbitals must have shape (n_sites, 2)")
    orbitals = _orbitals(
        raw_orbitals, raw_orbitals.shape[0], density.shape[0]
    )
    molecular = density[np.ix_(orbitals.ravel(), orbitals.ravel())]
    n_up = np.diag(density)[orbitals[:, 0]].real
    n_down = np.diag(density)[orbitals[:, 1]].real
    double = np.empty(orbitals.shape[0], dtype=np.float64)
    for i, (up, down) in enumerate(orbitals):
        double[i] = float(
            (
                density[up, up] * density[down, down]
                - density[up, down] * density[down, up]
            ).real
        )
    occupations = n_up + n_down
    local_moments = occupations - 2.0 * double
    bond_orders = np.empty(len(bonds), dtype=np.float64)
    for k, (i, j) in enumerate(bonds):
        if not 0 <= i < orbitals.shape[0] or not 0 <= j < orbitals.shape[0]:
            raise ValueError("bond site is outside impurity_orbitals")
        value = 0.0 + 0.0j
        for spin in (0, 1):
            p = int(orbitals[i, spin])
            q = int(orbitals[j, spin])
            value += density[q, p] + density[p, q]
        bond_orders[k] = float(value.real)
    variance = float(
        np.trace(molecular @ (np.eye(molecular.shape[0]) - molecular)).real
    )
    return MolecularGHFObservables(
        one_body_density=np.asarray(molecular),
        n_up=np.asarray(n_up),
        n_down=np.asarray(n_down),
        occupations=np.asarray(occupations),
        double_occupancies=np.asarray(double),
        local_moments=np.asarray(local_moments),
        bond_orders=np.asarray(bond_orders),
        particle_number=float(np.trace(molecular).real),
        particle_number_variance=variance,
    )


__all__ = [
    "GHFIteration",
    "GHFResult",
    "MolecularGHFObservables",
    "QuadraticResponse",
    "centered_one_body",
    "charge_centered_fock",
    "charge_centered_wick_energy",
    "density_vertex",
    "molecular_observables",
    "quadratic_frequency_response",
    "quadratic_time_response",
    "restrict_spin_density",
    "solve_ghf",
    "solve_density_rhf",
    "solve_rhf",
    "wick_fock",
]

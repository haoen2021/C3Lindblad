"""Nonunitary quadratic normal modes with tilde-compatible normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from c3l.conventions import TolerancePolicy
from c3l.models import FermionicLindbladModel, QuadraticOpenSystem

from .generator import quadratic_mode_matrix

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
PhysicalModel = QuadraticOpenSystem | FermionicLindbladModel

NORMAL_MODE_CANONICALIZATION_VERSION = "tilde-bilagrangian-polar-v3"


def _immutable(array: ArrayLike, dtype: np.dtype[Any]) -> NDArray[Any]:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(
        contiguous.shape
    )
    immutable.flags.writeable = False
    return immutable


def _mode_input_matrices(
    model: PhysicalModel,
) -> tuple[ComplexMatrix, ComplexMatrix, ComplexMatrix]:
    if isinstance(model, QuadraticOpenSystem):
        return model.E, model.gamma_loss, model.gamma_gain
    if isinstance(model, FermionicLindbladModel):
        if model.has_interactions:
            raise ValueError(
                "normal_mode_reference requires a quadratic model; use a Hartree-Fock reference for interactions"
            )
        return model.E, model.gamma_loss, model.gamma_gain
    raise TypeError(
        "model must be a validated QuadraticOpenSystem or FermionicLindbladModel"
    )


def _explicit_mode_matrix(
    value: ArrayLike,
    name: str,
    policy: TolerancePolicy,
) -> ComplexMatrix:
    matrix = np.array(value, dtype=np.complex128, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    defect = np.linalg.norm(matrix - matrix.conj().T, ord="fro")
    threshold = policy.input_abs_rel.threshold(np.linalg.norm(matrix, ord="fro"))
    if defect > threshold:
        raise ValueError(
            f"{name} Hermiticity defect {defect:.3e} exceeds {threshold:.3e}"
        )
    return matrix


def _strict_degenerate_groups(
    eigenvalues: ComplexVector,
    indices: NDArray[np.int64],
    threshold: float,
) -> tuple[tuple[int, ...], ...]:
    """Group only eigenvalues indistinguishable at eigensolver roundoff."""

    remaining = set(int(index) for index in indices)
    groups: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        group = {
            index
            for index in remaining
            if abs(eigenvalues[index] - eigenvalues[seed]) <= threshold
        }
        remaining -= group
        groups.append(tuple(sorted(group)))
    return tuple(groups)


def _canonical_eigenspace_basis(raw_basis: ComplexMatrix) -> ComplexMatrix:
    """Choose a deterministic canonical-coordinate basis for one eigenspace.

    The orthogonal projector depends only on the right eigenspace, not on the
    arbitrary basis returned by LAPACK.  Repeatedly projecting the canonical
    coordinate with the largest unexplained norm gives a reproducible
    orthonormal basis.  Near ties are resolved by the lowest coordinate index.
    """

    dimension, rank = raw_basis.shape
    left, singular_values, _ = scipy.linalg.svd(
        raw_basis, full_matrices=False, check_finite=False
    )
    scale = max(float(singular_values[0]), 1.0)
    rank_threshold = 128.0 * np.finfo(np.float64).eps * scale * dimension
    if singular_values[-1] <= rank_threshold:
        raise ValueError("degenerate normal-mode eigenspace is rank deficient")
    orthogonal = left[:, :rank]
    projector = orthogonal @ orthogonal.conj().T
    basis: list[ComplexVector] = []
    for _ in range(rank):
        residuals = np.empty(dimension, dtype=np.float64)
        candidates: list[ComplexVector] = []
        for coordinate in range(dimension):
            vector = np.array(projector[:, coordinate], copy=True)
            for accepted in basis:
                vector -= accepted * np.vdot(accepted, vector)
            # A second pass controls loss of orthogonality for nonnormal input.
            for accepted in basis:
                vector -= accepted * np.vdot(accepted, vector)
            candidates.append(vector)
            residuals[coordinate] = np.linalg.norm(vector)
        maximum = float(np.max(residuals))
        vector_threshold = 128.0 * np.finfo(np.float64).eps * dimension
        if maximum <= vector_threshold:
            raise ValueError("failed to resolve a canonical degenerate-mode basis")
        tie_threshold = 128.0 * np.finfo(np.float64).eps * max(maximum, 1.0)
        pivot = int(np.flatnonzero(residuals >= maximum - tie_threshold)[0])
        basis.append(candidates[pivot] / residuals[pivot])
    return np.column_stack(basis)


def _tilde_adapt_strict_degeneracies(
    h: ComplexMatrix,
    eigenvalues: ComplexVector,
    eigenvectors: ComplexMatrix,
    policy: TolerancePolicy,
) -> tuple[ComplexVector, ComplexMatrix]:
    """Fix the basis gauge in exactly repeated tilde-paired eigenspaces.

    For a negative-half-plane right basis ``Vn``, pseudo-Hermiticity makes
    ``X @ conj(Vn)`` a positive-half-plane left basis.  Rotating the matching
    positive right block ``Wp`` by ``inv((X conj(Vn))^T Wp)`` makes the two
    bases algebraic duals, so tilde conjugation is monomial.  Only strict
    machine-scale degeneracies are rotated; ordinary nondegenerate modes are
    returned byte-for-byte unchanged by this helper.
    """

    values = np.array(eigenvalues, dtype=np.complex128, copy=True)
    vectors = np.array(eigenvectors, dtype=np.complex128, copy=True)
    dimension = h.shape[0]
    h_scale = max(float(np.linalg.norm(h, ord=2)), 1.0)
    strict_threshold = max(
        float(policy.input_abs_rel.absolute),
        128.0 * np.finfo(np.float64).eps * h_scale,
    )
    negative = np.flatnonzero(values.imag < 0.0).astype(np.int64)
    positive = np.flatnonzero(values.imag > 0.0).astype(np.int64)
    negative_groups = _strict_degenerate_groups(values, negative, strict_threshold)
    positive_groups = _strict_degenerate_groups(values, positive, strict_threshold)
    exchange = np.block(
        [
            [np.zeros((dimension // 2, dimension // 2)), np.eye(dimension // 2)],
            [np.eye(dimension // 2), np.zeros((dimension // 2, dimension // 2))],
        ]
    ).astype(np.complex128)
    used_positive: set[int] = set()
    for negative_group in negative_groups:
        if len(negative_group) == 1:
            continue
        negative_value = complex(np.mean(values[list(negative_group)]))
        matches = [
            group
            for group in positive_groups
            if len(group) == len(negative_group)
            and not used_positive.intersection(group)
            and abs(complex(np.mean(values[list(group)])) - negative_value.conjugate())
            <= strict_threshold
        ]
        if len(matches) != 1:
            raise ValueError(
                "strictly degenerate normal modes lack a unique conjugate eigenspace"
            )
        positive_group = matches[0]
        used_positive.update(positive_group)

        Vn = _canonical_eigenspace_basis(vectors[:, list(negative_group)])
        left_positive = exchange @ Vn.conj()
        Wp = vectors[:, list(positive_group)]
        overlap = left_positive.T @ Wp
        overlap_condition = float(np.linalg.cond(overlap, p=2))
        if (
            not np.isfinite(overlap_condition)
            or overlap_condition * np.finfo(np.float64).eps
            > policy.algebra_abs_rel.threshold(1.0)
        ):
            raise ValueError("tilde-paired degenerate eigenspaces are ill conditioned")
        Vp = scipy.linalg.solve(
            overlap.T, Wp.T, assume_a="gen", check_finite=False
        ).T

        vectors[:, list(negative_group)] = Vn
        vectors[:, list(positive_group)] = Vp
        values[list(negative_group)] = negative_value
        values[list(positive_group)] = negative_value.conjugate()

    return values, vectors


def _mode_pivot(vector: ComplexVector) -> int:
    """Return the lowest coordinate tied for largest magnitude at roundoff."""

    magnitudes = np.abs(vector)
    maximum = float(np.max(magnitudes))
    tie_threshold = (
        128.0
        * np.finfo(np.float64).eps
        * max(maximum, 1.0)
        * vector.size
    )
    return int(np.flatnonzero(magnitudes >= maximum - tie_threshold)[0])


def _normalize_mode_columns(vectors: ComplexMatrix) -> None:
    """Normalize columns and fix their deterministic pivot phases in place."""

    for p in range(vectors.shape[1]):
        norm = np.linalg.norm(vectors[:, p])
        if norm == 0.0:
            raise ValueError("normal-mode eigensolver returned a zero eigenvector")
        vectors[:, p] /= norm
        pivot = _mode_pivot(vectors[:, p])
        vectors[:, p] *= np.exp(-1j * np.angle(vectors[pivot, p]))
        if vectors[pivot, p].real < 0.0:
            vectors[:, p] *= -1.0


@dataclass(frozen=True, slots=True)
class _CanonicalizationDiagnostics:
    applied: bool
    trigger_threshold: np.float64
    neutral_defect_before: np.float64
    neutral_defect_after: np.float64
    relative_change: np.float64
    chart_condition: np.float64
    overlap_condition: np.float64


def _tilde_lagrangian_projection(
    h: ComplexMatrix,
    eigenvalues: ComplexVector,
    eigenvectors: ComplexMatrix,
    policy: TolerancePolicy,
) -> tuple[ComplexMatrix, _CanonicalizationDiagnostics]:
    r"""Project a numerically non-neutral stable subspace onto its exact form.

    The exchange matrix ``X`` obeys ``h = X h^\dagger X``.  Consequently both
    half-plane invariant subspaces are maximal ``X``-neutral:
    ``V_-^\dagger X V_-=V_+^\dagger X V_+=0``.  A weakly damped conjugate pair
    can amplify the backward error of an otherwise accurate eigensolve,
    making either identity fail at a level too large for the anti-linear
    tilde map.

    In the unitary coordinates that diagonalize ``X`` to ``diag(I,-I)``, write
    the stable subspace as the graph ``[I;R]``.  Exact neutrality is equivalent
    to ``R`` being unitary.  Replacing ``R`` by its polar unitary factor gives
    the nearest structure-preserving graph.  The same projection is applied
    to the positive graph before it is biorthogonalized against
    ``X conj(V_-)``.  Neutrality of both subspaces is required: neutralizing
    only the stable graph makes the creator tilde map appear monomial while
    its CAR-complementary annihilator sector remains nonmonomial.  The result
    is accepted only by the ordinary eigen-residual and conditioning gates
    applied below; no eigenvalue, physical matrix, or tolerance is changed.
    """

    dimension = h.shape[0]
    n = dimension // 2
    negative = np.flatnonzero(eigenvalues.imag < 0.0).astype(np.int64)
    positive = np.flatnonzero(eigenvalues.imag > 0.0).astype(np.int64)
    if negative.size != n or positive.size != n:
        zero = np.float64(0.0)
        return eigenvectors, _CanonicalizationDiagnostics(
            applied=False,
            trigger_threshold=np.float64(policy.algebra_abs_rel.threshold(1.0)),
            neutral_defect_before=zero,
            neutral_defect_after=zero,
            relative_change=zero,
            chart_condition=np.float64(1.0),
            overlap_condition=np.float64(1.0),
        )
    identity = np.eye(n, dtype=np.complex128)
    exchange = np.block(
        [
            [np.zeros((n, n)), identity],
            [identity, np.zeros((n, n))],
        ]
    )
    stable = eigenvectors[:, negative]
    positive_subspace = eigenvectors[:, positive]
    stable_neutral_before = np.float64(
        np.linalg.norm(stable.conj().T @ exchange @ stable, ord=2)
    )
    positive_neutral_before = np.float64(
        np.linalg.norm(
            positive_subspace.conj().T @ exchange @ positive_subspace,
            ord=2,
        )
    )
    neutral_before = np.float64(
        max(stable_neutral_before, positive_neutral_before)
    )
    # X-neutrality is an exact CAR identity, not an eigensolver residual.
    # X-neutrality is a zero identity.  Use the absolute algebra tolerance for
    # the trigger: a small relative-tolerance allowance can be amplified by a
    # nearly degenerate mode pair into a nonmonomial two-sided tilde action.
    trigger = np.float64(policy.algebra_abs_rel.absolute)
    if neutral_before <= trigger:
        return eigenvectors, _CanonicalizationDiagnostics(
            applied=False,
            trigger_threshold=trigger,
            neutral_defect_before=neutral_before,
            neutral_defect_after=neutral_before,
            relative_change=np.float64(0.0),
            chart_condition=np.float64(1.0),
            overlap_condition=np.float64(1.0),
        )

    transform = np.block([[identity, identity], [identity, -identity]]) / np.sqrt(2.0)
    graph_coordinates = transform.conj().T @ stable
    upper = graph_coordinates[:n]
    lower = graph_coordinates[n:]
    chart_condition = np.float64(np.linalg.cond(upper, p=2))
    condition_limit = (
        policy.normal_mode_condition_budget_fraction
        * policy.algebra_abs_rel.threshold(1.0)
    )
    if (
        not np.isfinite(chart_condition)
        or chart_condition * np.finfo(np.float64).eps > condition_limit
    ):
        raise ValueError("tilde-Lagrangian graph chart is ill conditioned")
    graph = scipy.linalg.solve(
        upper.T, lower.T, assume_a="gen", check_finite=False
    ).T
    left, _, right = scipy.linalg.svd(
        graph, full_matrices=False, check_finite=False
    )
    unitary_graph = left @ right
    projected_stable = transform @ np.vstack((upper, unitary_graph @ upper))

    pairing_cost = abs(
        eigenvalues[negative, None].conj() - eigenvalues[positive][None, :]
    )
    rows, columns = linear_sum_assignment(pairing_cost)
    if not np.array_equal(rows, np.arange(n)):
        raise ValueError("conjugate mode assignment is incomplete")
    paired_positive = positive[columns]
    pair_threshold = policy.eigensolver_abs_rel.threshold(
        max(float(np.linalg.norm(h, ord=2)), 1.0)
    )
    if np.max(pairing_cost[rows, columns]) > pair_threshold:
        raise ValueError("normal-mode spectrum lacks conjugate tilde partners")

    positive_modes = eigenvectors[:, paired_positive]
    positive_coordinates = transform.conj().T @ positive_modes
    positive_upper = positive_coordinates[:n]
    positive_lower = positive_coordinates[n:]
    positive_chart_condition = np.float64(np.linalg.cond(positive_upper, p=2))
    if (
        not np.isfinite(positive_chart_condition)
        or positive_chart_condition * np.finfo(np.float64).eps > condition_limit
    ):
        raise ValueError("positive tilde-Lagrangian graph chart is ill conditioned")
    positive_graph = scipy.linalg.solve(
        positive_upper.T,
        positive_lower.T,
        assume_a="gen",
        check_finite=False,
    ).T
    positive_left_graph, _, positive_right_graph = scipy.linalg.svd(
        positive_graph, full_matrices=False, check_finite=False
    )
    positive_unitary_graph = positive_left_graph @ positive_right_graph
    neutral_positive = transform @ np.vstack(
        (positive_upper, positive_unitary_graph @ positive_upper)
    )

    positive_left = exchange @ projected_stable.conj()
    overlap = positive_left.T @ neutral_positive
    overlap_condition = np.float64(np.linalg.cond(overlap, p=2))
    if (
        not np.isfinite(overlap_condition)
        or overlap_condition * np.finfo(np.float64).eps > condition_limit
    ):
        raise ValueError("tilde-paired mode overlap is ill conditioned")
    projected_positive = scipy.linalg.solve(
        overlap.T,
        neutral_positive.T,
        assume_a="gen",
        check_finite=False,
    ).T

    stable_relative_change = np.float64(
        np.linalg.norm(projected_stable - stable, ord="fro")
        / max(float(np.linalg.norm(stable, ord="fro")), 1.0)
    )
    positive_relative_change = np.float64(
        np.linalg.norm(neutral_positive - positive_modes, ord="fro")
        / max(float(np.linalg.norm(positive_modes, ord="fro")), 1.0)
    )
    # Cross-biorthogonalization fixes a basis gauge and may change individual
    # columns by O(1); the physically relevant projection change is the
    # subspace-neutralization step before that gauge choice.
    relative_change = np.float64(
        max(stable_relative_change, positive_relative_change)
    )
    result = np.array(eigenvectors, dtype=np.complex128, copy=True)
    result[:, negative] = projected_stable
    result[:, paired_positive] = projected_positive
    _normalize_mode_columns(result)
    projected = result[:, negative]
    stable_neutral_after = np.float64(
        np.linalg.norm(projected.conj().T @ exchange @ projected, ord=2)
    )
    positive_neutral_after = np.float64(
        np.linalg.norm(
            result[:, paired_positive].conj().T
            @ exchange
            @ result[:, paired_positive],
            ord=2,
        )
    )
    neutral_after = np.float64(
        max(stable_neutral_after, positive_neutral_after)
    )
    if stable_neutral_after > policy.algebra_abs_rel.threshold(1.0):
        raise ValueError("tilde-Lagrangian projection did not restore neutrality")
    if positive_neutral_after > policy.algebra_abs_rel.threshold(1.0):
        raise ValueError(
            "positive tilde-Lagrangian projection did not restore neutrality"
        )
    return result, _CanonicalizationDiagnostics(
        applied=True,
        trigger_threshold=trigger,
        neutral_defect_before=neutral_before,
        neutral_defect_after=neutral_after,
        relative_change=relative_change,
        chart_condition=np.float64(
            max(chart_condition, positive_chart_condition)
        ),
        overlap_condition=overlap_condition,
    )


@dataclass(frozen=True, slots=True)
class NormalModeReference:
    """Biorthogonal one-channel modes with explicit pre/post-PH spectra."""

    policy: TolerancePolicy
    h: ComplexMatrix
    eigenvalues: ComplexVector
    V: ComplexMatrix
    V_inv: ComplexMatrix
    vacuum_mask: NDArray[np.bool_]
    mode_charges: NDArray[np.int8]
    rapidities: ComplexVector
    original_order: NDArray[np.int64]
    condition_number: np.float64
    condition_roundoff: np.float64
    condition_limit: np.float64
    occupation_gap: np.float64
    occupation_threshold: np.float64
    eigen_residual: np.float64
    inverse_residual: np.float64
    scalar_eta: complex
    scalar_cancellation: np.float64
    canonicalization_version: str
    canonicalization_applied: bool
    canonicalization_trigger_threshold: np.float64
    canonicalization_neutral_defect_before: np.float64
    canonicalization_neutral_defect_after: np.float64
    canonicalization_relative_change: np.float64
    canonicalization_chart_condition: np.float64
    canonicalization_overlap_condition: np.float64

    def __post_init__(self) -> None:
        if not isinstance(self.policy, TolerancePolicy):
            raise TypeError("policy must be a validated TolerancePolicy")
        h = np.asarray(self.h, dtype=np.complex128)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] % 2:
            raise ValueError("h must be square with even dimension")
        num_modes = h.shape[0]
        if np.asarray(self.eigenvalues).shape != (num_modes,):
            raise ValueError("eigenvalue count does not match h")
        if (
            np.asarray(self.V).shape != h.shape
            or np.asarray(self.V_inv).shape != h.shape
        ):
            raise ValueError("V and V_inv must match h")
        raw_vacuum_mask = np.asarray(self.vacuum_mask)
        raw_mode_charges = np.asarray(self.mode_charges)
        raw_original_order = np.asarray(self.original_order)
        if raw_vacuum_mask.dtype.kind != "b":
            raise TypeError("vacuum_mask must have boolean dtype")
        if raw_mode_charges.dtype.kind not in "iu":
            raise TypeError("mode_charges must have integer dtype")
        if raw_original_order.dtype.kind not in "iu":
            raise TypeError("original_order must have integer dtype")
        for values in (
            self.vacuum_mask,
            self.mode_charges,
            self.rapidities,
            self.original_order,
        ):
            if np.asarray(values).shape != (num_modes,):
                raise ValueError("mode metadata length does not match h")
        V = np.asarray(self.V, dtype=np.complex128)
        V_inv = np.asarray(self.V_inv, dtype=np.complex128)
        eigenvalues = np.asarray(self.eigenvalues, dtype=np.complex128)
        vacuum_mask = np.asarray(self.vacuum_mask, dtype=np.bool_)
        mode_charges = np.asarray(self.mode_charges, dtype=np.int8)
        rapidities = np.asarray(self.rapidities, dtype=np.complex128)
        original_order = np.asarray(self.original_order, dtype=np.int64)
        arrays = (h, V, V_inv, eigenvalues, rapidities)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("normal-mode arrays must be finite")
        if not np.array_equal(np.sort(raw_original_order), np.arange(num_modes)):
            raise ValueError("original_order must be a permutation")
        ordering = sorted(
            range(num_modes),
            key=lambda p: (
                0 if eigenvalues[p].imag < 0.0 else 1,
                eigenvalues[p].imag,
                eigenvalues[p].real,
            ),
        )
        if ordering != list(range(num_modes)):
            raise ValueError("eigenvalues do not follow the frozen mode ordering")
        for p in range(num_modes):
            if not np.isclose(np.linalg.norm(V[:, p]), 1.0, atol=1.0e-12, rtol=1.0e-12):
                raise ValueError("normal-mode columns must have unit 2-norm")
            pivot = _mode_pivot(V[:, p])
            if abs(V[pivot, p].imag) > 1.0e-12 or V[pivot, p].real <= 0.0:
                raise ValueError("normal-mode pivot must be real and positive")
        if np.count_nonzero(vacuum_mask) != num_modes // 2:
            raise ValueError("vacuum_mask must select exactly half the modes")
        if not np.array_equal(vacuum_mask, eigenvalues.imag > 0.0):
            raise ValueError("vacuum_mask disagrees with eigenvalue half-planes")
        expected_charges = np.where(vacuum_mask, -1, 1).astype(np.int8)
        if not np.array_equal(raw_mode_charges, expected_charges):
            raise ValueError("mode_charges disagree with the PH convention")
        expected_rapidities = (1 - 2 * vacuum_mask.astype(np.int8)) * eigenvalues
        if not np.allclose(rapidities, expected_rapidities, atol=1.0e-13, rtol=1.0e-12):
            raise ValueError("rapidities disagree with the occupied-mode sign change")
        if np.linalg.matrix_rank(V) != num_modes:
            raise ValueError("V must be invertible")
        inverse_defect = np.linalg.norm(V_inv @ V - np.eye(num_modes), ord="fro")
        expected_inverse_residual = inverse_defect / np.sqrt(num_modes)
        inverse_threshold = self.policy.algebra_abs_rel.threshold(1.0)
        if expected_inverse_residual > inverse_threshold:
            raise ValueError("V_inv is not the algebraic inverse of V")
        if not np.isclose(
            self.inverse_residual,
            expected_inverse_residual,
            atol=1.0e-13,
            rtol=1.0e-10,
        ):
            raise ValueError("inverse-residual metadata is inconsistent")
        scalar_values = (
            self.condition_number,
            self.condition_roundoff,
            self.condition_limit,
            self.occupation_gap,
            self.occupation_threshold,
            self.eigen_residual,
            self.inverse_residual,
            self.scalar_cancellation,
            self.canonicalization_trigger_threshold,
            self.canonicalization_neutral_defect_before,
            self.canonicalization_neutral_defect_after,
            self.canonicalization_relative_change,
            self.canonicalization_chart_condition,
            self.canonicalization_overlap_condition,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in scalar_values):
            raise ValueError("normal-mode diagnostics must be finite and nonnegative")
        if self.canonicalization_version != NORMAL_MODE_CANONICALIZATION_VERSION:
            raise ValueError("normal-mode canonicalization version is unsupported")
        if not isinstance(self.canonicalization_applied, (bool, np.bool_)):
            raise TypeError("canonicalization_applied must be boolean")
        if (
            self.canonicalization_applied
            and self.canonicalization_neutral_defect_after
            > self.policy.algebra_abs_rel.threshold(1.0)
        ):
            raise ValueError("canonicalized stable subspace is not tilde neutral")
        expected_condition_limit = (
            self.policy.normal_mode_condition_budget_fraction
            * self.policy.algebra_abs_rel.threshold(1.0)
        )
        if not np.isclose(
            self.condition_limit,
            expected_condition_limit,
            atol=1.0e-30,
            rtol=1.0e-12,
        ):
            raise ValueError("condition-limit metadata disagrees with policy")
        if (
            self.condition_number < 1.0
            or self.condition_roundoff > self.condition_limit
        ):
            raise ValueError("normal-mode conditioning metadata is inconsistent")
        if not np.isclose(
            self.condition_roundoff,
            self.condition_number * np.finfo(np.float64).eps,
            atol=1.0e-30,
            rtol=1.0e-12,
        ):
            raise ValueError("condition-roundoff metadata is inconsistent")
        expected_gap = np.min(np.abs(eigenvalues.imag))
        if not np.isclose(
            self.occupation_gap, expected_gap, atol=1.0e-13, rtol=1.0e-12
        ):
            raise ValueError("occupation-gap metadata is inconsistent")
        if self.occupation_gap <= self.occupation_threshold:
            raise ValueError("normal-mode occupation metadata is ambiguous")
        expected_occupation_threshold = self.policy.eigensolver_abs_rel.threshold(
            np.linalg.norm(h, ord=2)
        )
        if not np.isclose(
            self.occupation_threshold,
            expected_occupation_threshold,
            atol=1.0e-30,
            rtol=1.0e-12,
        ):
            raise ValueError("occupation-threshold metadata disagrees with policy")
        expected_scalar = abs(
            complex(self.scalar_eta) + np.sum(eigenvalues[vacuum_mask])
        )
        if not np.isfinite(complex(self.scalar_eta).real) or not np.isfinite(
            complex(self.scalar_eta).imag
        ):
            raise ValueError("scalar_eta must be finite")
        if not np.isclose(
            self.scalar_cancellation, expected_scalar, atol=1.0e-13, rtol=1.0e-12
        ):
            raise ValueError("scalar-cancellation metadata is inconsistent")
        scalar_threshold = self.policy.algebra_abs_rel.threshold(
            max(abs(complex(self.scalar_eta)), 1.0)
        )
        if expected_scalar > scalar_threshold:
            raise ValueError("scalar cancellation exceeds policy")
        eigen_denominator = max(
            float(np.linalg.norm(h, ord="fro") * np.linalg.norm(V, ord="fro")),
            1.0e-13,
        )
        expected_eigen_residual = (
            np.linalg.norm(h @ V - V @ np.diag(eigenvalues), ord="fro")
            / eigen_denominator
        )
        if not np.isclose(
            self.eigen_residual,
            expected_eigen_residual,
            atol=1.0e-13,
            rtol=1.0e-10,
        ):
            raise ValueError("eigen-residual metadata is inconsistent")
        if expected_eigen_residual > self.policy.eigensolver_abs_rel.threshold(1.0):
            raise ValueError("eigen residual exceeds policy")
        expected_condition = np.linalg.cond(V, p=2)
        if not np.isclose(
            self.condition_number, expected_condition, atol=1.0e-13, rtol=1.0e-10
        ):
            raise ValueError("condition-number metadata is inconsistent")
        object.__setattr__(self, "h", _immutable(self.h, np.dtype(np.complex128)))
        object.__setattr__(
            self, "eigenvalues", _immutable(self.eigenvalues, np.dtype(np.complex128))
        )
        object.__setattr__(self, "V", _immutable(self.V, np.dtype(np.complex128)))
        object.__setattr__(
            self, "V_inv", _immutable(self.V_inv, np.dtype(np.complex128))
        )
        object.__setattr__(
            self, "vacuum_mask", _immutable(self.vacuum_mask, np.dtype(np.bool_))
        )
        object.__setattr__(
            self, "mode_charges", _immutable(self.mode_charges, np.dtype(np.int8))
        )
        object.__setattr__(
            self, "rapidities", _immutable(self.rapidities, np.dtype(np.complex128))
        )
        object.__setattr__(
            self, "original_order", _immutable(self.original_order, np.dtype(np.int64))
        )

    @property
    def n_orbitals(self) -> int:
        return self.h.shape[0] // 2

    def diagnostics_dict(self) -> dict[str, Any]:
        return {
            "eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in self.eigenvalues
            ],
            "vacuum_mask": self.vacuum_mask.tolist(),
            "mode_charges": self.mode_charges.tolist(),
            "rapidities": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in self.rapidities
            ],
            "original_order": self.original_order.tolist(),
            "condition_number": float(self.condition_number),
            "condition_roundoff": float(self.condition_roundoff),
            "condition_limit": float(self.condition_limit),
            "occupation_gap": float(self.occupation_gap),
            "occupation_threshold": float(self.occupation_threshold),
            "eigen_residual": float(self.eigen_residual),
            "inverse_residual": float(self.inverse_residual),
            "scalar_eta": {
                "real": float(self.scalar_eta.real),
                "imag": float(self.scalar_eta.imag),
            },
            "scalar_cancellation": float(self.scalar_cancellation),
            "canonicalization": {
                "version": self.canonicalization_version,
                "applied": bool(self.canonicalization_applied),
                "trigger_threshold": float(self.canonicalization_trigger_threshold),
                "neutral_defect_before": float(
                    self.canonicalization_neutral_defect_before
                ),
                "neutral_defect_after": float(
                    self.canonicalization_neutral_defect_after
                ),
                "relative_change": float(self.canonicalization_relative_change),
                "chart_condition": float(self.canonicalization_chart_condition),
                "overlap_condition": float(self.canonicalization_overlap_condition),
                "ordering": "negative half-plane first; imag then real",
                "phase": "largest-magnitude pivot real and positive",
            },
        }


def normal_mode_reference_from_arrays(
    E: ArrayLike,
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    policy: TolerancePolicy | None = None,
) -> NormalModeReference:
    """Build audited modes from finite Hermitian one-body matrices.

    Positive-semidefiniteness is intentionally not an input gate here.  A
    physical model validates that condition before calling this factory;
    archival sensitivity calculations may instead pass an explicitly
    labelled indefinite Gamma without changing the normal-mode algorithm.
    """

    tolerance_policy = TolerancePolicy.first() if policy is None else policy
    if not isinstance(tolerance_policy, TolerancePolicy):
        raise TypeError("policy must be a validated TolerancePolicy")
    E = _explicit_mode_matrix(E, "E", tolerance_policy)
    gamma_loss = _explicit_mode_matrix(gamma_loss, "gamma_loss", tolerance_policy)
    gamma_gain = _explicit_mode_matrix(gamma_gain, "gamma_gain", tolerance_policy)
    if E.shape != gamma_loss.shape or E.shape != gamma_gain.shape:
        raise ValueError("E, gamma_loss, and gamma_gain must have identical shapes")
    h = quadratic_mode_matrix(E, gamma_loss, gamma_gain)
    eigenvalues, V = scipy.linalg.eig(h)
    eigenvalues, V = _tilde_adapt_strict_degeneracies(
        h, eigenvalues, V, tolerance_policy
    )
    _normalize_mode_columns(V)
    V, canonicalization = _tilde_lagrangian_projection(
        h, eigenvalues, V, tolerance_policy
    )

    h_scale = np.linalg.norm(h, ord=2)
    occupation_threshold = tolerance_policy.eigensolver_abs_rel.threshold(h_scale)
    occupation_gap = np.float64(np.min(np.abs(eigenvalues.imag)))
    if occupation_gap <= occupation_threshold:
        raise ValueError(
            f"normal-mode occupation is ambiguous: gap={occupation_gap:.3e}, "
            f"threshold={occupation_threshold:.3e}"
        )
    n_orbitals = E.shape[0]
    negative = eigenvalues.imag < 0.0
    if np.count_nonzero(negative) != n_orbitals:
        raise ValueError(
            "normal-mode spectrum must split equally across damping half-planes"
        )

    original_order = np.array(
        sorted(
            range(eigenvalues.size),
            key=lambda p: (
                0 if eigenvalues[p].imag < 0.0 else 1,
                eigenvalues[p].imag,
                eigenvalues[p].real,
            ),
        ),
        dtype=np.int64,
    )
    eigenvalues = eigenvalues[original_order]
    V = V[:, original_order]
    V_inv = scipy.linalg.inv(V)

    condition_number = np.float64(np.linalg.cond(V, p=2))
    condition_roundoff = np.float64(condition_number * np.finfo(np.float64).eps)
    condition_limit = np.float64(
        tolerance_policy.normal_mode_condition_budget_fraction
        * tolerance_policy.algebra_abs_rel.threshold(1.0)
    )
    if condition_roundoff > condition_limit:
        raise ValueError(
            f"normal-mode eigenvectors are too ill conditioned: kappa*u="
            f"{condition_roundoff:.3e} exceeds {condition_limit:.3e}"
        )

    eigen_denominator = max(
        float(np.linalg.norm(h, ord="fro") * np.linalg.norm(V, ord="fro")),
        float(tolerance_policy.dense_scale_floor),
    )
    eigen_residual = np.float64(
        np.linalg.norm(h @ V - V @ np.diag(eigenvalues), ord="fro") / eigen_denominator
    )
    identity = np.eye(eigenvalues.size, dtype=np.complex128)
    inverse_residual = np.float64(
        np.linalg.norm(V_inv @ V - identity, ord="fro")
        / max(
            float(np.linalg.norm(identity, ord="fro")),
            float(tolerance_policy.dense_scale_floor),
        )
    )
    algebra_threshold = tolerance_policy.algebra_abs_rel.threshold(1.0)
    if eigen_residual > tolerance_policy.eigensolver_abs_rel.threshold(1.0):
        raise ValueError("normal-mode eigenvalue residual exceeds policy")
    if inverse_residual > algebra_threshold:
        raise ValueError("normal-mode inverse residual exceeds policy")

    vacuum_mask = eigenvalues.imag > 0.0
    mode_charges = np.where(vacuum_mask, -1, 1).astype(np.int8)
    rapidities = (1 - 2 * vacuum_mask.astype(np.int8)) * eigenvalues
    Lambda = gamma_loss + gamma_gain
    scalar_eta = complex(-np.trace(E) - 1j * np.trace(Lambda))
    scalar_cancellation = np.float64(abs(scalar_eta + np.sum(eigenvalues[vacuum_mask])))
    scalar_scale = max(abs(scalar_eta), 1.0)
    if scalar_cancellation > tolerance_policy.algebra_abs_rel.threshold(scalar_scale):
        raise ValueError("occupied-mode scalar cancellation exceeds policy")

    return NormalModeReference(
        policy=tolerance_policy,
        h=h,
        eigenvalues=eigenvalues,
        V=V,
        V_inv=V_inv,
        vacuum_mask=vacuum_mask,
        mode_charges=mode_charges,
        rapidities=rapidities,
        original_order=original_order,
        condition_number=condition_number,
        condition_roundoff=condition_roundoff,
        condition_limit=condition_limit,
        occupation_gap=occupation_gap,
        occupation_threshold=occupation_threshold,
        eigen_residual=eigen_residual,
        inverse_residual=inverse_residual,
        scalar_eta=scalar_eta,
        scalar_cancellation=scalar_cancellation,
        canonicalization_version=NORMAL_MODE_CANONICALIZATION_VERSION,
        canonicalization_applied=canonicalization.applied,
        canonicalization_trigger_threshold=canonicalization.trigger_threshold,
        canonicalization_neutral_defect_before=(
            canonicalization.neutral_defect_before
        ),
        canonicalization_neutral_defect_after=canonicalization.neutral_defect_after,
        canonicalization_relative_change=canonicalization.relative_change,
        canonicalization_chart_condition=canonicalization.chart_condition,
        canonicalization_overlap_condition=canonicalization.overlap_condition,
    )


def normal_mode_reference(
    model: PhysicalModel, policy: TolerancePolicy | None = None
) -> NormalModeReference:
    """Diagonalize ``h V = V epsilon`` for a validated physical model."""

    E, gamma_loss, gamma_gain = _mode_input_matrices(model)
    return normal_mode_reference_from_arrays(E, gamma_loss, gamma_gain, policy)


def quadratic_covariance(model: PhysicalModel) -> ComplexMatrix:
    """Solve ``A R + R A^dagger + 2 gamma_gain = 0`` independently."""

    E, gamma_loss, gamma_gain = _mode_input_matrices(model)
    Lambda = gamma_loss + gamma_gain
    A = -1j * E - Lambda
    return scipy.linalg.solve_sylvester(A, A.conj().T, -2.0 * gamma_gain)


def match_normal_modes(
    previous: NormalModeReference, current: NormalModeReference
) -> NDArray[np.int64]:
    """Match a parameter step by algebraic left-right overlap.

    The returned array gives the current index assigned to each previous mode.
    A half-plane crossing is rejected rather than silently changing the PH
    vacuum.
    """

    if previous.h.shape != current.h.shape:
        raise ValueError("normal-mode references must have the same dimension")
    overlap = np.abs(previous.V_inv @ current.V)
    rows, columns = linear_sum_assignment(-overlap)
    assignment = np.empty(previous.eigenvalues.size, dtype=np.int64)
    assignment[rows] = columns
    if not np.array_equal(previous.vacuum_mask, current.vacuum_mask[assignment]):
        raise ValueError("normal-mode matching crosses the occupation half-plane")
    return _immutable(assignment, np.dtype(np.int64))


def near_degenerate_groups(
    reference: NormalModeReference, tolerance: float | None = None
) -> tuple[tuple[int, ...], ...]:
    """Return connected same-half-plane eigenvalue multiplets.

    A group identifies an eigenspace whose individual eigenvectors may rotate
    under a parameter change.
    """

    threshold = (
        float(reference.occupation_threshold) if tolerance is None else float(tolerance)
    )
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("degeneracy tolerance must be finite and nonnegative")
    remaining = set(range(reference.eigenvalues.size))
    groups: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        group = {seed}
        changed = True
        while changed:
            changed = False
            for p in tuple(remaining - group):
                if reference.vacuum_mask[p] != reference.vacuum_mask[seed]:
                    continue
                if any(
                    abs(reference.eigenvalues[p] - reference.eigenvalues[q])
                    <= threshold
                    for q in group
                ):
                    group.add(p)
                    changed = True
        remaining -= group
        groups.append(tuple(sorted(group)))
    return tuple(groups)

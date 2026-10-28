"""Anti-linear tilde action on nonunitary particle-hole creator modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from c3l.conventions import TolerancePolicy
from c3l.superfermion.normal_modes import NormalModeReference


def _immutable(array: ArrayLike, dtype: np.dtype) -> NDArray:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(
        contiguous.shape
    )
    immutable.flags.writeable = False
    return immutable


@dataclass(frozen=True, slots=True)
class TildeMap:
    """Monomial anti-linear tilde map for spin or total-charge PH creators.

    ``tilde(bar_P_p) = factors[p] * bar_P_partners[p]``. Tilde preserves
    product order; determinant factors include the sign required to sort the
    mapped creator sequence back into canonical increasing order. Factors are
    nonzero complex scalars and need not have unit modulus for the frozen
    Euclidean normalization of nonunitary modes.
    """

    n: int
    partners: NDArray[np.int64]
    factors: NDArray[np.complex128]
    directional_residuals: NDArray[np.float64]
    double_tilde_defect: np.float64
    residual_threshold: np.float64
    double_tilde_threshold: np.float64
    symmetry: str = "spin"

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError("n must be a positive integer")
        if self.symmetry not in ("spin", "charge"):
            raise ValueError("symmetry must be 'spin' or 'charge'")
        num_modes = self.num_modes
        raw_partners = np.asarray(self.partners)
        if raw_partners.dtype.kind not in "iu":
            raise TypeError("tilde partners must have integer dtype")
        partners = np.asarray(self.partners, dtype=np.int64)
        factors = np.asarray(self.factors, dtype=np.complex128)
        residuals = np.asarray(self.directional_residuals, dtype=np.float64)
        if partners.shape != (num_modes,) or factors.shape != (num_modes,):
            raise ValueError(f"tilde partner/factor arrays must have length {num_modes}")
        if residuals.shape != (num_modes,):
            raise ValueError(f"directional_residuals must have length {num_modes}")
        if np.any(partners < 0) or np.any(partners >= num_modes):
            raise ValueError("tilde partner index is outside the mode layout")
        if not np.array_equal(partners[partners], np.arange(num_modes)):
            raise ValueError("tilde partners must be an involution")
        if np.any(partners == np.arange(num_modes)):
            raise ValueError("tilde modes may not be self-partners")
        if not np.all(np.isfinite(factors)) or np.any(factors == 0.0):
            raise ValueError("tilde factors must be finite and nonzero")
        if not np.all(np.isfinite(residuals)) or np.any(residuals < 0.0):
            raise ValueError("tilde residuals must be finite and nonnegative")
        if (
            not np.isfinite(self.residual_threshold)
            or not np.isfinite(self.double_tilde_threshold)
            or self.residual_threshold < 0.0
            or self.double_tilde_threshold < 0.0
        ):
            raise ValueError("tilde thresholds must be finite and nonnegative")
        if np.max(residuals) > self.residual_threshold:
            raise ValueError("tilde directional residual exceeds its threshold")
        defects = np.abs(factors.conj() * factors[partners] - 1.0)
        if not np.isclose(
            self.double_tilde_defect,
            np.max(defects),
            atol=1.0e-13,
            rtol=1.0e-12,
        ):
            raise ValueError("double-tilde metadata is inconsistent")
        if self.double_tilde_defect > self.double_tilde_threshold:
            raise ValueError("double-tilde defect exceeds its threshold")
        object.__setattr__(self, "partners", _immutable(partners, np.dtype(np.int64)))
        object.__setattr__(
            self, "factors", _immutable(factors, np.dtype(np.complex128))
        )
        object.__setattr__(
            self,
            "directional_residuals",
            _immutable(residuals, np.dtype(np.float64)),
        )

    @property
    def num_modes(self) -> int:
        return (2 if self.symmetry == "charge" else 4) * self.n

    def determinant(self, mask: int) -> tuple[int, complex]:
        """Return canonical tilde partner mask and its fermionic scalar."""

        if not 0 <= mask < (1 << self.num_modes):
            raise ValueError("mask is outside the tilde map")
        occupied = [p for p in range(self.num_modes) if mask & (1 << p)]
        mapped = [int(self.partners[p]) for p in occupied]
        inversions = sum(
            mapped[i] > mapped[j]
            for i in range(len(mapped))
            for j in range(i + 1, len(mapped))
        )
        factor = complex(np.prod(self.factors[occupied])) if occupied else 1.0 + 0.0j
        if inversions % 2:
            factor = -factor
        partner_mask = sum(1 << p for p in mapped)
        return partner_mask, factor


def _scaled_fit(target: NDArray, source: NDArray) -> tuple[complex, float]:
    denominator = np.vdot(source, source)
    if denominator == 0.0:
        raise ValueError("cannot fit a zero tilde source vector")
    coefficient = complex(np.vdot(source, target) / denominator)
    residual = float(
        np.linalg.norm(target - coefficient * source)
        / max(float(np.linalg.norm(target)), 1.0e-13)
    )
    return coefficient, residual


def _local_tilde_map(
    reference: NormalModeReference, policy: TolerancePolicy
) -> tuple[NDArray[np.int64], NDArray[np.complex128], NDArray[np.float64]]:
    n = reference.n_orbitals
    negative = np.flatnonzero(~reference.vacuum_mask)
    positive = np.flatnonzero(reference.vacuum_mask)
    exchange = np.block(
        [
            [np.zeros((n, n)), np.eye(n)],
            [np.eye(n), np.zeros((n, n))],
        ]
    ).astype(np.complex128)
    costs = np.empty((n, n), dtype=np.float64)
    fits: dict[tuple[int, int], tuple[complex, float]] = {}
    for row, p in enumerate(negative):
        source = exchange @ reference.V[:, p].conj()
        for column, q in enumerate(positive):
            target = reference.V_inv[q, :]
            alpha, residual = _scaled_fit(target, source)
            fits[row, column] = (alpha, residual)
            costs[row, column] = residual
    rows, columns = linear_sum_assignment(costs)
    threshold = policy.eigensolver_abs_rel.threshold(1.0)
    if np.max(costs[rows, columns]) > threshold:
        raise ValueError("tilde action is not monomial in the accepted normal modes")

    partners = np.empty(2 * n, dtype=np.int64)
    factors = np.empty(2 * n, dtype=np.complex128)
    residuals = np.empty(2 * n, dtype=np.float64)
    for row, column in zip(rows, columns, strict=True):
        p = int(negative[row])
        q = int(positive[column])
        alpha, residual = fits[row, column]
        partners[p] = q
        factors[p] = 1.0 / alpha
        residuals[p] = residual

        source_reverse = exchange @ reference.V_inv[q, :].conj()
        target_reverse = reference.V[:, p]
        beta, reverse_residual = _scaled_fit(target_reverse, source_reverse)
        partners[q] = p
        factors[q] = 1.0 / beta
        residuals[q] = reverse_residual

        # The two fits above establish the anti-linear action on the creator
        # pair bar_P_p, bar_P_q.  They do not test its CAR-complementary
        # annihilator pair.  That second direction requires W_p proportional
        # to X V_q^* and V_q proportional to X W_p^*.  Without both checks a
        # one-sided Lagrangian projection can appear monomial while physical
        # c/tilde-c forms still violate tilde covariance.
        complement_q, complement_q_residual = _scaled_fit(
            reference.V_inv[p, :],
            exchange @ reference.V[:, q].conj(),
        )
        complement_p, complement_p_residual = _scaled_fit(
            reference.V[:, q],
            exchange @ reference.V_inv[p, :].conj(),
        )
        factor_q_defect = abs(complement_q - factors[q]) / max(
            abs(complement_q), abs(factors[q]), 1.0e-13
        )
        factor_p_defect = abs(complement_p - factors[p]) / max(
            abs(complement_p), abs(factors[p]), 1.0e-13
        )
        residuals[p] = max(
            residuals[p],
            complement_p_residual,
            factor_p_defect,
        )
        residuals[q] = max(
            residuals[q],
            complement_q_residual,
            factor_q_defect,
        )
    return partners, factors, residuals


def tilde_map_from_references(
    up: NormalModeReference,
    down: NormalModeReference | None = None,
    policy: TolerancePolicy | None = None,
) -> TildeMap:
    """Derive tilde partners for one full charge block or two spin blocks."""

    if down is not None and up.n_orbitals != down.n_orbitals:
        raise ValueError("up and down references must have the same n")
    tolerance_policy = TolerancePolicy.first() if policy is None else policy
    n = up.n_orbitals
    references = (up,) if down is None else (up, down)
    num_modes = 2 * n * len(references)
    partners = np.empty(num_modes, dtype=np.int64)
    factors = np.empty(num_modes, dtype=np.complex128)
    residuals = np.empty(num_modes, dtype=np.float64)
    for spin, reference in enumerate(references):
        local_partners, local_factors, local_residuals = _local_tilde_map(
            reference, tolerance_policy
        )
        base = 2 * n * spin
        partners[base : base + 2 * n] = local_partners + base
        factors[base : base + 2 * n] = local_factors
        residuals[base : base + 2 * n] = local_residuals
    double_defect = np.float64(np.max(np.abs(factors.conj() * factors[partners] - 1.0)))
    residual_threshold = tolerance_policy.eigensolver_abs_rel.threshold(1.0)
    double_threshold = tolerance_policy.algebra_abs_rel.threshold(1.0)
    if np.max(residuals) > residual_threshold:
        raise ValueError("derived tilde directional residual exceeds policy")
    if double_defect > double_threshold:
        raise ValueError("derived mode factors violate double-tilde invariance")
    return TildeMap(
        n=n,
        partners=partners,
        factors=factors,
        directional_residuals=residuals,
        double_tilde_defect=double_defect,
        residual_threshold=residual_threshold,
        double_tilde_threshold=double_threshold,
        symmetry="charge" if down is None else "spin",
    )

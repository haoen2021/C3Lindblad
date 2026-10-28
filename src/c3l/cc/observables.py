"""Low-rank CC density coefficients for one- and two-body observables.

Only creator ranks zero, two and four contribute to these trace contractions:
exp(T)|rho_0> = (1 + T1 + T2 + T1 wedge T1 / 2 + ...) |rho_0>.
The construction applies to spin- and charge-conserving models and does not
form the full density matrix.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import comb
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray
from c3l.ci.basis import CIBasis, ness_basis
from c3l.ci.matrix_free import (
    FactorizedModeProduct,
    factorized_mode_product_expectation,
    physical_mode_operator,
)
from c3l.ci.reference import HartreeFockReference
from c3l.ci.sectors import mode_layout_from_references
from .basis import ClusterBasis

ComplexVector = NDArray[np.complex128]


def _immutable_complex(value: ArrayLike) -> NDArray[np.complex128]:
    contiguous = np.ascontiguousarray(value, dtype=np.complex128)
    frozen = np.frombuffer(contiguous.tobytes(), dtype=np.complex128).reshape(
        contiguous.shape
    )
    frozen.flags.writeable = False
    return frozen


@dataclass(frozen=True, slots=True)
class LowRankCCState:
    """The exact rank-0/2/4 coefficient prefix needed by static observables."""

    basis: CIBasis
    coefficients: ComplexVector
    connected_rank_two_dimension: int
    connected_rank_four_dimension: int
    scientific_scope: str = "exact_rank_0_2_4_prefix_for_one_and_two_body_expectations"

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.complex128)
        if self.basis.spec.sector != "ness" or self.basis.spec.max_ph_rank != 2:
            raise ValueError("low-rank CC observables require the R0/R2/R4 NESS basis")
        if coefficients.shape != (self.basis.dimension,):
            raise ValueError("coefficients do not match the low-rank NESS basis")
        if not np.all(np.isfinite(coefficients)) or coefficients[0] != 1.0:
            raise ValueError("low-rank coefficients must be finite and trace fixed")
        object.__setattr__(self, "coefficients", _immutable_complex(coefficients))

    @property
    def dimension(self) -> int:
        return self.basis.dimension

    @property
    def rank_zero_dimension(self) -> int:
        return 1

    @property
    def rank_two_dimension(self) -> int:
        return sum(rank == 1 for rank in self.basis.ranks)

    @property
    def rank_four_dimension(self) -> int:
        return sum(rank == 2 for rank in self.basis.ranks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficient_dimension": self.dimension,
            "creator_rank_dimensions": {
                "R0": self.rank_zero_dimension,
                "R2": self.rank_two_dimension,
                "R4": self.rank_four_dimension,
            },
            "connected_rank_two_dimension": self.connected_rank_two_dimension,
            "connected_rank_four_dimension": self.connected_rank_four_dimension,
            "maximum_creator_rank": 4,
            "complete_sector_constructed": self.dimension
            == comb(2 * self.basis.layout.n, self.basis.layout.n)
            ** (1 if self.basis.layout.symmetry == "charge" else 2),
            "scientific_scope": self.scientific_scope,
        }


def _wedge_sign(left_mask: int, right_mask: int) -> float:
    """Sign for sorting ``tau_left tau_right`` into canonical creator order."""

    inversions = sum(
        (right_mask & ((1 << p) - 1)).bit_count()
        for p in range(left_mask.bit_length())
        if left_mask & (1 << p)
    )
    return -1.0 if inversions % 2 else 1.0


def low_rank_cc_coefficients(
    reference: HartreeFockReference,
    cluster_basis: ClusterBasis,
    amplitudes: ArrayLike,
) -> LowRankCCState:
    """Form only the rank-0/2/4 prefix of ``exp(T)|rho_0>``.

    Connected rank-two amplitudes enter directly.  At creator rank four the
    coefficient is ``T_2 + T_1 wedge T_1 / 2``.  The pair loop visits each
    unordered pair once; two even creator strings commute, so this is exactly
    the factor ``1/2`` in the exponential without forming either ordered term.
    """

    if not isinstance(reference, HartreeFockReference):
        raise TypeError("reference must be a HartreeFockReference")
    if not isinstance(cluster_basis, ClusterBasis):
        raise TypeError("cluster_basis must be a deterministic ClusterBasis")
    if cluster_basis.num_modes != reference.num_ph_modes:
        raise ValueError("cluster basis and reference mode counts disagree")
    if cluster_basis.maximum_rank > 4:
        raise ValueError("static low-rank observables accept CCS or CCSD amplitudes")
    t = np.asarray(amplitudes, dtype=np.complex128)
    if t.shape != (cluster_basis.dimension,) or not np.all(np.isfinite(t)):
        raise ValueError("amplitudes must be finite and match the cluster basis")

    layout = mode_layout_from_references(*reference.mode_references)
    basis = ness_basis(reference.n_sites, 2, layout)
    for mask in cluster_basis.masks:
        if mask not in basis.index_of:
            raise ValueError(
                "cluster basis must contain only charge-neutral rank-two/four masks"
            )

    coefficients = np.zeros(basis.dimension, dtype=np.complex128)
    coefficients[0] = 1.0
    for value, mask in zip(t, cluster_basis.masks, strict=True):
        coefficients[basis.index_of[mask]] = value

    singles = [
        (mask, t[i])
        for i, (mask, rank) in enumerate(
            zip(cluster_basis.masks, cluster_basis.ranks, strict=True)
        )
        if rank == 2
    ]
    for i, (left_mask, left_value) in enumerate(singles):
        if left_value == 0.0:
            continue
        for right_mask, right_value in singles[i + 1 :]:
            if right_value == 0.0 or left_mask & right_mask:
                continue
            target = left_mask | right_mask
            coefficients[basis.index_of[target]] += (
                _wedge_sign(left_mask, right_mask) * left_value * right_value
            )

    return LowRankCCState(
        basis=basis,
        coefficients=coefficients,
        connected_rank_two_dimension=sum(rank == 2 for rank in cluster_basis.ranks),
        connected_rank_four_dimension=sum(rank == 4 for rank in cluster_basis.ranks),
    )


def factorized_ccsd_covariance(
    reference: HartreeFockReference,
    cluster_basis: ClusterBasis,
    amplitudes: ArrayLike,
) -> NDArray[np.complex128]:
    """Return ``C[p,q] = <c_q^dagger c_p>`` in physical orbital order.

    Only the low-rank coefficient prefix is formed. Values are not
    symmetrized or clipped, so numerical defects remain visible.
    """

    state = low_rank_cc_coefficients(reference, cluster_basis, amplitudes)
    num_orbitals = reference.model.n_spin_orbitals
    annihilators = tuple(
        physical_mode_operator(
            reference, *reference.model.orbital_site_spin(orbital), "c"
        )
        for orbital in range(num_orbitals)
    )
    creators = tuple(
        physical_mode_operator(
            reference, *reference.model.orbital_site_spin(orbital), "c_dagger"
        )
        for orbital in range(num_orbitals)
    )
    covariance = np.empty((num_orbitals, num_orbitals), dtype=np.complex128)
    for row, annihilator in enumerate(annihilators):
        for column, creator in enumerate(creators):
            product = FactorizedModeProduct(
                "CCSD_covariance", 1.0, (creator, annihilator)
            )
            covariance[row, column] = factorized_mode_product_expectation(
                state.basis, state.coefficients, product
            )
    return _immutable_complex(covariance)

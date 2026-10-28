"""Construct the odd response basis shared by S13 and full S135 solvers.

S13 contains one and three particle-hole creators. S135 adds the entire
rank-five shell, whose dimension is recorded here; the tensor solvers build
its coordinates separately. Optional explicit shell masks are for checks.
"""

from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from c3l.ci.basis import CIBasis, response_basis
from c3l.ci.sectors import (
    full_response_dimension,
    mode_layout_from_references,
    response_dimension,
)
from .factorized import FactorizedCCProblem


@dataclass(frozen=True, slots=True)
class EOMCCSDProblem:
    """Deterministic rank-one/rank-three EOM space and rank-five shell."""

    cc: FactorizedCCProblem
    response_spin: int
    basis: CIBasis
    next_rank_masks: tuple[int, ...]
    next_rank_dimension_count: int
    retained_index: Mapping[int, int]
    next_rank_index: Mapping[int, int]
    complete_response_dimension: int
    maximum_basis_dimension_constructed: int

    def __post_init__(self) -> None:
        if self.cc.level != "CCSD":
            raise ValueError("EOM-CCSD requires a factorized CCSD problem")
        if self.response_spin not in (0, 1):
            raise ValueError("response_spin must be zero or one")
        if self.basis.layout.symmetry == "charge" and self.response_spin != 0:
            raise ValueError("charge symmetry requires response_spin=0")
        if self.basis.spec.sector != "response" or self.basis.spec.max_ph_rank != 1:
            raise ValueError(
                "the EOM-CCSD basis must contain creator ranks one and three"
            )
        if any(mask.bit_count() != 5 for mask in self.next_rank_masks):
            raise ValueError("the immediate EOM exterior must have creator rank five")
        if self.next_rank_dimension_count < len(self.next_rank_masks):
            raise ValueError(
                "next-rank dimension metadata is smaller than stored masks"
            )
        if self.maximum_basis_dimension_constructed > self.complete_response_dimension:
            raise ValueError("constructed EOM dimension exceeds the complete sector")
        if (
            self.basis.layout.symmetry == "spin"
            and self.maximum_basis_dimension_constructed == self.complete_response_dimension
        ):
            raise ValueError(
                "the scalable EOM path must not construct the complete sector"
            )
        object.__setattr__(
            self, "retained_index", MappingProxyType(dict(self.retained_index))
        )
        object.__setattr__(
            self, "next_rank_index", MappingProxyType(dict(self.next_rank_index))
        )

    @property
    def dimension(self) -> int:
        return self.basis.dimension

    @property
    def next_rank_dimension(self) -> int:
        return self.next_rank_dimension_count

    def construction_record(self) -> dict[str, int | bool | str]:
        return {
            "method": "factorized_EOM_CCSD",
            "operator_convention": "printed_K_equals_i_L_response",
            "response_spin": self.response_spin,
            "retained_creator_ranks": "1,3",
            "retained_dimension": self.dimension,
            "next_creator_rank": 5,
            "next_rank_dimension": self.next_rank_dimension,
            "maximum_basis_dimension_constructed": self.maximum_basis_dimension_constructed,
            "complete_odd_union_dimension": self.complete_response_dimension,
            "complete_odd_sector_constructed": self.maximum_basis_dimension_constructed == self.complete_response_dimension,
            "complete_physical_density_vector_constructed": False,
            "dense_response_matrix_constructed": False,
        }


def eom_ccsd_problem(
    cc: FactorizedCCProblem,
    *,
    response_spin: int = 0,
    build_next_rank_masks: bool = False,
) -> EOMCCSDProblem:
    """Build the polynomial-size EOM-CCSD odd space and its next shell."""

    if not isinstance(cc, FactorizedCCProblem):
        raise TypeError("cc must be a factorized CC problem")
    if response_spin not in (0, 1):
        raise ValueError("response_spin must be zero or one")
    layout = mode_layout_from_references(cc.reference.up, cc.reference.down)
    retained = response_basis(
        cc.model.n_sites, 1, response_spin=response_spin, layout=layout
    )
    n = cc.model.n_sites
    next_dimension = (
        response_dimension(n, 2, symmetry=layout.symmetry)
        - response_dimension(n, 1, symmetry=layout.symmetry)
    )
    if build_next_rank_masks:
        through_next = response_basis(n, 2, response_spin=response_spin, layout=layout)
        next_masks = tuple(
            mask
            for mask, rank in zip(through_next.masks, through_next.ranks, strict=True)
            if rank == 2
        )
        maximum_basis_dimension = through_next.dimension
    else:
        next_masks = ()
        maximum_basis_dimension = retained.dimension
    return EOMCCSDProblem(
        cc=cc,
        response_spin=response_spin,
        basis=retained,
        next_rank_masks=next_masks,
        next_rank_dimension_count=next_dimension,
        retained_index={mask: i for i, mask in enumerate(retained.masks)},
        next_rank_index={mask: i for i, mask in enumerate(next_masks)},
        complete_response_dimension=full_response_dimension(n, symmetry=layout.symmetry),
        maximum_basis_dimension_constructed=maximum_basis_dimension,
    )

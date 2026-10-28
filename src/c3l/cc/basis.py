"""Deterministic pure-creation coupled-cluster amplitude spaces.

After the particle--hole transformation, a connected CC excitation is a
canonically ordered even string of ``bar_P`` creators.  CCS contains strings
with two creators and CCSD also contains strings with four creators.  Charge
neutrality is imposed independently in every supplied charge column.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np



@dataclass(frozen=True, slots=True)
class ClusterBasis:
    """Connected creation strings ordered by ``(creator rank, mask)``.

    ``masks[mu]`` is the bit mask of the canonical product ``tau_mu`` and
    ``ranks[mu]`` is its number of creators (two for CCS, four for doubles).
    The reference mask is deliberately absent: there is no scalar ``T_0``.
    """

    num_modes: int
    masks: tuple[int, ...]
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_modes, bool)
            or not isinstance(self.num_modes, int)
            or self.num_modes < 1
        ):
            raise ValueError("num_modes must be a positive integer")
        masks = tuple(int(mask) for mask in self.masks)
        ranks = tuple(int(rank) for rank in self.ranks)
        if len(masks) != len(ranks):
            raise ValueError("masks and ranks must be parallel")
        if len(set(masks)) != len(masks):
            raise ValueError("cluster basis contains duplicate masks")
        if any(mask <= 0 or mask >= (1 << self.num_modes) for mask in masks):
            raise ValueError("cluster masks must be nonzero and inside the mode space")
        if any(rank < 2 or rank % 2 for rank in ranks):
            raise ValueError("cluster creator ranks must be positive and even")
        if any(rank != mask.bit_count() for rank, mask in zip(ranks, masks, strict=True)):
            raise ValueError("cluster ranks must equal mask populations")
        if list(zip(ranks, masks, strict=True)) != sorted(
            zip(ranks, masks, strict=True)
        ):
            raise ValueError("cluster strings must be ordered by rank and mask")
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "ranks", ranks)

    @property
    def dimension(self) -> int:
        return len(self.masks)

    @property
    def maximum_rank(self) -> int:
        return max(self.ranks, default=0)


def _charge_table(charges: np.ndarray) -> np.ndarray:
    raw = np.asarray(charges)
    if raw.ndim not in (1, 2) or raw.shape[0] == 0:
        raise ValueError("charges must have shape (num_modes,) or (num_modes,n_charge)")
    if raw.dtype.kind not in "iu":
        raise TypeError("charges must have integer dtype")
    return np.asarray(raw[:, None] if raw.ndim == 1 else raw, dtype=np.int64)


def cluster_basis(charges: np.ndarray, maximum_rank: int) -> ClusterBasis:
    """Return all even strings neutral in every charge through a creator rank.

    For the spinful C3L layout, ``charges`` has two columns containing
    ``(Q_up,Q_down)``.  A one-column array is also useful for spinless oracle
    models.  Enumeration is deterministic and independent of hash order.
    """

    q = _charge_table(charges)
    if (
        isinstance(maximum_rank, bool)
        or not isinstance(maximum_rank, int)
        or maximum_rank < 2
        or maximum_rank % 2
    ):
        raise ValueError("maximum_rank must be an even integer of at least two")
    records: list[tuple[int, int]] = []
    for rank in range(2, min(maximum_rank, q.shape[0]) + 1, 2):
        for idx in combinations(range(q.shape[0]), rank):
            if np.all(np.sum(q[np.asarray(idx)], axis=0) == 0):
                records.append((rank, sum(1 << p for p in idx)))
    records.sort()
    return ClusterBasis(
        q.shape[0],
        tuple(mask for _, mask in records),
        tuple(rank for rank, _ in records),
    )


def ccs_basis(charges: np.ndarray) -> ClusterBasis:
    r"""Return the deterministic connected two-creator ($T_1$) space."""

    return cluster_basis(charges, 2)


def ccsd_basis(charges: np.ndarray) -> ClusterBasis:
    r"""Return the deterministic connected two- and four-creator space."""

    return cluster_basis(charges, 4)

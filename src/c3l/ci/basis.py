"""Particle-hole determinant masks and fermionic signs for CC/EOM coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sectors import ModeLayout, canonical_mode_layout

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]


def _immutable(array: ArrayLike, dtype: np.dtype) -> NDArray:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(
        contiguous.shape
    )
    immutable.flags.writeable = False
    return immutable


@dataclass(frozen=True, slots=True)
class ExcitationSpec:
    """Excitation sector, maximum particle-hole rank, and spin charge."""

    max_ph_rank: int
    sector: str = "ness"
    response_spin: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_ph_rank, bool)
            or not isinstance(self.max_ph_rank, int)
            or self.max_ph_rank < 0
        ):
            raise ValueError("max_ph_rank must be a nonnegative integer")
        if self.sector not in ("ness", "response"):
            raise ValueError("sector must be 'ness' or 'response'")
        if self.sector == "ness" and self.response_spin is not None:
            raise ValueError("NESS specifications do not have a response spin")
        if self.sector == "response" and self.response_spin not in (0, 1):
            raise ValueError("response_spin must be 0 or 1")


@dataclass(frozen=True, slots=True)
class CIBasis:
    """Canonical mask ordering with immutable ranks, charges, and lookup."""

    layout: ModeLayout
    spec: ExcitationSpec
    masks: tuple[int, ...]
    ranks: tuple[int, ...]
    charge_labels: NDArray[np.int8]
    index_of: Mapping[int, int] = field(init=False)

    def __post_init__(self) -> None:
        if any(
            isinstance(mask, (bool, np.bool_))
            or not isinstance(mask, (int, np.integer))
            for mask in self.masks
        ):
            raise TypeError("CI masks must be integers")
        if any(
            isinstance(rank, (bool, np.bool_))
            or not isinstance(rank, (int, np.integer))
            for rank in self.ranks
        ):
            raise TypeError("CI ranks must be integers")
        masks = tuple(int(mask) for mask in self.masks)
        ranks = tuple(int(rank) for rank in self.ranks)
        raw_charges = np.asarray(self.charge_labels)
        if raw_charges.dtype.kind not in "iu":
            raise TypeError("charge_labels must have integer dtype")
        charges = np.asarray(self.charge_labels, dtype=np.int8)
        if len(ranks) != len(masks):
            raise ValueError("masks and ranks must be parallel one-dimensional arrays")
        if charges.shape != (len(masks), 2):
            raise ValueError("charge_labels must have shape (dimension, 2)")
        if any(mask < 0 or mask >= (1 << self.layout.num_modes) for mask in masks):
            raise ValueError("CI basis mask is outside the mode layout")
        if len(set(masks)) != len(masks):
            raise ValueError("CI basis contains duplicate masks")
        if masks and masks[0] != 0 and self.spec.sector == "ness":
            raise ValueError("NESS reference vacuum must be first")
        expected_charges = np.array(
            [self.layout.charge(mask) for mask in masks], dtype=np.int8
        )
        if not np.array_equal(raw_charges, expected_charges):
            raise ValueError("recorded charge labels do not match masks")
        if self.spec.sector == "ness":
            if np.any(expected_charges != 0):
                raise ValueError("NESS basis contains a nonzero-charge state")
            expected_ranks = tuple(mask.bit_count() // 2 for mask in masks)
            ordering = [(rank, mask) for rank, mask in zip(ranks, masks, strict=True)]
        else:
            spin = int(self.spec.response_spin)
            if self.layout.symmetry == "charge" and spin != 0:
                raise ValueError("charge symmetry requires response_spin=0")
            allowed = {(1, 0), (-1, 0)} if spin == 0 else {(0, 1), (0, -1)}
            if not all(tuple(charge) in allowed for charge in expected_charges):
                raise ValueError(
                    "response basis contains a state outside its declared sector"
                )
            expected_ranks = tuple((mask.bit_count() - 1) // 2 for mask in masks)
            ordering = [
                (rank, tuple(charge), mask)
                for rank, charge, mask in zip(
                    ranks, expected_charges, masks, strict=True
                )
            ]
        if ranks != expected_ranks:
            raise ValueError("recorded PH ranks do not match determinant populations")
        if ordering != sorted(ordering):
            raise ValueError("CI basis is not in deterministic canonical order")
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(
            self, "charge_labels", _immutable(charges, np.dtype(np.int8))
        )
        object.__setattr__(
            self,
            "index_of",
            MappingProxyType({mask: index for index, mask in enumerate(masks)}),
        )

    @property
    def dimension(self) -> int:
        return len(self.masks)

    def is_tilde_closed(self) -> bool:
        return all(
            self.layout.tilde_mask(int(mask)) in self.index_of for mask in self.masks
        )


def _mask(indices: Sequence[int]) -> int:
    mask = 0
    for index in indices:
        mask |= 1 << index
    return mask


def ness_basis(n: int, max_ph_rank: int, layout: ModeLayout | None = None) -> CIBasis:
    """Enumerate charge-zero NESS determinants through CIS/CISD/CISDT rank."""

    mode_layout = canonical_mode_layout(n) if layout is None else layout
    if mode_layout.n != n:
        raise ValueError("layout n does not match requested n")
    spec = ExcitationSpec(max_ph_rank=max_ph_rank, sector="ness")
    records: list[tuple[int, int]] = []
    up_plus = range(0, n)
    up_minus = range(n, 2 * n)
    down_plus = range(2 * n, 3 * n) if mode_layout.symmetry == "spin" else range(0)
    down_minus = range(3 * n, 4 * n) if mode_layout.symmetry == "spin" else range(0)
    for rank in range(max_ph_rank + 1):
        for a in (range(rank + 1) if mode_layout.symmetry == "spin" else (rank,)):
            b = rank - a
            if a > n or b > n:
                continue
            for up_p in combinations(up_plus, a):
                for up_m in combinations(up_minus, a):
                    for down_p in combinations(down_plus, b):
                        for down_m in combinations(down_minus, b):
                            records.append((rank, _mask(up_p + up_m + down_p + down_m)))
    records.sort(key=lambda item: (item[0], item[1]))
    masks = tuple(mask for _, mask in records)
    ranks = tuple(rank for rank, _ in records)
    charges = np.array([mode_layout.charge(int(mask)) for mask in masks], dtype=np.int8)
    basis = CIBasis(mode_layout, spec, masks, ranks, charges)
    if not basis.is_tilde_closed():
        raise ValueError("generated NESS basis is not closed under tilde")
    return basis


def response_basis(
    n: int,
    max_ph_rank: int,
    response_spin: int = 0,
    layout: ModeLayout | None = None,
) -> CIBasis:
    """Enumerate ``+/-1`` odd determinants for one spin or total charge."""

    mode_layout = canonical_mode_layout(n) if layout is None else layout
    if mode_layout.n != n:
        raise ValueError("layout n does not match requested n")
    spec = ExcitationSpec(
        max_ph_rank=max_ph_rank, sector="response", response_spin=response_spin
    )
    if mode_layout.symmetry == "charge" and response_spin != 0:
        raise ValueError("charge symmetry requires response_spin=0")
    other_spin = 1 - response_spin
    bases = (0, 2 * n)
    response_base = bases[response_spin]
    other_base = bases[other_spin]
    response_plus = range(response_base, response_base + n)
    response_minus = range(response_base + n, response_base + 2 * n)
    other_plus = (
        range(other_base, other_base + n) if mode_layout.symmetry == "spin" else range(0)
    )
    other_minus = (
        range(other_base + n, other_base + 2 * n)
        if mode_layout.symmetry == "spin" else range(0)
    )
    records: list[tuple[int, tuple[int, int], int]] = []
    for rank in range(max_ph_rank + 1):
        for a in (range(rank + 1) if mode_layout.symmetry == "spin" else (rank,)):
            b = rank - a
            if a >= n or b > n:
                continue
            for sign in (-1, 1):
                num_plus = a + 1 if sign == 1 else a
                num_minus = a if sign == 1 else a + 1
                for response_p in combinations(response_plus, num_plus):
                    for response_m in combinations(response_minus, num_minus):
                        for other_p in combinations(other_plus, b):
                            for other_m in combinations(other_minus, b):
                                mask = _mask(
                                    response_p + response_m + other_p + other_m
                                )
                                records.append((rank, mode_layout.charge(mask), mask))
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    masks = tuple(mask for _, _, mask in records)
    ranks = tuple(rank for rank, _, _ in records)
    charges = np.array([charge for _, charge, _ in records], dtype=np.int8)
    basis = CIBasis(mode_layout, spec, masks, ranks, charges)
    if not basis.is_tilde_closed():
        raise ValueError("generated response basis is not closed under tilde")
    return basis


def right_determinant(
    mask: int, creators: Sequence[ComplexMatrix], vacuum: ArrayLike
) -> ComplexVector:
    """Return canonically ordered ``bar_P`` determinant acting on the vacuum."""

    state = np.asarray(vacuum, dtype=np.complex128)
    occupied = [p for p in range(len(creators)) if mask & (1 << p)]
    for p in reversed(occupied):
        state = creators[p] @ state
    return state


def left_determinant(
    mask: int, annihilators: Sequence[ComplexMatrix], trace_bra: ArrayLike
) -> ComplexVector:
    """Return the algebraic dual ``<I| P_pr ... P_p1`` as a row array."""

    left = np.asarray(trace_bra, dtype=np.complex128)
    occupied = [p for p in range(len(annihilators)) if mask & (1 << p)]
    for p in reversed(occupied):
        left = left @ annihilators[p]
    return left


def projected_monomial_at_end(
    left_vectors: Sequence[ComplexVector],
    operators: Sequence[ComplexMatrix],
    right_vectors: Sequence[ComplexVector],
) -> ComplexMatrix:
    """Apply a complete algebraic operator product before any CI projection."""

    projected = np.empty((len(left_vectors), len(right_vectors)), dtype=np.complex128)
    for column, right in enumerate(right_vectors):
        result = np.asarray(right, dtype=np.complex128)
        for operator in reversed(operators):
            result = operator @ result
        for row, left in enumerate(left_vectors):
            projected[row, column] = left @ result
    return projected


def apply_mode_monomial(
    mask: int,
    operators: Sequence[tuple[str, int]],
    num_modes: int,
) -> tuple[int | None, complex]:
    """Apply a complete canonical ``P/bar_P`` string by independent bit CAR.

    Operators are supplied in algebraic left-to-right order with kind
    ``"annihilate"`` for ``P`` and ``"create"`` for ``bar_P``. No basis
    projection occurs between factors.
    """

    if not 0 <= mask < (1 << num_modes):
        raise ValueError("mask is outside the mode space")
    current = mask
    coefficient = 1.0 + 0.0j
    for kind, p in reversed(operators):
        if not 0 <= p < num_modes:
            raise ValueError("operator mode is outside the mode space")
        bit = 1 << p
        occupied = bool(current & bit)
        if kind == "annihilate":
            if not occupied:
                return None, 0.0 + 0.0j
            sign = -1.0 if (current & (bit - 1)).bit_count() % 2 else 1.0
            coefficient *= sign
            current ^= bit
        elif kind == "create":
            if occupied:
                return None, 0.0 + 0.0j
            sign = -1.0 if (current & (bit - 1)).bit_count() % 2 else 1.0
            coefficient *= sign
            current |= bit
        else:
            raise ValueError("operator kind must be 'create' or 'annihilate'")
    return current, coefficient


def projected_mode_monomial(
    basis: CIBasis, operators: Sequence[tuple[str, int]]
) -> ComplexMatrix:
    """Compile a whole mode monomial, looking up only its final mask."""

    matrix = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    for column, mask in enumerate(basis.masks):
        target, coefficient = apply_mode_monomial(
            mask, operators, basis.layout.num_modes
        )
        if target is not None and target in basis.index_of:
            matrix[basis.index_of[target], column] = coefficient
    return matrix

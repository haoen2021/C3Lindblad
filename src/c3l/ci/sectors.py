"""Mode charges, tilde closure, and dimensions of the excitation sectors."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.typing import ArrayLike, NDArray

from c3l.superfermion.normal_modes import NormalModeReference


def _immutable(array: ArrayLike, dtype: np.dtype) -> NDArray:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(
        contiguous.shape
    )
    immutable.flags.writeable = False
    return immutable


@dataclass(frozen=True, slots=True)
class ModeLayout:
    """Charge and tilde metadata for spin or total-charge PH modes.

    Modes are ordered as all ``2n`` up modes followed by all ``2n`` down
    modes. Within each spin, the first ``n`` creators carry charge ``+1`` and
    the final ``n`` carry charge ``-1``. The tilde map here is the mask-level
    involution required for tilde closure; amplitude phases are not inferred from
    mask data. Charge symmetry uses only the first ``2n`` modes, with
    total charge in the first column and a zero second column.
    """

    n: int
    mode_charges: NDArray[np.int8]
    tilde_partners: NDArray[np.int64]
    symmetry: str = "spin"

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError("n must be a positive integer")
        _validate_symmetry(self.symmetry)
        num_modes = self.num_modes
        charges = np.asarray(self.mode_charges)
        partners = np.asarray(self.tilde_partners)
        if charges.dtype.kind not in "iu":
            raise TypeError("mode_charges must have integer dtype")
        if partners.dtype.kind not in "iu":
            raise TypeError("tilde_partners must have integer dtype")
        if charges.shape != (num_modes, 2):
            raise ValueError(f"mode_charges must have shape ({num_modes}, 2)")
        if partners.shape != (num_modes,):
            raise ValueError(f"tilde_partners must have length {num_modes}")
        if np.any(partners < 0) or np.any(partners >= num_modes):
            raise ValueError("tilde partner index is outside the mode layout")
        if not np.array_equal(partners[partners], np.arange(num_modes)):
            raise ValueError("tilde_partners must be an involution")
        if np.any(partners == np.arange(num_modes)):
            raise ValueError("every PH mode must have a distinct tilde partner")
        if not np.array_equal(charges[partners], -charges):
            raise ValueError("tilde partners must carry opposite Liouville charge")
        expected = np.zeros_like(charges)
        expected[: self.n, 0] = 1
        expected[self.n : 2 * self.n, 0] = -1
        expected[2 * self.n : 3 * self.n, 1] = 1
        expected[3 * self.n :, 1] = -1
        if not np.array_equal(charges, expected):
            raise ValueError(
                "mode charges violate the frozen PH block order for the symmetry"
            )
        object.__setattr__(self, "mode_charges", _immutable(charges, np.dtype(np.int8)))
        object.__setattr__(
            self, "tilde_partners", _immutable(partners, np.dtype(np.int64))
        )

    @property
    def num_modes(self) -> int:
        return (2 if self.symmetry == "charge" else 4) * self.n

    def charge(self, mask: int) -> tuple[int, int]:
        """Return ``(Q_up,Q_down)`` or charge-only ``(Q,0)`` for a mask."""

        if not 0 <= mask < (1 << self.num_modes):
            raise ValueError("mask is outside the mode layout")
        total = np.zeros(2, dtype=np.int64)
        for p in range(self.num_modes):
            if mask & (1 << p):
                total += self.mode_charges[p]
        return int(total[0]), int(total[1])

    def tilde_mask(self, mask: int) -> int:
        """Apply the anti-linear tilde map to occupations, omitting amplitudes."""

        if not 0 <= mask < (1 << self.num_modes):
            raise ValueError("mask is outside the mode layout")
        transformed = 0
        for p in range(self.num_modes):
            if mask & (1 << p):
                transformed |= 1 << int(self.tilde_partners[p])
        return transformed

    def tilde_recanonicalization_sign(self, mask: int) -> int:
        """Return the fermionic sign from sorting mapped creators canonically."""

        mapped = [
            int(self.tilde_partners[p])
            for p in range(self.num_modes)
            if mask & (1 << p)
        ]
        inversions = sum(
            mapped[i] > mapped[j]
            for i in range(len(mapped))
            for j in range(i + 1, len(mapped))
        )
        return -1 if inversions % 2 else 1


def canonical_mode_layout(n: int, symmetry: str = "spin") -> ModeLayout:
    """Return a deterministic conjugate-pair layout without numerical modes."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    _validate_symmetry(symmetry)
    num_blocks = 1 if symmetry == "charge" else 2
    charges = np.zeros((2 * num_blocks * n, 2), dtype=np.int8)
    partners = np.empty(2 * num_blocks * n, dtype=np.int64)
    for spin in range(num_blocks):
        base = 2 * n * spin
        charges[base : base + n, spin] = 1
        charges[base + n : base + 2 * n, spin] = -1
        for k in range(n):
            partner = base + 2 * n - 1 - k
            partners[base + k] = partner
            partners[partner] = base + k
    return ModeLayout(n=n, mode_charges=charges, tilde_partners=partners, symmetry=symmetry)


def mode_layout_from_references(
    up: NormalModeReference, down: NormalModeReference | None = None
) -> ModeLayout:
    """Match conjugate modes in one full charge block or two spin blocks."""

    if down is not None and up.n_orbitals != down.n_orbitals:
        raise ValueError("up and down references must have the same n")
    from .tilde import tilde_map_from_references

    n = up.n_orbitals
    tilde_map = tilde_map_from_references(up, down)
    references = (up,) if down is None else (up, down)
    charges = np.zeros((tilde_map.num_modes, 2), dtype=np.int8)
    for spin, reference in enumerate(references):
        base = 2 * n * spin
        charges[base : base + 2 * n, spin] = reference.mode_charges
    return ModeLayout(
        n=n, mode_charges=charges, tilde_partners=tilde_map.partners,
        symmetry=tilde_map.symmetry,
    )


def ness_dimension(n: int, max_ph_rank: int, symmetry: str = "spin") -> int:
    """Exact charge-zero NESS CI dimension through total PH rank ``R``."""

    _validate_dimension_args(n, max_ph_rank)
    _validate_symmetry(symmetry)
    if symmetry == "charge":
        return sum(comb(n, rank) ** 2 for rank in range(min(n, max_ph_rank) + 1))
    return sum(
        comb(n, a) ** 2 * comb(n, b) ** 2
        for a in range(n + 1)
        for b in range(n + 1)
        if a + b <= max_ph_rank
    )


def response_dimension(n: int, max_ph_rank: int, symmetry: str = "spin") -> int:
    """Union dimension of ``+/-1`` sectors for one spin or total charge."""

    _validate_dimension_args(n, max_ph_rank)
    _validate_symmetry(symmetry)
    if symmetry == "charge":
        return 2 * sum(
            comb(n, rank + 1) * comb(n, rank)
            for rank in range(min(n - 1, max_ph_rank) + 1)
        )
    return 2 * sum(
        comb(n, a + 1) * comb(n, a) * comb(n, b) ** 2
        for a in range(n)
        for b in range(n + 1)
        if a + b <= max_ph_rank
    )


def full_ness_dimension(n: int, symmetry: str = "spin") -> int:
    _validate_dimension_args(n, 0)
    _validate_symmetry(symmetry)
    if symmetry == "charge":
        return comb(2 * n, n)
    return comb(2 * n, n) ** 2


def full_response_dimension(n: int, symmetry: str = "spin") -> int:
    _validate_dimension_args(n, 0)
    _validate_symmetry(symmetry)
    if symmetry == "charge":
        return 2 * comb(2 * n, n + 1)
    return 2 * comb(2 * n, n) * comb(2 * n, n + 1)


def _validate_symmetry(symmetry: str) -> None:
    if symmetry not in ("spin", "charge"):
        raise ValueError("symmetry must be 'spin' or 'charge'")


def _validate_dimension_args(n: int, max_ph_rank: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if (
        isinstance(max_ph_rank, bool)
        or not isinstance(max_ph_rank, int)
        or max_ph_rank < 0
    ):
        raise ValueError("max_ph_rank must be a nonnegative integer")

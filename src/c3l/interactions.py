"""Sparse antisymmetrized number-conserving two-body interactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexMatrix = NDArray[np.complex128]
TwoBodyTerm = tuple[int, int, int, int, complex]


@dataclass(frozen=True, slots=True)
class SparseTwoBodyInteraction:
    r"""Canonical nonzero entries of ``v[p,q,r,s]``.

    The Hamiltonian convention is

    ``H2 = 1/4 sum(pqrs) v[p,q,r,s] a_p^dag a_q^dag a_s a_r``.

    Stored entries use ``p < q`` and ``r < s``; their coefficient is therefore
    directly the coefficient of ``a_p^dag a_q^dag a_s a_r``.  Inputs may use
    either order inside an antisymmetric pair and are canonicalized with the
    corresponding fermionic sign.

    This representation includes density couplings, exchange and pair hopping;
    it is not tied to a particular impurity model. It enforces Hermiticity and
    particle-number conservation but does not itself impose spin-projection
    conservation. Component CC/EOM solvers impose that additional restriction.
    """

    n_spin_orbitals: int
    terms: tuple[TwoBodyTerm, ...]

    def __post_init__(self) -> None:
        n = self.n_spin_orbitals
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n_spin_orbitals must be a positive integer")
        if not isinstance(self.terms, tuple):
            raise TypeError("terms must be a tuple")

        combined: dict[tuple[int, int, int, int], complex] = {}
        for raw in self.terms:
            if not isinstance(raw, tuple) or len(raw) != 5:
                raise TypeError("each two-body term must be a five-element tuple")
            p, q, r, s, raw_value = raw
            indices = (p, q, r, s)
            if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
                raise TypeError("two-body indices must be integers")
            if any(index < 0 or index >= n for index in indices):
                raise ValueError("a two-body index lies outside the spin-orbital space")
            if p == q or r == s:
                if complex(raw_value) != 0.0:
                    raise ValueError("antisymmetric index pairs must contain distinct orbitals")
                continue
            value = complex(raw_value)
            if not np.isfinite(value.real) or not np.isfinite(value.imag):
                raise ValueError("two-body coefficients must be finite")
            if p > q:
                p, q = q, p
                value = -value
            if r > s:
                r, s = s, r
                value = -value
            key = (p, q, r, s)
            combined[key] = combined.get(key, 0.0 + 0.0j) + value

        canonical = tuple(
            (*key, value)
            for key, value in sorted(combined.items())
            if value != 0.0
        )
        values = {term[:4]: term[4] for term in canonical}
        for (p, q, r, s), value in values.items():
            partner = values.get((r, s, p, q))
            scale = max(abs(value), abs(partner) if partner is not None else 0.0, 1.0)
            if partner is None or abs(partner - value.conjugate()) > 1.0e-12 * scale:
                raise ValueError("two-body interaction must be Hermitian")
        object.__setattr__(self, "terms", canonical)

    def coefficient(self, p: int, q: int, r: int, s: int) -> complex:
        """Return one full antisymmetrized tensor element."""

        indices = (p, q, r, s)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError("two-body indices must be integers")
        if any(index < 0 or index >= self.n_spin_orbitals for index in indices):
            raise ValueError("a two-body index lies outside the spin-orbital space")
        if p == q or r == s:
            return 0.0 + 0.0j
        sign = 1
        if p > q:
            p, q = q, p
            sign = -sign
        if r > s:
            r, s = s, r
            sign = -sign
        for a, b, c, d, value in self.terms:
            if (a, b, c, d) == (p, q, r, s):
                return sign * value
        return 0.0 + 0.0j


def density_density_interaction(
    n_sites: int,
    interaction_sites: tuple[int, ...],
    interaction_matrix: ArrayLike,
) -> SparseTwoBodyInteraction:
    r"""Convert density couplings into the general spin-orbital representation.

    The interaction is ``sum_i W_ii n_i_up n_i_down + sum_(i<j) W_ij n_i n_j``,
    with ``n_i = n_i_up + n_i_down``. Matrix indices follow ``interaction_sites``;
    ``n_sites`` also includes any noninteracting bath sites. The returned object
    uses exactly the same representation and solver path as exchange and
    pair-hopping inputs, rather than defining a separate density-only solver.
    """

    matrix = np.asarray(interaction_matrix, dtype=np.float64)
    if matrix.shape != (len(interaction_sites), len(interaction_sites)):
        raise ValueError("interaction_matrix does not match interaction_sites")
    terms: list[TwoBodyTerm] = []
    for a, site_i in enumerate(interaction_sites):
        up = 2 * site_i
        down = up + 1
        if matrix[a, a] != 0.0:
            terms.append((up, down, up, down, complex(matrix[a, a])))
        for b in range(a + 1, len(interaction_sites)):
            site_j = interaction_sites[b]
            value = complex(matrix[a, b])
            if value == 0.0:
                continue
            for spin_i in (0, 1):
                for spin_j in (0, 1):
                    p = 2 * site_i + spin_i
                    q = 2 * site_j + spin_j
                    terms.append((p, q, p, q, value))
    return SparseTwoBodyInteraction(2 * n_sites, tuple(terms))

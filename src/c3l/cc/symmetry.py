"""Tilde-constrained complex coupled-cluster amplitudes.

If ``tilde(tau_mu) = eta_mu tau_partner(mu)``, invariance of ``T`` requires

``t_partner(mu) = eta_mu conjugate(t_mu)``.

One paired orbit therefore has two independent real variables and a self
orbit has one.  Partners are generated algebraically; they are never passed
to the nonlinear solver as separate unknowns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TildeMap:
    """Anti-linear amplitude-orbit map with a real independent coordinate."""

    partner: tuple[int, ...]
    phase: tuple[complex, ...]
    independent: tuple[int, ...]

    def __post_init__(self) -> None:
        partner = tuple(int(j) for j in self.partner)
        phase = tuple(complex(value) for value in self.phase)
        independent = tuple(int(i) for i in self.independent)
        n = len(partner)
        if len(phase) != n:
            raise ValueError("tilde partner and phase arrays must be parallel")
        if any(j < 0 or j >= n for j in partner):
            raise ValueError("tilde partner is outside the amplitude basis")
        if any(partner[partner[i]] != i for i in range(n)):
            raise ValueError("tilde partners must form an involution")
        if any(not np.isfinite(value.real) or not np.isfinite(value.imag) or value == 0 for value in phase):
            raise ValueError("tilde phases must be finite and nonzero")
        for i, j in enumerate(partner):
            if abs(phase[j] * phase[i].conjugate() - 1.0) > 2.0e-10:
                raise ValueError("tilde phases violate the double-tilde identity")
        expected = tuple(i for i, j in enumerate(partner) if i <= j)
        if independent != expected:
            raise ValueError("independent representatives must be the lowest orbit indices")
        object.__setattr__(self, "partner", partner)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "independent", independent)

    @property
    def dimension(self) -> int:
        return len(self.partner)

    @property
    def real_size(self) -> int:
        return sum(1 if self.partner[i] == i else 2 for i in self.independent)

    def _self_scale(self, i: int) -> complex:
        eta = self.phase[i]
        unit = eta / abs(eta)
        scale = complex(np.sqrt(unit))
        # Choose the square-root branch which most accurately obeys s=eta*s*.
        if abs(scale - eta * scale.conjugate()) > abs(-scale - eta * (-scale).conjugate()):
            scale = -scale
        return scale

    def unpack(self, x: np.ndarray) -> np.ndarray:
        """Generate the full complex amplitude vector from real variables."""

        values = np.asarray(x, dtype=np.float64)
        if values.shape != (self.real_size,) or not np.all(np.isfinite(values)):
            raise ValueError("independent tilde variables have the wrong shape")
        t = np.zeros(self.dimension, dtype=np.complex128)
        cursor = 0
        for i in self.independent:
            j = self.partner[i]
            if j == i:
                t[i] = self._self_scale(i) * values[cursor]
                cursor += 1
            else:
                value = values[cursor] + 1j * values[cursor + 1]
                t[i] = value
                t[j] = self.phase[i] * value.conjugate()
                cursor += 2
        return t

    def pack(self, amplitudes: np.ndarray, *, check: bool = True) -> np.ndarray:
        """Return real orbit coordinates of a tilde-invariant complex vector."""

        t = np.asarray(amplitudes, dtype=np.complex128)
        if t.shape != (self.dimension,) or not np.all(np.isfinite(t)):
            raise ValueError("amplitudes have the wrong shape")
        if check and self.defect(t) > 2.0e-10:
            raise ValueError("amplitudes do not satisfy the tilde constraint")
        x = np.empty(self.real_size, dtype=np.float64)
        cursor = 0
        for i in self.independent:
            j = self.partner[i]
            if j == i:
                coordinate = t[i] / self._self_scale(i)
                x[cursor] = coordinate.real
                cursor += 1
            else:
                x[cursor] = t[i].real
                x[cursor + 1] = t[i].imag
                cursor += 2
        return x

    def defect(self, amplitudes: np.ndarray) -> float:
        """Maximum absolute violation of ``t_j=eta_i*t_i.conj()``."""

        t = np.asarray(amplitudes, dtype=np.complex128)
        if t.shape != (self.dimension,):
            raise ValueError("amplitudes have the wrong shape")
        if not len(t):
            return 0.0
        return float(
            max(
                abs(t[j] - self.phase[i] * t[i].conjugate())
                for i, j in enumerate(self.partner)
            )
        )

    def amplitudes(self, z: np.ndarray) -> np.ndarray:
        """Compatibility helper accepting one complex value per orbit.

        Paired-orbit representatives retain both components.  A self orbit
        has only one real degree of freedom, so the real part is used as its
        coordinate in the invariant phase line.
        """

        values = np.asarray(z, dtype=np.complex128)
        if values.shape != (len(self.independent),):
            raise ValueError("independent amplitudes have the wrong shape")
        x = np.empty(self.real_size, dtype=np.float64)
        cursor = 0
        for value, i in zip(values, self.independent, strict=True):
            if self.partner[i] == i:
                x[cursor] = value.real
                cursor += 1
            else:
                x[cursor : cursor + 2] = (value.real, value.imag)
                cursor += 2
        return self.unpack(x)


TildeAmplitudeSpace = TildeMap


def tilde_map(
    masks: tuple[int, ...],
    mode_partner: tuple[int, ...],
    mode_phase: tuple[complex, ...],
) -> TildeMap:
    """Induce the determinant tilde map in canonical creation-string order.

    Tilde preserves the written product order.  The mapped modes are then
    sorted, and the parity of that sorting supplies the fermionic sign.
    """

    if len(mode_partner) != len(mode_phase):
        raise ValueError("mode tilde partner and phase arrays must be parallel")
    if any(mode_partner[mode_partner[p]] != p for p in range(len(mode_partner))):
        raise ValueError("mode tilde partners must form an involution")
    index = {mask: i for i, mask in enumerate(masks)}
    if len(index) != len(masks):
        raise ValueError("cluster masks must be unique")
    partner: list[int] = []
    phase: list[complex] = []
    for mask in masks:
        modes = [p for p in range(len(mode_partner)) if mask >> p & 1]
        mapped = [mode_partner[p] for p in modes]
        target = sum(1 << p for p in mapped)
        if target not in index:
            raise ValueError("cluster basis is not tilde closed")
        inversions = sum(
            mapped[a] > mapped[b]
            for a in range(len(mapped))
            for b in range(a + 1, len(mapped))
        )
        factor = complex(np.prod([mode_phase[p] for p in modes]))
        if inversions % 2:
            factor = -factor
        partner.append(index[target])
        phase.append(factor)
    independent = tuple(i for i, j in enumerate(partner) if i <= j)
    return TildeMap(tuple(partner), tuple(phase), independent)


def tilde_amplitude_space(
    masks: tuple[int, ...],
    mode_partner: tuple[int, ...],
    mode_phase: tuple[complex, ...],
) -> TildeAmplitudeSpace:
    """Named constructor emphasizing the independent real CC variables."""

    return tilde_map(masks, mode_partner, mode_phase)


__all__ = ["TildeAmplitudeSpace", "TildeMap", "tilde_amplitude_space", "tilde_map"]

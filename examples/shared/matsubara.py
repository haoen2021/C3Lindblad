"""Positive-Matsubara to imaginary-time transform for the Kanamori benchmark."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def matsubara_to_tau(
    positive_frequency: ArrayLike,
    positive_green: ArrayLike,
    beta: float,
    tau: ArrayLike,
) -> NDArray[np.complex128]:
    r"""Inverse-transform positive frequencies with fixed M0 and fitted M1."""

    frequency = np.asarray(positive_frequency, dtype=np.float64)
    green = np.asarray(positive_green, dtype=np.complex128)
    times = np.asarray(tau, dtype=np.float64)
    inverse_temperature = float(beta)
    if (
        frequency.ndim != 1
        or frequency.size == 0
        or np.any(frequency <= 0.0)
        or np.any(np.diff(frequency) <= 0.0)
        or green.ndim != 3
        or green.shape[0] != frequency.size
        or green.shape[1] != green.shape[2]
        or times.ndim != 1
        or np.any(times < 0.0)
        or np.any(times > inverse_temperature)
        or not np.all(np.isfinite(frequency))
        or not np.all(np.isfinite(green))
        or not np.all(np.isfinite(times))
        or inverse_temperature <= 0.0
    ):
        raise ValueError("Matsubara-transform inputs are invalid")
    z = 1j * frequency
    identity = np.eye(green.shape[1], dtype=np.complex128)
    start = max(green.shape[0] // 2, 1)
    local_z = z[start:]
    powers = tuple(range(2, 7))
    design = np.column_stack([1.0 / local_z**power for power in powers])
    target = (green[start:] - identity[None, :, :] / local_z[:, None, None]).reshape(
        local_z.size, -1
    )
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    raw_moment = coefficients[0].reshape(identity.shape)
    moment = 0.5 * (raw_moment + raw_moment.conj().T)
    reference_inverse = z[:, None, None] * identity[None, :, :] - moment[None, :, :]
    reference_iw = np.linalg.solve(
        reference_inverse, np.broadcast_to(identity, reference_inverse.shape)
    )
    remainder = green - reference_iw
    phase = np.exp(-np.outer(times, z))
    positive = np.einsum("tn,nij->tij", phase, remainder, optimize=True)
    negative = np.einsum(
        "tn,nji->tij",
        phase.conj(),
        remainder.conj(),
        optimize=True,
    )
    energies, vectors = np.linalg.eigh(moment)
    scalar = np.empty((times.size, energies.size), dtype=np.float64)
    for index, energy in enumerate(energies):
        if energy >= 0.0:
            scalar[:, index] = -np.exp(-energy * times) / (
                1.0 + np.exp(-inverse_temperature * energy)
            )
        else:
            scalar[:, index] = -np.exp(energy * (inverse_temperature - times)) / (
                1.0 + np.exp(inverse_temperature * energy)
            )
    reference_tau = np.einsum(
        "ia,ta,ja->tij", vectors, scalar, vectors.conj(), optimize=True
    )
    return np.asarray(
        reference_tau + (positive + negative) / inverse_temperature,
        dtype=np.complex128,
    )

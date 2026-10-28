"""Quadratic 2n by 2n normal-mode matrix used by the production HF reference."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]


def quadratic_mode_matrix(
    E: ComplexMatrix, gamma_loss: ComplexMatrix, gamma_gain: ComplexMatrix
) -> ComplexMatrix:
    """Return the paper normal-mode matrix ``h`` with unmodified block indices."""

    Lambda = gamma_loss + gamma_gain
    Omega = gamma_gain - gamma_loss
    return np.block(
        [
            [E + 1j * Omega, 2.0 * gamma_gain],
            [-2.0 * gamma_loss, E - 1j * Omega],
        ]
    ).astype(np.complex128, copy=False)

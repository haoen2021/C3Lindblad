"""Quadratic Lindblad steady covariance in the physical orbital basis."""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]


def _matrix(value: ArrayLike, name: str) -> ComplexArray:
    a = np.asarray(value, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite square matrix")
    return a


def finite_pm_covariance(
    E: ArrayLike, gamma_loss: ArrayLike, gamma_gain: ArrayLike
) -> tuple[ComplexArray, float]:
    r"""Solve the quadratic NESS covariance and return its residual.

    With ``gamma[p,q]=<a_q^dagger a_p>`` and

    .. math::

       A=-iE-\Gamma_{\rm loss}-\Gamma_{\rm gain},

    the covariance satisfies

    .. math::

       A\gamma+\gamma A^\dagger+2\Gamma_{\rm gain}=0.
    """

    h = _matrix(E, "E")
    loss = _matrix(gamma_loss, "gamma_loss")
    gain = _matrix(gamma_gain, "gamma_gain")
    if h.shape != loss.shape or h.shape != gain.shape:
        raise ValueError("E, gamma_loss, and gamma_gain must have the same shape")
    A = -1j * h - loss - gain
    gamma = scipy.linalg.solve_sylvester(A, A.conj().T, -2.0 * gain)
    R = A @ gamma + gamma @ A.conj().T + 2.0 * gain
    scale = max(
        np.linalg.norm(A, ord=2) * np.linalg.norm(gamma, ord="fro"),
        np.linalg.norm(2.0 * gain, ord="fro"),
        1.0,
    )
    return np.asarray(gamma), float(np.linalg.norm(R, ord="fro") / scale)


__all__ = ["finite_pm_covariance"]

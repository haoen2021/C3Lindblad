"""Prescribed semicircular and smoothed flat-band benchmark baths."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

ComplexVector = NDArray[np.complex128]


def semicircle_bcf(
    t: ArrayLike,
    *,
    lesser: bool,
    W: float,
    Gamma_bath: float,
    beta: float,
    mu: float = 0.0,
    quadrature_order: int = 2048,
) -> ComplexVector:
    r"""Evaluate a semicircular lesser or greater BCF.

    The convention is

    ``Delta(t) = integral J(w) occupation(w) exp(-1j*w*t) dw``

    with ``occupation=f`` for the lesser BCF and ``occupation=1-f`` for the
    greater BCF.  Gauss--Legendre quadrature is restricted exactly to the
    compact support ``[-W,W]``.
    """

    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("t must be a finite nonempty one-dimensional grid")
    if W <= 0.0 or Gamma_bath <= 0.0 or beta <= 0.0:
        raise ValueError("W, Gamma_bath, and beta must be positive")
    if quadrature_order < 64:
        raise ValueError("quadrature_order must be at least 64")
    nodes, weights = leggauss(quadrature_order)
    omega = W * nodes
    measure = W * weights
    J = Gamma_bath / np.pi * np.sqrt(np.maximum(0.0, 1.0 - nodes**2))
    f = expit(-beta * (omega - mu))
    occupation = f if lesser else 1.0 - f
    return np.asarray(
        (measure * J * occupation) @ np.exp(-1j * np.outer(omega, times)),
        dtype=np.complex128,
    )

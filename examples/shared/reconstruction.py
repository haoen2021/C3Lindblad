"""Compact-response reconstruction for the paper multiorbital cases."""

from __future__ import annotations
import numpy as np
import scipy.linalg
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def projected_transfer(
    hessenberg: ComplexArray,
    sink_projection: ComplexArray,
    source_norm: float,
    shifts: ComplexArray,
    dimension: int,
) -> ComplexArray:
    """Evaluate a saved one-sided Arnoldi transfer on a new shift grid."""

    matrix = np.asarray(hessenberg, dtype=np.complex128)
    sinks = np.asarray(sink_projection, dtype=np.complex128)
    z = np.asarray(shifts, dtype=np.complex128)
    m = int(dimension)
    if (
        m <= 0
        or matrix.shape[0] < m + 1
        or matrix.shape[1] < m
        or sinks.ndim != 2
        or sinks.shape[1] < m
        or z.ndim != 1
        or not np.all(np.isfinite(matrix[: m + 1, :m]))
        or not np.all(np.isfinite(sinks[:, :m]))
        or not np.all(np.isfinite(z))
    ):
        raise ValueError("saved projected-response inputs are invalid")
    reduced = matrix[:m, :m]
    rhs = np.zeros(m, dtype=np.complex128)
    rhs[0] = float(source_norm)
    triangular, vectors = scipy.linalg.schur(
        reduced, output="complex", check_finite=False
    )
    reduced_rhs = vectors.conj().T @ rhs
    identity = np.eye(m, dtype=np.complex128)
    coefficients = np.empty((z.size, m), dtype=np.complex128)
    for index, shift in enumerate(z):
        local = scipy.linalg.solve_triangular(
            shift * identity - triangular,
            reduced_rhs,
            lower=False,
            check_finite=False,
        )
        coefficients[index] = vectors @ local
    return np.asarray(coefficients @ sinks[:, :m].T, dtype=np.complex128)


def orbital_total(spectrum: ComplexArray) -> RealArray:
    """Return the per-spin orbital trace used in all reported spectra."""

    values = np.asarray(spectrum, dtype=np.complex128)
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("spectral matrix must have shape (frequency,2,2)")
    return np.asarray(np.trace(values, axis1=1, axis2=2).real, dtype=np.float64)


def matsubara_components(values: ComplexArray) -> tuple[RealArray, RealArray]:
    """Return ``G00`` and the reciprocity-symmetrized ``G01`` curve."""

    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.ndim != 3 or matrix.shape[1:] != (2, 2):
        raise ValueError("Matsubara display matrix must have shape (time,2,2)")
    return (
        np.asarray(matrix[:, 0, 0].real, dtype=np.float64),
        np.asarray(
            0.5 * (matrix[:, 0, 1] + matrix[:, 1, 0]).real,
            dtype=np.float64,
        ),
    )

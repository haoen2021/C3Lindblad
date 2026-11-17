r"""Fermionic Matsubara grids: explicit index/frequency sets and DLR nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexGrid = NDArray[np.complex128]

_MATSUBARA_TOLERANCE = 1.0e-8


@dataclass(frozen=True, slots=True, eq=False)
class MatsubaraGrid:
    r"""Fermionic Matsubara points :math:`i\omega_n`, :math:`\omega_n=(2n+1)\pi/\beta`.

    ``indices`` are the integers ``n`` in the caller's order, which is kept in
    every array evaluated on the grid.  ``upper_shifts`` are the distinct
    points :math:`z=i|\omega_n|` at which a retarded solver must be evaluated;
    :meth:`from_upper` maps those values back onto the grid.
    """

    beta: float
    indices: NDArray[np.int64]
    kind: str = "explicit"
    eps: float | None = None
    lamb: float | None = None
    dlr: Any = field(default=None, repr=False)
    _upper_odd: NDArray[np.int64] = field(init=False, repr=False)
    _upper_index: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        beta = float(self.beta)
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be finite and positive")
        raw = np.asarray(self.indices)
        if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("indices must be a nonempty integer vector")
        indices = raw.astype(np.int64)
        if np.unique(indices).size != indices.size:
            raise ValueError("Matsubara indices must be unique")
        if self.kind not in ("explicit", "dlr"):
            raise ValueError("kind must be 'explicit' or 'dlr'")
        indices.setflags(write=False)
        odd, inverse = np.unique(np.abs(2 * indices + 1), return_inverse=True)
        odd.setflags(write=False)
        inverse = inverse.astype(np.int64)
        inverse.setflags(write=False)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "_upper_odd", odd)
        object.__setattr__(self, "_upper_index", inverse)

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def omega_n(self) -> NDArray[np.float64]:
        """Real Matsubara frequencies :math:`\\omega_n` in grid order."""

        return (2 * self.indices + 1) * np.pi / self.beta

    @property
    def iw(self) -> ComplexGrid:
        """Complex Matsubara points :math:`i\\omega_n` in grid order."""

        return 1j * self.omega_n

    @property
    def upper_shifts(self) -> ComplexGrid:
        r"""Distinct upper-half-plane points :math:`i|\omega_n|`, increasing."""

        return 1j * self._upper_odd * np.pi / self.beta

    def from_upper(self, values: ArrayLike) -> ComplexGrid:
        r"""Map retarded values at :attr:`upper_shifts` onto the grid.

        ``values`` has shape ``(len(upper_shifts),)`` for a scalar function
        or ``(len(upper_shifts), m, m)`` for a matrix.  Negative frequencies
        receive the conjugate transpose,
        :math:`G(-i\omega)=G(i\omega)^\dagger`.
        """

        upper = np.asarray(values, dtype=np.complex128)
        if upper.ndim not in (1, 3) or upper.shape[0] != self._upper_odd.size:
            raise ValueError("values must have shape (len(upper_shifts),) or (len(upper_shifts), m, m)")
        if upper.ndim == 3 and upper.shape[1] != upper.shape[2]:
            raise ValueError("matrix values must be square")
        out = upper[self._upper_index].copy()
        negative = self.indices < 0
        if upper.ndim == 1:
            out[negative] = out[negative].conj()
        else:
            out[negative] = out[negative].conj().swapaxes(1, 2)
        return out

    @property
    def dlr_poles(self) -> NDArray[np.float64]:
        r"""Real DLR frequencies :math:`\omega_l` (requires a DLR grid).

        Values on the grid are expanded as
        :math:`G(i\omega_n)=\sum_l g_l/(i\omega_n-\omega_l)`; the number of
        poles equals the number of Matsubara nodes.
        """

        if self.kind != "dlr" or self.dlr is None:
            raise ValueError("DLR transforms need a grid from dlr_matsubara_grid")
        poles = np.asarray(self.dlr.dlrrf, dtype=np.float64).reshape(-1) / self.beta
        if poles.size != self.indices.size:
            raise RuntimeError("DLR rank and Matsubara node count differ")
        return poles

    def dlr_coefficients(self, values: ArrayLike) -> ComplexGrid:
        r"""DLR coefficients :math:`g_l` of ``values`` given on this DLR grid.

        ``values`` has shape ``(len(grid), ...)`` (scalar or matrix valued);
        the result has the same shape, indexed by pole instead of frequency.
        """

        poles = self.dlr_poles
        data = np.asarray(values, dtype=np.complex128)
        if data.ndim < 1 or data.shape[0] != len(self) or not np.all(np.isfinite(data)):
            raise ValueError("values must be finite with leading dimension len(grid)")
        kernel = 1.0 / (self.iw[:, None] - poles[None, :])
        coefficients = np.linalg.solve(kernel, data.reshape(len(self), -1))
        return coefficients.reshape(data.shape)

    def dlr_tau(self, values: ArrayLike, tau: ArrayLike) -> ComplexGrid:
        r"""Evaluate :math:`G(\tau)=-\sum_l g_l e^{-\omega_l\tau}/(1+e^{-\beta\omega_l})`.

        ``values`` are on this DLR grid; ``tau`` lies in :math:`[0,\beta]`.
        Returns shape ``(len(tau), ...)``.
        """

        coefficients = self.dlr_coefficients(values)
        times = np.asarray(tau, dtype=np.float64)
        if (
            times.ndim != 1
            or not np.all(np.isfinite(times))
            or np.any(times < 0.0)
            or np.any(times > self.beta)
        ):
            raise ValueError("tau must be a finite vector in [0, beta]")
        poles = self.dlr_poles
        magnitude = np.abs(poles)[None, :]
        t = times[:, None]
        decay = np.where(poles[None, :] >= 0.0, np.exp(-magnitude * t), np.exp(-magnitude * (self.beta - t)))
        kernel = -decay / (1.0 + np.exp(-self.beta * magnitude))
        return np.tensordot(kernel, coefficients, axes=(1, 0))

    def dlr_matsubara(self, values: ArrayLike, points: ArrayLike) -> ComplexGrid:
        r"""Evaluate the DLR expansion of ``values`` at Matsubara points :math:`i\omega`.

        ``points`` are complex :math:`i\omega` (e.g. ``other_grid.iw``).
        Returns shape ``(len(points), ...)``.
        """

        coefficients = self.dlr_coefficients(values)
        z = np.asarray(points, dtype=np.complex128)
        if z.ndim != 1 or not np.all(np.isfinite(z)):
            raise ValueError("points must be a finite complex vector")
        kernel = 1.0 / (z[:, None] - self.dlr_poles[None, :])
        return np.tensordot(kernel, coefficients, axes=(1, 0))


def matsubara_grid(
    beta: float,
    *,
    indices: ArrayLike | None = None,
    omega_n: ArrayLike | None = None,
    n_iw: int | None = None,
) -> MatsubaraGrid:
    r"""Build an explicit fermionic Matsubara grid.

    Give exactly one of ``indices`` (integers ``n``), ``omega_n`` (real
    frequencies, or the purely imaginary points :math:`i\omega_n`), or
    ``n_iw`` (the symmetric window ``n = -n_iw, ..., n_iw - 1``, as in TRIQS
    ``MeshImFreq``).  Frequencies must lie on the fermionic lattice of this
    ``beta`` to relative accuracy ``1e-8``.
    """

    given = [value is not None for value in (indices, omega_n, n_iw)]
    if sum(given) != 1:
        raise ValueError("give exactly one of indices, omega_n, or n_iw")
    inverse_temperature = float(beta)
    if not np.isfinite(inverse_temperature) or inverse_temperature <= 0.0:
        raise ValueError("beta must be finite and positive")
    if n_iw is not None:
        if isinstance(n_iw, bool) or not isinstance(n_iw, (int, np.integer)) or n_iw <= 0:
            raise ValueError("n_iw must be a positive integer")
        return MatsubaraGrid(inverse_temperature, np.arange(-int(n_iw), int(n_iw)))
    if indices is not None:
        return MatsubaraGrid(inverse_temperature, np.asarray(indices))
    frequency = np.asarray(omega_n)
    if np.iscomplexobj(frequency):
        if np.any(np.abs(frequency.real) > _MATSUBARA_TOLERANCE * np.maximum(np.abs(frequency), 1.0)):
            raise ValueError("complex Matsubara points must be purely imaginary")
        frequency = frequency.imag
    frequency = np.asarray(frequency, dtype=np.float64)
    if frequency.ndim != 1 or frequency.size == 0 or not np.all(np.isfinite(frequency)):
        raise ValueError("omega_n must be a finite nonempty vector")
    odd = frequency * inverse_temperature / np.pi
    nearest = np.rint(odd)
    if (
        np.any(np.abs(odd - nearest) > _MATSUBARA_TOLERANCE * np.maximum(np.abs(odd), 1.0))
        or np.any(nearest.astype(np.int64) % 2 == 0)
    ):
        raise ValueError("omega_n are not fermionic Matsubara frequencies for this beta")
    return MatsubaraGrid(inverse_temperature, (nearest.astype(np.int64) - 1) // 2)


def dlr_matsubara_grid(
    beta: float,
    eps: float,
    *,
    lamb: float | None = None,
    wmax: float | None = None,
) -> MatsubaraGrid:
    r"""Fermionic DLR Matsubara nodes for ``beta``, accuracy ``eps`` and cutoff.

    The DLR needs a dimensionless cutoff :math:`\Lambda=\beta\omega_{\max}`,
    where :math:`\omega_{\max}` bounds the support of the impurity spectral
    function (bandwidth plus interaction scale).  Give either ``lamb`` or
    ``wmax``.  Nodes keep ``pydlr``'s order so ``grid.dlr`` transforms
    (``dlr_from_matsubara``, ``tau_from_dlr``, ...) apply directly to arrays
    evaluated on the grid.
    """

    if (lamb is None) == (wmax is None):
        raise ValueError("give exactly one of lamb or wmax")
    inverse_temperature = float(beta)
    if not np.isfinite(inverse_temperature) or inverse_temperature <= 0.0:
        raise ValueError("beta must be finite and positive")
    cutoff = float(lamb) if lamb is not None else inverse_temperature * float(wmax)
    accuracy = float(eps)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("lamb (or beta * wmax) must be finite and positive")
    if not np.isfinite(accuracy) or not 0.0 < accuracy < 1.0:
        raise ValueError("eps must lie in (0, 1)")
    try:
        from pydlr import dlr
    except ImportError as exc:  # pragma: no cover - exercised only without pydlr
        raise ImportError(
            'DLR grids require pydlr; install it with pip install "c3l[dlr]"'
        ) from exc
    basis = dlr(lamb=cutoff, eps=accuracy, xi=-1)
    return MatsubaraGrid(
        inverse_temperature,
        np.asarray(basis.dlrmf, dtype=np.int64).reshape(-1),
        kind="dlr",
        eps=accuracy,
        lamb=cutoff,
        dlr=basis,
    )


__all__ = ["ComplexGrid", "MatsubaraGrid", "dlr_matsubara_grid", "matsubara_grid"]

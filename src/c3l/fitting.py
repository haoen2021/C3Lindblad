"""RealTimeBath adapter for physical lesser/greater pseudomode fitting."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .pseudomode import PseudomodeRealization


@dataclass(frozen=True, slots=True)
class FermionicBathFit:
    """Physical realizations and unmodified backend diagnostics for both branches."""

    lesser: PseudomodeRealization
    greater: PseudomodeRealization
    diagnostics: dict[str, Any]
    convention: str
    backend: str = "realtimebath"


def _realization(model, branch: str, n_modes: int) -> PseudomodeRealization:
    hamiltonian = np.array(model.hamiltonian, dtype=np.complex128, copy=True)
    damping = np.array(model.damping, dtype=np.complex128, copy=True)
    coupling = np.array(model.coupling, dtype=np.complex128, copy=True)
    if hamiltonian.shape != (n_modes, n_modes) or damping.shape != hamiltonian.shape or coupling.shape != (n_modes,):
        raise ValueError("fitted realization does not preserve the requested mode count")
    if not all(np.all(np.isfinite(value)) for value in (hamiltonian, damping, coupling)):
        raise ValueError("fitted realization contains nonfinite values")
    for name, matrix in (("Hamiltonian", hamiltonian), ("damping", damping)):
        tolerance = 1e-10 * max(1.0, float(np.linalg.norm(matrix)))
        if np.linalg.norm(matrix - matrix.conj().T) > tolerance:
            raise ValueError(f"fitted {name} is not Hermitian")
    if np.linalg.eigvalsh(damping).min() < -1e-10 * max(1.0, float(np.linalg.norm(damping))):
        raise ValueError("fitted damping is not positive semidefinite")
    if branch == "lesser":
        hamiltonian *= -1
    for value in (hamiltonian, damping, coupling):
        value.flags.writeable = False
    return PseudomodeRealization(hamiltonian, damping, coupling)


def fit_fermionic_bath(
    time: ArrayLike,
    lesser: ArrayLike,
    greater: ArrayLike,
    *,
    n_modes: int,
    convention: str = "correlation",
    **fit_options: Any,
) -> FermionicBathFit:
    """Fit scalar branch kernels; use matrix_pseudomode_model for embedding.

    Correlation inputs are positive kernels. NEGF inputs obey
    Delta_lesser=+i*C_lesser and Delta_greater=-i*C_greater.
    The time grid must start at zero and increase uniformly.
    """

    times = np.asarray(time, dtype=np.float64)
    branches = tuple(np.asarray(value, dtype=np.complex128) for value in (lesser, greater))
    if times.ndim != 1 or times.size < 3 or not np.all(np.isfinite(times)):
        raise ValueError("time must be a finite vector with at least three samples")
    spacing = np.diff(times)
    if times[0] != 0 or np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=0):
        raise ValueError("time must start at zero and increase uniformly")
    if any(value.shape != times.shape or not np.all(np.isfinite(value)) for value in branches):
        raise ValueError("each branch must be a finite scalar kernel on the time grid")
    if isinstance(n_modes, bool) or not isinstance(n_modes, int) or n_modes < 1:
        raise ValueError("n_modes must be a positive integer")
    if convention not in ("correlation", "negf"):
        raise ValueError("convention must be 'correlation' or 'negf'")
    try:
        from realtimebath import fit_fermionic
    except ImportError as error:
        raise ImportError("Bath fitting requires RealTimeBath; install c3l[fitting]. No old fitter is used as a fallback.") from error
    options = dict(method="sdp", optimize=True, solver="CLARABEL")
    options.update(fit_options)
    fitted = fit_fermionic(times, *branches, convention=convention, n_modes=n_modes, **options)
    return FermionicBathFit(
        lesser=_realization(fitted.lesser.model, "lesser", n_modes),
        greater=_realization(fitted.greater.model, "greater", n_modes),
        diagnostics={branch: asdict(getattr(fitted, branch).diagnostics) for branch in ("lesser", "greater")},
        convention=convention,
    )


__all__ = ["FermionicBathFit", "fit_fermionic_bath"]

"""Small deterministic AIM used only by exact full-Q5 tests."""

import numpy as np

from c3l import FermionicLindbladModel


def s135_validation_model(num_bath_sites: int = 2) -> FermionicLindbladModel:
    """Return the compact interacting model used by dense response oracles."""

    if num_bath_sites != 2:
        raise ValueError("the compact full-Q5 fixture uses two bath sites")
    U = 1.2
    bath = np.linspace(-1.2, 1.2, num_bath_sites)
    E = np.diag(np.concatenate(([-U / 2.0], bath))).astype(np.complex128)
    coupling = 0.28 + 0.04 * (1.0 - np.abs(bath) / np.max(np.abs(bath)))
    E[0, 1:] = coupling
    E[1:, 0] = coupling
    scaled = (bath - bath[0]) / (bath[-1] - bath[0])
    loss = np.zeros(num_bath_sites + 1)
    gain = np.zeros(num_bath_sites + 1)
    loss[1:] = 0.06 + 0.42 * scaled
    gain[1:] = 0.48 - 0.42 * scaled
    return FermionicLindbladModel(
        E,
        np.diag(loss),
        np.diag(gain),
        U=U,
        units="deterministic_full_Q5_validation",
    )


__all__ = ["s135_validation_model"]

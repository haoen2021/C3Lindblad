"""Two-orbital quadratic NESS and retarded response.

Energy is measured in Gamma and time in Gamma^{-1}; hbar=1.
"""

import numpy as np

from c3l import QuadraticOpenSystem
from c3l.gaussian_mean_field import quadratic_frequency_response
from c3l.quadratic_ness import finite_pm_covariance
from c3l.spectra import matrix_spectral_diagnostics


energy_unit = "Gamma"
E = np.array([[0.0, 0.35], [0.35, 0.4]], dtype=np.complex128)
Gamma_loss = np.diag([0.0, 0.20])
Gamma_gain = np.diag([0.0, 0.10])
model = QuadraticOpenSystem(
    E,
    Gamma_loss,
    Gamma_gain,
    units=energy_unit,
)

gamma, ness_residual = finite_pm_covariance(
    model.E,
    model.gamma_loss,
    model.gamma_gain,
)
omega = np.linspace(-1.0, 1.0, 9)
z = omega + 0.05j
response = quadratic_frequency_response(
    model.E,
    model.gamma_loss,
    model.gamma_gain,
    gamma,
    z,
)
spectral = matrix_spectral_diagnostics(response.retarded)

eigenvalues = np.linalg.eigvalsh((gamma + gamma.conj().T) / 2.0)
assert ness_residual < 1.0e-12
assert eigenvalues.min() >= -1.0e-12
assert eigenvalues.max() <= 1.0 + 1.0e-12
assert response.reconstruction_defect < 1.0e-12
assert spectral.hermiticity_defect.max() < 1.0e-12

print(f"C3L example: quadratic NESS [{energy_unit}]")
print(f"covariance residual = {ness_residual:.3e}")
print(f"occupations = {np.real(np.diag(gamma))}")

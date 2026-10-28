"""Ordered matrix Dyson reconstruction on a two-site fixture.

Energy is measured in eV and time in eV^{-1}; hbar=1.
"""

import numpy as np

from c3l.spectra import (
    matrix_dyson_green,
    matrix_ordered_dyson_reconstruction,
)


energy_unit = "eV"
omega = np.linspace(-2.0, 2.0, 41)
z = omega + 0.08j
h = np.array([[0.0, -0.45], [-0.45, 0.2]], dtype=np.complex128)

Delta_pm = np.zeros((omega.size, 2, 2), dtype=np.complex128)
Delta_physical = np.zeros_like(Delta_pm)
Delta_pm[:, 0, 0] = 0.25 / (z + 0.7j)
Delta_pm[:, 1, 1] = 0.20 / (z + 0.5j)
Delta_physical[:, 0, 0] = 0.28 / (z + 0.9j)
Delta_physical[:, 1, 1] = 0.18 / (z + 0.8j)

Sigma = np.broadcast_to(
    np.array([[0.12 - 0.04j, 0.03], [0.03, -0.08 - 0.03j]]),
    Delta_pm.shape,
).copy()
G_pm = matrix_dyson_green(z, h, Delta_pm, Sigma).retarded
reconstructed = matrix_ordered_dyson_reconstruction(
    G_pm,
    Delta_pm,
    Delta_physical,
)
direct = matrix_dyson_green(z, h, Delta_physical, Sigma).retarded

relative_error = np.linalg.norm(reconstructed.retarded - direct) / np.linalg.norm(
    direct
)
assert relative_error < 1.0e-12
assert reconstructed.ordering_difference.max() < 1.0e-12
assert reconstructed.left_ordered_residual.max() < 1.0e-12
assert reconstructed.right_ordered_residual.max() < 1.0e-12

print(f"C3L example: spectral reconstruction [{energy_unit}]")
print(f"ordered reconstruction error = {relative_error:.3e}")

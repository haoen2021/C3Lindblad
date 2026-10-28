"""Gates for bounded disk-backed Krylov basis diagnostics."""

from __future__ import annotations

import numpy as np


def test_exact_orthogonality_diagnostic_reports_known_defect() -> None:
    """Using transpose instead of adjoint or omitting off-diagonals must fail."""

    from c3l.green.krylov_diagnostics import orthogonality_diagnostics

    basis = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.25j),
            (0.0, 0.0, np.sqrt(1.0 - 0.25**2)),
            (0.0, 0.0, 0.0),
        ),
        dtype=np.complex128,
    )

    observed = orthogonality_diagnostics(
        basis,
        dimension=2,
        exact_limit=3,
        row_block_size=2,
    )

    gram = basis.conj().T @ basis
    defect = gram - np.eye(3)
    assert observed["mode"] == "exact"
    assert observed["column_count"] == 3
    np.testing.assert_allclose(
        observed["maximum_absolute_defect"], np.max(np.abs(defect)), atol=1e-15
    )
    np.testing.assert_allclose(
        observed["frobenius_defect"], np.linalg.norm(defect), atol=1e-15
    )


def test_large_basis_uses_deterministic_bounded_column_sample() -> None:
    """A diagnostic that scales to all m columns above the cap must fail."""

    from c3l.green.krylov_diagnostics import orthogonality_diagnostics

    basis = np.eye(12, dtype=np.complex128)
    first = orthogonality_diagnostics(
        basis,
        dimension=11,
        exact_limit=4,
        sample_columns=5,
        row_block_size=3,
    )
    second = orthogonality_diagnostics(
        basis,
        dimension=11,
        exact_limit=4,
        sample_columns=5,
        row_block_size=5,
    )

    assert first["mode"] == "sampled"
    assert first["column_count"] == 5
    assert first["columns"] == second["columns"]
    assert first["maximum_absolute_defect"] == 0.0

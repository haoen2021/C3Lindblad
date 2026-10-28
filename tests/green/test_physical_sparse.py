"""Small independent oracle checks retained by the clean benchmark tree."""

import numpy as np

from examples.siam_ed.physical_sparse import (
    compile_physical_sector_operator,
    physical_density_action,
    solve_trace_fixed_static,
)
from c3l import FermionicLindbladModel


def test_one_site_trace_fixed_sparse_ness() -> None:
    model = FermionicLindbladModel(
        np.zeros((1, 1)),
        np.asarray([[0.7]]),
        np.asarray([[0.3]]),
        U=0.0,
    )
    operator = compile_physical_sector_operator(
        physical_density_action(model), maximum_bytes=1 << 20
    )
    ness = solve_trace_fixed_static(operator)
    assert ness.trace_defect < 1.0e-13
    assert ness.scaled_residual < 1.0e-13
    assert ness.hermiticity_defect < 1.0e-13
    assert ness.minimum_eigenvalue >= -1.0e-13

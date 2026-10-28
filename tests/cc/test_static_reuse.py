"""Production static kernels must reuse structure when interaction values change."""

import numpy as np
import pytest

from c3l import FermionicLindbladModel
from c3l.cc.component_tensor import ComponentStaticProblem, component_static_residual, component_static_jvp
from c3l.cc.factorized import factorized_cc_problem


def test_same_structure_reuses_runtime_residual_and_jvp():
    pytest.importorskip("jax")
    from c3l.cc.component_jax import JaxStaticCCSDOperator
    from c3l.cc._static_gmres import get_cycle

    engines = []
    counts = []
    for strength, hopping, gain in ((0.0, 0.0, 0.0), (0.4, 0.07, 0.17), (-0.12, 0.1, 0.2)):
        model = FermionicLindbladModel(np.array([[0.2, hopping], [hopping, -0.15]]), np.diag([0.3, 0.32]), np.diag([gain, gain]), U=strength, interaction_sites=(0,))
        problem = ComponentStaticProblem.compile(factorized_cc_problem(model, "CCSD", build_next_rank_masks=False))
        operator = JaxStaticCCSDOperator.from_cpu(problem, platform="cpu", production=False)
        engines.append(operator.engine)
        values = problem.oracle.tilde.unpack(np.full(problem.oracle.num_variables, 1e-3))
        direction = problem.oracle.tilde.unpack(np.full(problem.oracle.num_variables, 0.03))
        np.testing.assert_allclose(operator.residual(values), component_static_residual(problem, values).retained, atol=3e-11, rtol=3e-11)
        np.testing.assert_allclose(operator.jvp(values, direction), component_static_jvp(problem, values, direction).retained, atol=3e-11, rtol=3e-11)
        cycle = get_cycle(operator.layout, 3)
        put = lambda value: operator.jax.device_put(value, operator.device)
        correction = cycle(put(values), put(np.full(operator.real_dimension, 1e-4)), put(np.zeros(operator.real_dimension)), operator.data, put(np.ones(operator.dimension, dtype=np.complex128)))
        correction.block_until_ready()
        assert np.all(np.isfinite(np.asarray(correction)))
        counts.append((operator.engine.residual._cache_size(), operator.engine.jvp._cache_size(), cycle._cache_size()))
    assert all(engine is engines[0] for engine in engines)
    assert all(count == counts[0] for count in counts)

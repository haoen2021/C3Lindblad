"""Charge-only optimized JAX Newton solves and resident S13 against physical ED."""

import numpy as np
import pytest

from c3l.cc.component_tensor import component_eom_context, component_eom_sources
from c3l.cc.newton_krylov import NewtonKrylovPolicy
from tests.cc.test_charge_s13 import exact_retarded_green, solved_charge


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
def test_charge_device_newton_and_resident_s13_match_lindblad_ed(platform):
    jax = pytest.importorskip("jax")
    try:
        devices = jax.devices(platform)
    except RuntimeError:
        if platform == "gpu":
            pytest.skip("no JAX GPU is visible")
        raise
    if not devices:
        pytest.skip(f"no JAX {platform} device is visible")
    jax.config.update("jax_enable_x64", True)

    from c3l.cc.component_jax import JaxRetainedEOMOperator, solve_component_ccsd_jax
    from c3l.green.resident_arnoldi import ResidentArnoldi

    model, static, expected_solution, context, _ = solved_charge()
    solution = solve_component_ccsd_jax(
        static,
        policy=NewtonKrylovPolicy(tolerance=1e-12, gmres_restart=5),
        platform=platform,
        production=True,
    )
    assert solution.converged
    np.testing.assert_allclose(
        solution.amplitudes, expected_solution.amplitudes, atol=1e-10, rtol=1e-10
    )
    device_context = component_eom_context(context.problem, solution.amplitudes)
    sources = tuple(
        component_eom_sources(device_context, site=orbital) for orbital in (0, 1)
    )
    operator = JaxRetainedEOMOperator.from_cpu(
        device_context, platform=platform, production=True
    )
    assert operator.device.platform == platform
    frequencies = np.array([-0.6, 0.0, 0.7])
    broadening = 0.13
    expected = exact_retarded_green(model, frequencies, broadening)
    observed = np.empty_like(expected)
    sinks = np.vstack([source.sink for source in sources])
    for column, source in enumerate(sources):
        right = source.addition + source.removal
        chain = ResidentArnoldi(
            operator.device_matvec, right, sinks, operator.dimension,
            panel=2, device=operator.device, workspace_bytes=0,
        )
        chain.advance(operator.dimension)
        assert chain.current > 0
        assert chain.basis.devices() == {operator.device}
        reduced = chain.hessenberg[:chain.current, :chain.current]
        reduced_right = np.zeros(chain.current, dtype=complex)
        reduced_right[0] = np.linalg.norm(right)
        for position, frequency in enumerate(frequencies):
            shift = frequency + 1j * broadening
            coordinates = np.linalg.solve(
                shift * np.eye(chain.current) - reduced, reduced_right
            )
            observed[position, :, column] = chain.projected[:, :chain.current] @ coordinates
            reconstructed = chain.reconstruct(coordinates)
            assert reconstructed.devices() == {operator.device}
            residual = shift * reconstructed - operator.device_matvec(reconstructed)
            np.testing.assert_allclose(np.asarray(residual), right, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(observed, expected, atol=1e-9, rtol=1e-9)

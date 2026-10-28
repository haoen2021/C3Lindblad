"""SOC full-S135 at fixed CCSD amplitudes; physical ED is asserted only at U=0."""

import numpy as np
import pytest
import scipy.linalg

from c3l import FermionicLindbladModel, SparseTwoBodyInteraction
from c3l.cc.component_tensor import (
    ComponentEOMProblem,
    ComponentStaticProblem,
    component_eom_context,
    component_eom_sources,
)
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import factorized_cc_problem
from c3l.cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator
from tests.cc.test_charge_s13 import exact_lindblad
from tests.reference.generator import printed_quadratic_paper_K
from tests.reference.hf_operators import (
    creation_matrix,
    global_gaussian_vacuum,
    global_particle_hole_operators,
)


def _charge_s135(interaction=1.0):
    coupling = interaction * (0.06 + 0.03j)
    model = FermionicLindbladModel(
        np.array([
            [0.13, 0.09 + 0.07j, 0.04 - 0.02j],
            [0.09 - 0.07j, -0.21, 0.06 + 0.03j],
            [0.04 + 0.02j, 0.06 - 0.03j, 0.08],
        ]),
        np.array([
            [0.31, 0.025j, 0.01],
            [-0.025j, 0.24, 0.012j],
            [0.01, -0.012j, 0.28],
        ]),
        np.array([
            [0.17, 0.02, -0.009j],
            [0.02, 0.11, 0.008],
            [0.009j, 0.008, 0.14],
        ]),
        U=0.0, symmetry="charge", interaction_sites=(0, 1, 2),
        two_body_interaction=SparseTwoBodyInteraction(3, (
            (0, 1, 0, 1, 0.13 * interaction),
            (1, 2, 1, 2, 0.09 * interaction),
            (0, 2, 0, 1, coupling),
            (0, 1, 0, 2, coupling.conjugate()),
        )),
    )
    oracle = factorized_cc_problem(model, "CCSD", build_next_rank_masks=False)
    static = ComponentStaticProblem.compile(oracle)
    amplitudes = oracle.tilde.unpack(
        interaction * 0.002 * np.sin(np.arange(oracle.num_variables) + 0.19)
    )
    eom = eom_ccsd_problem(oracle, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(problem, amplitudes)
    operator = FullS135ResponseOperator(
        CompiledFullS135Action(context, materialization_batch_size=4),
        s5_tile_size=2,
    )
    assert oracle.cluster_basis.num_modes == 6
    assert (operator.p_dimension, operator.s5_dimension, operator.dimension) == (12, 3, 15)
    return model, operator, amplitudes


def _dense_projected_action(operator, amplitudes):
    """Construct [exp(-T) K exp(T), R]|0> in independent determinant space."""
    oracle = operator.action.context.problem.static.oracle
    vacuum = global_gaussian_vacuum(oracle.reference)
    _, creators = global_particle_hole_operators(oracle.reference)
    dimension = vacuum.state.size
    determinants = np.empty((dimension, dimension), dtype=complex)
    determinants[:, 0] = vacuum.state
    for mask in range(1, dimension):
        lowest = mask & -mask
        determinants[:, mask] = (
            creators[lowest.bit_length() - 1] @ determinants[:, mask ^ lowest]
        )
    generator = np.linalg.solve(
        determinants, printed_quadratic_paper_K(oracle.model) @ determinants
    )
    cluster = sum(
        amplitude * creation_matrix(mask, oracle.cluster_basis.num_modes)
        for amplitude, mask in zip(amplitudes, oracle.cluster_basis.masks)
    )
    transformed = scipy.linalg.expm(-cluster) @ generator @ scipy.linalg.expm(cluster)
    masks = np.empty(operator.dimension, dtype=np.int64)
    for block in operator.coordinates.blocks:
        offsets = np.repeat(np.arange(4) * operator.coordinates.n, block.key)
        masks[block.vector_positions] = np.sum(1 << (block.indices + offsets), axis=1)
    assert len(set(masks)) == operator.dimension
    assert {int(mask) for mask in masks[operator.p_dimension:]} == {
        mask for mask in range(dimension)
        if mask.bit_count() == 5 and (mask & 0b111).bit_count() == 3
    }
    projected = np.empty((operator.dimension, operator.dimension), dtype=complex)
    for column, mask in enumerate(masks):
        creator = creation_matrix(int(mask), oracle.cluster_basis.num_modes)
        projected[:, column] = (
            transformed[:, mask] - creator @ transformed[:, 0]
        )[masks]
    return projected


@pytest.fixture(scope="module")
def interacting_s135():
    _, operator, amplitudes = _charge_s135()
    projected = _dense_projected_action(operator, amplitudes)
    retained = operator.p_dimension
    assert np.linalg.norm(projected[:retained, retained:]) > 1e-8
    assert np.linalg.norm(projected[retained:, :retained]) > 1e-8
    return operator, projected


def test_charge_soc_full_s135_cpu_matches_dense_projection(interacting_s135):
    operator, expected = interacting_s135
    probes = (1.0 + 0.2j) * np.eye(operator.dimension)
    observed = np.column_stack([
        operator.matvec(vector) for vector in probes.T
    ])
    np.testing.assert_allclose(observed, expected @ probes, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
def test_charge_soc_full_s135_jax_and_resident_response(interacting_s135, platform):
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

    from c3l.cc.s135_jax import JaxFullS135ResponseOperator
    from c3l.green.resident_arnoldi import ResidentArnoldi

    cpu, expected = interacting_s135
    operator = JaxFullS135ResponseOperator.from_cpu(
        cpu, platform=platform, production=True
    )
    assert operator.device.platform == platform
    probes = (1.0 + 0.2j) * np.eye(cpu.dimension)
    observed = np.column_stack([
        operator.matvec(vector) for vector in probes.T
    ])
    np.testing.assert_allclose(observed, expected @ probes, atol=1e-10, rtol=1e-10)

    sources = tuple(
        component_eom_sources(cpu.action.context, site=orbital) for orbital in range(3)
    )
    sinks = np.pad(
        np.vstack([source.sink for source in sources]), ((0, 0), (0, cpu.s5_dimension))
    )
    shifts = np.array([-0.6, 0.0, 0.7]) + 0.13j
    for source in sources:
        right = np.pad(source.addition + source.removal, (0, cpu.s5_dimension))
        chain = ResidentArnoldi(
            operator.device_matvec, right, sinks, cpu.dimension,
            panel=2, device=operator.device, workspace_bytes=0,
        )
        chain.advance(cpu.dimension)
        assert chain.current > 0
        assert chain.basis.devices() == {operator.device}
        reduced = chain.hessenberg[:chain.current, :chain.current]
        reduced_right = np.zeros(chain.current, dtype=complex)
        reduced_right[0] = np.linalg.norm(right)
        for shift in shifts:
            dense_response = np.linalg.solve(shift * np.eye(cpu.dimension) - expected, right)
            assert np.linalg.norm(dense_response[cpu.p_dimension:]) > 1e-8
            coordinates = np.linalg.solve(
                shift * np.eye(chain.current) - reduced, reduced_right
            )
            np.testing.assert_allclose(
                chain.projected[:, :chain.current] @ coordinates,
                sinks @ dense_response, atol=1e-9, rtol=1e-9,
            )
            reconstructed = chain.reconstruct(coordinates)
            assert reconstructed.devices() == {operator.device}
            np.testing.assert_allclose(
                np.asarray(reconstructed), dense_response, atol=1e-9, rtol=1e-9
            )


def test_charge_soc_noninteracting_full_s135_matches_physical_ed():
    model, operator, _ = _charge_s135(interaction=0.0)
    sources = tuple(
        component_eom_sources(operator.action.context, site=orbital) for orbital in range(3)
    )
    frequencies = np.array([-0.6, 0.0, 0.7])
    broadening = 0.13
    identity = np.eye(operator.dimension)
    action = np.column_stack([operator.matvec(vector) for vector in identity])
    right = np.column_stack([
        np.pad(source.addition + source.removal, (0, operator.s5_dimension))
        for source in sources
    ])
    sinks = np.pad(
        np.vstack([source.sink for source in sources]),
        ((0, 0), (0, operator.s5_dimension)),
    )
    observed = np.array([
        sinks @ np.linalg.solve((frequency + 1j * broadening) * identity - action, right)
        for frequency in frequencies
    ])
    density, _, generator, annihilation = exact_lindblad(model)
    expected = np.empty((frequencies.size, 3, 3), dtype=complex)
    for position, frequency in enumerate(frequencies):
        resolvent = (frequency + 1j * broadening) * np.eye(generator.shape[0]) - 1j * generator
        for column, operator_annihilation in enumerate(annihilation):
            creator = operator_annihilation.conj().T
            right = (creator @ density + density @ creator).reshape(-1, order="F")
            response = np.linalg.solve(resolvent, right).reshape(density.shape, order="F")
            for row, sink in enumerate(annihilation):
                expected[position, row, column] = np.trace(sink @ response)
    assert np.max(np.abs(expected[:, 0, 1])) > 1e-3
    assert np.max(np.abs(expected[:, 1, 0])) > 1e-3
    np.testing.assert_allclose(observed, expected, atol=1e-9, rtol=1e-9)

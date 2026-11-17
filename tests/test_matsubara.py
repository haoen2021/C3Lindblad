"""Matsubara-axis G(iw_n) and Sigma(iw_n) against exact Lindblad references."""

import numpy as np
import pytest

from c3l import FermionicLindbladModel
from c3l.cc.component_response import (
    ComponentSigmaOperator,
    component_retarded_matrix_green,
    component_retarded_matrix_green_arnoldi,
    component_retarded_matrix_green_shifts,
)
from c3l.matsubara import (
    MatsubaraGrid,
    component_matsubara_green,
    dlr_matsubara_grid,
    matsubara_dyson_green,
    matsubara_grid,
    matsubara_noninteracting_green,
    matsubara_self_energy,
    noninteracting_retarded_green,
    solve_matsubara,
)
from tests.cc.test_charge_s13 import charge_model, exact_retarded_green, solved_charge


BETA = 2.0
INDICES = (-4, -1, 0, 1, 3, 7)


def exact_matsubara(model, grid):
    """Exact Lindblad G^R at i|w_n|, with the adjoint on negative frequencies."""

    values = np.stack([exact_retarded_green(model, [0.0], abs(w))[0] for w in grid.omega_n])
    negative = grid.indices < 0
    values[negative] = values[negative].conj().swapaxes(1, 2)
    return values


def spin_model(U=0.4):
    """Impurity site 0 with one damped bath site, spin-mode symmetry."""

    return FermionicLindbladModel(
        [[-0.2, 0.17], [0.17, 0.11]],
        [[0.0, 0.0], [0.0, 0.32]],
        [[0.0, 0.0], [0.0, 0.18]],
        U=U,
    )


def as_charge_model(model):
    return FermionicLindbladModel(
        model.physical_E,
        model.physical_gamma_loss,
        model.physical_gamma_gain,
        U=0.0,
        symmetry="charge",
        interaction_sites=(0, 1),
        two_body_interaction=model.interaction,
    )


def test_explicit_grids_agree_and_map_negative_frequencies_by_adjoint():
    by_index = matsubara_grid(BETA, indices=INDICES)
    by_frequency = matsubara_grid(BETA, omega_n=by_index.omega_n)
    by_point = matsubara_grid(BETA, omega_n=by_index.iw)
    np.testing.assert_array_equal(by_frequency.indices, by_index.indices)
    np.testing.assert_array_equal(by_point.indices, by_index.indices)
    window = matsubara_grid(BETA, n_iw=3)
    np.testing.assert_array_equal(window.indices, np.arange(-3, 3))
    np.testing.assert_allclose(by_index.omega_n, (2 * np.array(INDICES) + 1) * np.pi / BETA)
    # Distinct |w_n| only: n=-1 and n=0 share |w|=pi/beta, n=-4 and n=3 share 7 pi/beta.
    np.testing.assert_allclose(by_index.upper_shifts, 1j * np.array([1, 3, 7, 15]) * np.pi / BETA)

    rng = np.random.default_rng(3)
    upper = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
    mapped = by_index.from_upper(upper)
    lookup = {1: 0, 3: 1, 7: 2, 15: 3}
    for position, n in enumerate(INDICES):
        expected = upper[lookup[abs(2 * n + 1)]]
        expected = expected.conj().T if n < 0 else expected
        np.testing.assert_array_equal(mapped[position], expected)
    np.testing.assert_array_equal(by_index.from_upper(upper[:, 0, 1])[1], upper[0, 0, 1].conj())


@pytest.mark.parametrize(
    "arguments",
    [
        dict(omega_n=[np.pi]),
        dict(omega_n=[2 * np.pi / BETA]),
        dict(omega_n=[1.0 + 1j * np.pi / BETA]),
        dict(indices=[0, 0]),
        dict(indices=[0.0]),
        dict(n_iw=0),
        dict(indices=[0], n_iw=2),
    ],
)
def test_invalid_grids_are_rejected(arguments):
    with pytest.raises(ValueError):
        matsubara_grid(BETA, **arguments)


@pytest.mark.parametrize("interaction", [0.0, 0.4])
def test_charge_s13_matsubara_matches_lindblad_ed(interaction):
    model, _, _, context, sources = solved_charge(interaction)
    grid = matsubara_grid(BETA, indices=INDICES)
    result = component_matsubara_green(
        ComponentSigmaOperator(context), sources, grid, minimum_krylov_dimension=4
    )
    expected = exact_matsubara(model, grid)
    assert np.max(np.abs(expected[:, 0, 1])) > 1e-3
    np.testing.assert_allclose(result.green, expected, atol=1e-10, rtol=1e-10)
    assert result.max_relative_residual < 1e-10


def test_shifted_arnoldi_matches_direct_gmres_and_real_axis_arnoldi():
    _, _, _, context, sources = solved_charge(0.4)
    omega = np.array([-0.6, 0.2, 0.7])
    eta = 0.13
    shifted = component_retarded_matrix_green_shifts(
        ComponentSigmaOperator(context), sources, omega + 1j * eta, minimum_krylov_dimension=4
    )
    arnoldi = component_retarded_matrix_green_arnoldi(
        ComponentSigmaOperator(context), sources, omega, eta, minimum_krylov_dimension=4
    )
    direct = component_retarded_matrix_green(
        ComponentSigmaOperator(context), sources, omega, eta, rtol=1e-12, restart=10, maxiter=30
    )
    np.testing.assert_array_equal(shifted.retarded, arnoldi.retarded)
    np.testing.assert_allclose(shifted.retarded, direct.retarded, atol=1e-10)
    with pytest.raises(ValueError):
        component_retarded_matrix_green_shifts(ComponentSigmaOperator(context), sources, [0.3 - 0.1j])


def test_solve_matsubara_charge_s13_and_s135_match_lindblad_ed():
    model = charge_model(0.4)
    grid = matsubara_grid(BETA, indices=INDICES)
    expected = exact_matsubara(model, grid)
    for rank in ("S13", "S135"):
        solution = solve_matsubara(model, grid, sites=(0, 1), rank=rank, minimum_krylov_dimension=4)
        np.testing.assert_allclose(solution.green, expected, atol=1e-9, rtol=1e-9)
        assert solution.static_residual < 1e-10
        # The Dyson identity holds with the PM hybridization, and Sigma is
        # the self-energy of the exact G on this two-orbital model.
        np.testing.assert_allclose(solution.dyson(solution.hybridization), solution.green, atol=1e-12)
        exact_sigma = np.linalg.inv(solution.noninteracting) - np.linalg.inv(expected)
        np.testing.assert_allclose(solution.self_energy, exact_sigma, atol=1e-8)
        # Matsubara symmetry Sigma(-iw) = Sigma(iw)^dagger and causality Im Sigma_aa(iw>0) <= 0.
        np.testing.assert_allclose(
            solution.self_energy[INDICES.index(-1)],
            solution.self_energy[INDICES.index(0)].conj().T,
            atol=1e-12,
        )
        positive = grid.indices >= 0
        assert np.all(np.diagonal(solution.self_energy[positive], axis1=1, axis2=2).imag < 1e-12)


def test_noninteracting_self_energy_vanishes_and_g0_is_the_pm_resolvent():
    model = spin_model(U=0.0)
    grid = matsubara_grid(BETA, n_iw=4)
    solution = solve_matsubara(model, grid, minimum_krylov_dimension=4)
    assert solution.sites == (0,)
    np.testing.assert_allclose(solution.self_energy, 0.0, atol=1e-11)
    np.testing.assert_allclose(solution.green, solution.noninteracting, atol=1e-11)
    # PM hybridization of one bath site: V^2 / (z - e_b + i Lambda_b).
    z = grid.iw
    lam = 0.32 + 0.18
    retarded = 0.17**2 / (np.abs(z.imag) * 1j - 0.11 + 1j * lam)
    expected = np.where(grid.indices >= 0, retarded, retarded.conj())
    np.testing.assert_allclose(solution.hybridization[:, 0, 0], expected, atol=1e-13)
    np.testing.assert_allclose(
        noninteracting_retarded_green(model, z[grid.indices >= 0], (0,))[:, 0, 0],
        1.0 / (z[grid.indices >= 0] + 0.2 - expected[grid.indices >= 0]),
        atol=1e-13,
    )


def test_spin_blocks_match_charge_mode_spin_orbitals():
    spin = spin_model(U=0.4)
    charge = as_charge_model(spin)
    grid = matsubara_grid(BETA, indices=INDICES)
    up = solve_matsubara(spin, grid, response_spin=0, minimum_krylov_dimension=4)
    down = solve_matsubara(spin, grid, response_spin=1, minimum_krylov_dimension=4)
    full = solve_matsubara(charge, grid, sites=(0, 1), minimum_krylov_dimension=4)
    np.testing.assert_allclose(full.green[:, 0, 0], up.green[:, 0, 0], atol=1e-9)
    np.testing.assert_allclose(full.green[:, 1, 1], down.green[:, 0, 0], atol=1e-9)
    np.testing.assert_allclose(full.green[:, 0, 1], 0.0, atol=1e-10)
    np.testing.assert_allclose(full.self_energy[:, 0, 0], up.self_energy[:, 0, 0], atol=1e-9)
    assert np.max(np.abs(up.self_energy)) > 1e-2


def test_dyson_reinserts_sigma_into_a_target_hybridization():
    spin = spin_model(U=0.4)
    grid = matsubara_grid(BETA, n_iw=5)
    solution = solve_matsubara(spin, grid, minimum_krylov_dimension=4)
    # A target bath (semicircle, half-bandwidth 2, hopping 0.3) on the Matsubara axis.
    z = grid.iw
    target = 0.09 * (z - np.sign(z.imag) * np.sqrt(z**2 - 4.0 + 0j)) / 2.0
    target = target[:, None, None]
    observed = solution.dyson(target)
    expected = 1.0 / (z + 0.2 - target[:, 0, 0] - solution.self_energy[:, 0, 0])
    np.testing.assert_allclose(observed[:, 0, 0], expected, atol=1e-13)
    np.testing.assert_allclose(
        matsubara_dyson_green(grid, solution.one_body, solution.hybridization, solution.self_energy),
        solution.green,
        atol=1e-12,
    )


def test_self_energy_requires_every_interacting_site():
    model = as_charge_model(spin_model(U=0.4))
    grid = matsubara_grid(BETA, n_iw=2)
    green = matsubara_noninteracting_green(model, grid, (0, 1))
    with pytest.raises(ValueError, match="interaction site"):
        matsubara_self_energy(model, grid, green[:, :1, :1], sites=(0,))
    with pytest.raises(ValueError, match="shape"):
        matsubara_self_energy(model, grid, green[:, :1, :1], sites=(0, 1))


def test_jax_cpu_response_matches_numpy():
    jax = pytest.importorskip("jax")
    if not jax.config.jax_enable_x64:
        pytest.skip("JAX response requires JAX_ENABLE_X64")
    model = charge_model(0.4)
    grid = matsubara_grid(BETA, indices=INDICES)
    reference = solve_matsubara(model, grid, sites=(0, 1), minimum_krylov_dimension=4)
    for rank in ("S13", "S135"):
        observed = solve_matsubara(
            model, grid, sites=(0, 1), rank=rank, platform="cpu",
            amplitudes=reference.amplitudes, minimum_krylov_dimension=4,
        )
        np.testing.assert_allclose(observed.green, reference.green, atol=1e-10)


def test_dlr_grid_matches_pydlr_and_interpolates_matsubara_data():
    pydlr = pytest.importorskip("pydlr")
    beta, eps, wmax = 10.0, 1e-12, 4.0
    grid = dlr_matsubara_grid(beta, eps, wmax=wmax)
    assert isinstance(grid, MatsubaraGrid) and grid.kind == "dlr"
    assert grid.lamb == pytest.approx(beta * wmax)
    reference = pydlr.dlr(lamb=beta * wmax, eps=eps)
    np.testing.assert_allclose(grid.iw, reference.get_matsubara_frequencies(beta), atol=1e-14)
    assert len(dlr_matsubara_grid(beta, eps, lamb=beta * wmax)) == len(grid)

    # A non-Hermitian matrix sum of poles inside [-wmax, wmax] is recovered off the grid,
    # on the Matsubara axis and in imaginary time.
    rng = np.random.default_rng(7)
    poles = rng.uniform(-wmax, wmax, 6)
    weights = rng.normal(size=(6, 2, 2)) + 1j * rng.normal(size=(6, 2, 2))

    def green(z):
        return np.einsum("kab,wk->wab", weights, 1.0 / (z[:, None] - poles[None, :]))

    def green_tau(tau):
        kernel = np.where(
            poles[None, :] >= 0,
            np.exp(-np.abs(poles) * tau[:, None]),
            np.exp(-np.abs(poles) * (beta - tau[:, None])),
        ) / (1.0 + np.exp(-beta * np.abs(poles)))
        return -np.einsum("kab,tk->tab", weights, kernel)

    values = green(grid.iw)
    dense = matsubara_grid(beta, n_iw=300)
    np.testing.assert_allclose(grid.dlr_matsubara(values, dense.iw), green(dense.iw), atol=1e-9)
    tau = np.linspace(0.0, beta, 41)
    np.testing.assert_allclose(grid.dlr_tau(values, tau), green_tau(tau), atol=1e-9)
    np.testing.assert_allclose(grid.dlr_tau(values[:, 0, 1], tau), green_tau(tau)[:, 0, 1], atol=1e-9)
    with pytest.raises(ValueError, match="DLR"):
        matsubara_grid(beta, n_iw=4).dlr_tau(np.zeros(8), tau)
    with pytest.raises(ValueError):
        dlr_matsubara_grid(beta, eps)
    with pytest.raises(ValueError):
        dlr_matsubara_grid(beta, eps, lamb=10.0, wmax=1.0)


def test_solver_on_a_dlr_grid_equals_the_explicit_grid_with_the_same_nodes():
    pytest.importorskip("pydlr")
    model = spin_model(U=0.4)
    grid = dlr_matsubara_grid(BETA, 1e-8, wmax=3.0)
    explicit = matsubara_grid(BETA, indices=grid.indices)
    on_dlr = solve_matsubara(model, grid, minimum_krylov_dimension=4)
    on_explicit = solve_matsubara(model, explicit, amplitudes=on_dlr.amplitudes, minimum_krylov_dimension=4)
    np.testing.assert_array_equal(on_dlr.green, on_explicit.green)
    np.testing.assert_array_equal(on_dlr.self_energy, on_explicit.self_energy)

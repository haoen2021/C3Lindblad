"""Matsubara-axis CCSD impurity solver: G(iw_n) and Sigma(iw_n) on a DLR grid.

This is the imaginary-axis interface an imaginary-time DMFT loop needs:

1. Fit pseudomodes to an equilibrium semicircle bath at inverse temperature
   beta (RealTimeBath, as in the SIAM examples).
2. Solve CCSD and the S13 response, evaluated at the fermionic DLR Matsubara
   nodes for (beta, eps, Lambda = beta * wmax).  G(iw_n) = G^R(iw_n) for
   w_n > 0 and G(iw_n) = G^R(-iw_n)^dagger for w_n < 0.
3. Sigma = G0_PM^{-1} - G^{-1}, inserted into the exact semicircle
   hybridization Delta(iw_n) (the continuous bath the modes were fitted to).
4. Transform to imaginary time with the DLR, check n = -G(beta) = 1/2 and
   the Hartree tail Sigma(iw -> inf) = U/2 at particle-hole symmetry.

Energies are in units of Gamma; requires the optional pydlr package
(pip install "c3l[dlr]").  Runs in about 15 s on one CPU core; increase
--modes for a more accurate bath fit:

    python examples/basic/05_matsubara_dlr.py --output matsubara_dlr.npz
"""

import argparse

import numpy as np
from scipy.special import expit, roots_legendre

from c3l import FermionicLindbladModel
from c3l.matsubara import dlr_matsubara_grid, matsubara_grid, solve_matsubara


def semicircle_weights(Gamma, W, order=1024):
    """Nodes and weights of J(w) = Gamma sqrt(1 - (w/W)^2) / pi on [-W, W]."""

    x, w = roots_legendre(order)
    return W * x, Gamma * W * w * np.sqrt(1.0 - x * x) / np.pi


def lesser_correlation(t, beta, Gamma, W):
    """C_less(t) = integral J(w) f(w) exp(-i w t) dw."""

    nodes, weights = semicircle_weights(Gamma, W)
    return (weights * expit(-beta * nodes)) @ np.exp(-1j * np.outer(nodes, t))


def matsubara_hybridization(iw, Gamma, W):
    """Delta(iw_n) = integral J(w) / (iw_n - w) dw, as a (n, 1, 1) array."""

    nodes, weights = semicircle_weights(Gamma, W)
    return (weights[None, :] / (iw[:, None] - nodes[None, :])).sum(axis=1)[:, None, None]


def siam(H, D, g, U):
    """Filled bath (H, D, g) and its particle-hole conjugate empty bath."""

    N = len(g)
    E = np.zeros((1 + 2 * N, 1 + 2 * N), complex)
    loss, gain = np.zeros_like(E), np.zeros_like(E)
    filled, empty = slice(1, 1 + N), slice(1 + N, 1 + 2 * N)
    E[0, 0] = -U / 2
    E[filled, filled], E[empty, empty] = H, -H.conj()
    E[filled, 0], E[0, filled] = g, g.conj()
    E[empty, 0], E[0, empty] = g.conj(), g
    gain[filled, filled], loss[empty, empty] = D, D.conj()
    return FermionicLindbladModel(E, loss, gain, U=U, units="Gamma")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--U", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--W", type=float, default=10.0, help="semicircle half-bandwidth")
    parser.add_argument("--modes", type=int, default=3, help="pseudomodes per branch")
    parser.add_argument("--eps", type=float, default=1e-10, help="DLR accuracy")
    parser.add_argument("--output", default=None, help="optional .npz output")
    args = parser.parse_args()
    from realtimebath import fit_correlation

    Gamma = 1.0
    t = np.linspace(0.0, 10.0, 501)
    fit = fit_correlation(
        t, lesser_correlation(t, args.beta, Gamma, args.W), n_modes=args.modes,
        method="sdp", solver="CLARABEL", optimize=True,
        max_nfev=5000, optimization_max_nfev=20000, optimization_max_points=len(t),
    )
    t_check = np.linspace(0.0, 10.0, 1002)
    exact = lesser_correlation(t_check, args.beta, Gamma, args.W)
    error = np.linalg.norm(fit.evaluate(t_check) - exact) / np.linalg.norm(exact)
    print(f"bath fit: {args.modes} modes per branch, relative L2 error {error:.2e}")
    model = siam(fit.model.hamiltonian, fit.model.damping, fit.model.coupling, args.U)

    # Lambda = beta * wmax must cover the impurity spectrum: band plus Hubbard bands.
    wmax = args.W + args.U
    grid = dlr_matsubara_grid(args.beta, args.eps, wmax=wmax)
    print(f"DLR grid: Lambda={grid.lamb:g}, eps={grid.eps:g}, {len(grid)} Matsubara nodes")
    solution = solve_matsubara(model, grid, rank="S13")
    print(f"CCSD residual {solution.static_residual:.1e}; "
          f"max Arnoldi residual {solution.response.max_relative_residual:.1e}")

    delta = matsubara_hybridization(grid.iw, Gamma, args.W)
    G = solution.dyson(delta)          # physical impurity G with the exact bath
    Sigma = solution.self_energy

    dlr, beta = grid.dlr, args.beta
    tau = np.linspace(0.0, beta, 201)
    G_tau = dlr.eval_dlr_tau(dlr.dlr_from_matsubara(G, beta), tau, beta)
    G_tau_pm = dlr.eval_dlr_tau(dlr.dlr_from_matsubara(solution.green, beta), tau, beta)
    print(f"n_sigma = -G(beta): reconstructed {-G_tau[-1, 0, 0].real:.6f}, "
          f"PM {-G_tau_pm[-1, 0, 0].real:.6f} (exact 1/2)")
    print(f"particle-hole: max |Re G(iw_n)| = {np.max(np.abs(G.real)):.1e}")
    largest = np.argmax(np.abs(grid.omega_n))
    print(f"Sigma at the largest node w={grid.omega_n[largest]:.1f}: {Sigma[largest, 0, 0]:.4f} "
          f"(Hartree limit U/2 = {args.U / 2:g})")

    # DLR interpolation check against a direct evaluation on a dense grid.
    dense = matsubara_grid(beta, n_iw=64)
    direct = solve_matsubara(model, dense, amplitudes=solution.amplitudes)
    interpolated = dlr.eval_dlr_freq(dlr.dlr_from_matsubara(solution.green, beta), dense.iw, beta)
    print(f"DLR interpolation of the PM G on 128 dense points: max error "
          f"{np.max(np.abs(interpolated - direct.green)):.1e}")

    if args.output:
        np.savez(
            args.output, beta=beta, U=args.U, n=grid.indices, iw=grid.iw, lamb=grid.lamb, eps=grid.eps,
            G_iw=G, G_iw_PM=solution.green, Sigma_iw=Sigma, Delta_iw=delta,
            Delta_iw_PM=solution.hybridization, tau=tau, G_tau=G_tau,
        )
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()

"""RealTimeBath → particle-hole-symmetric SIAM → CCSD + S_{1,3} spectra.

Run with the optional RealTimeBath dependency installed.
N counts lesser AND greater modes: 1+2N spatial orbitals, 2+4N spin orbitals.
All energies are in Gamma=1, hbar=1; no continuum Dyson reconstruction is used.
"""

from pathlib import Path
from ..parameters import parameters
from time import perf_counter
from dataclasses import asdict
import argparse
import json
import resource

import numpy as np
from scipy.special import roots_legendre, expit

from c3l import FermionicLindbladModel

PARAMETERS = parameters("siam_modes")
U, W, beta, eta = (PARAMETERS[k] for k in ("U", "W", "beta", "eta"))
t_fit = np.linspace(*PARAMETERS["fit"]["time"])
t_check = np.linspace(*PARAMETERS["fit"]["validation_time"])
omega = np.linspace(*PARAMETERS["frequency"])


def target(t, order=PARAMETERS["fit"]["quadrature_order"]):
    """C_less(t)=integral J(w) f(w) exp(-iwt) dw; J=semicircle/pi."""
    x, w = roots_legendre(order)
    weight = PARAMETERS["Gamma"] * W * w * np.sqrt(1 - x * x) / np.pi * expit(-beta * W * x)
    return weight @ np.exp(-1j * np.outer(W * x, t))


def siam(H, D, g, U=U):
    """Filled bath (H,D,g); PH-conjugate empty bath (-H*,D*,g*).

    C3L uses D(2a rho a†-{a†a,rho}), so its damping matrix is D,
    not RealTimeBath's jump Gram matrix 2D. Spin copies are independent.
    """
    N = len(g)
    E = np.zeros((1 + 2 * N, 1 + 2 * N), complex)
    loss, gain = np.zeros_like(E), np.zeros_like(E)
    l, r = slice(1, 1 + N), slice(1 + N, 1 + 2 * N)
    E[0, 0] = -U / 2
    E[l, l], E[r, r] = H, -H.conj()
    E[l, 0], E[0, l] = g, g.conj()
    E[r, 0], E[0, r] = g.conj(), g
    gain[l, l], loss[r, r] = D, D.conj()
    return FermionicLindbladModel(E, loss, gain, U=U, units="Gamma")


def check_parameters(folder):
    """Do not silently reuse a bath or spectrum from different input settings."""
    record = folder / "parameters.json"
    if record.exists():
        if json.loads(record.read_text()) != PARAMETERS:
            raise ValueError("Parameters changed; choose a fresh --output directory")
    elif any(folder.glob("bath_N*.npz")) or any(folder.glob("cc_N*.npz")) or any(folder.glob("spectrum_N*.csv")):
        raise ValueError("Existing output has no parameter record; choose a fresh --output directory")
    else:
        record.write_text(json.dumps(PARAMETERS, indent=2) + "\n")


def fit_bath(N, folder):
    """Fit once; retain physical parameters and optimizer termination details."""
    from realtimebath import fit_correlation

    path = folder / f"bath_N{N}.npz"
    if path.exists():
        with np.load(path) as d:
            return d["H"], d["D"], d["g"]
    start = perf_counter()
    print(f"N={N}: fitting bath (50000 refinement evaluations allowed)", flush=True)
    fit = fit_correlation(
        t_fit,
        target(t_fit),
        n_modes=N,
        method="sdp",
        solver="CLARABEL",
        optimize=True,
        max_nfev=PARAMETERS["fit"]["max_nfev"],
        optimization_max_nfev=PARAMETERS["fit"]["optimization_max_nfev"],
        optimization_max_points=len(t_fit),
    )
    H, D, g = fit.model.hamiltonian, fit.model.damping, fit.model.coupling
    exact, fitted = target(t_check), fit.evaluate(t_check)
    J = PARAMETERS["Gamma"] * np.sqrt(np.maximum(0.0, 1 - (omega / W) ** 2)) / np.pi
    # spectral_density is 2*pi*J*f, not J*f.
    Jfit = (fit.spectral_density(omega) + fit.spectral_density(-omega)) / (2 * np.pi)
    metrics = asdict(fit.diagnostics)
    metrics.update(
        wall_seconds=perf_counter() - start,
        validation_relative_L2=float(
            np.linalg.norm(fitted - exact) / np.linalg.norm(exact)
        ),
        equal_time_relative_error=float(abs(fitted[0] - exact[0]) / abs(exact[0])),
        spectrum_relative_L2=float(np.linalg.norm(Jfit - J) / np.linalg.norm(J)),
        U=U,
        W=W,
        beta=beta,
        Gamma=PARAMETERS["Gamma"],
        N=N,
        time_window=[0.0, 10.0],
        time_points=len(t_fit),
        frequency_window=[-15.0, 15.0],
    )
    (folder / f"bath_N{N}.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x),
        )
        + "\n"
    )
    np.savez(
        path,
        H=H,
        D=D,
        g=g,
        t=t_check,
        C=exact,
        C_fit=fitted,
        omega=omega,
        J=J,
        J_fit=Jfit,
    )
    print(
        f"  bath validation L2={metrics['validation_relative_L2']:.3e}; "
        f"J L2={metrics['spectrum_relative_L2']:.3e}; "
        f"{metrics['wall_seconds']:.1f}s",
        flush=True,
    )
    return H, D, g


def spectrum(N, H, D, g, folder):
    """Solve static CCSD and evaluate the S13 Green function."""
    from c3l.cc.factorized import factorized_cc_problem
    from c3l.cc.component_tensor import (
        ComponentStaticProblem,
        ComponentEOMProblem,
        component_eom_context,
        component_eom_sources,
        component_static_residual,
    )
    from c3l.cc.component_solver import solve_component_ccsd
    from c3l.cc.newton_krylov import NewtonKrylovPolicy
    from c3l.cc.eom import eom_ccsd_problem
    from c3l.cc.component_response import (
        ComponentSigmaOperator,
        component_retarded_matrix_green_arnoldi,
    )

    start = perf_counter()
    print(f"N={N}: compiling CCSD at U={U}", flush=True)
    model = siam(H, D, g)
    # PH symmetry gives n_up=n_down=1/2: the restricted onsite HF shift
    # cancels epsilon=-U/2. m_up fixes the existing reference explicitly.
    cc = factorized_cc_problem(
        model,
        "CCSD",
        m_up=0.5,
        build_next_rank_masks=False,
        reference_mode="legacy_half_filled",
    )
    static = ComponentStaticProblem.compile(cc)
    policy = NewtonKrylovPolicy(**PARAMETERS["static"])

    def accepted(k, T, norm):
        print(f"  Newton {k}: residual={norm:.6e}", flush=True)
        np.savez(folder / f"cc_N{N}_latest.npz", amplitudes=T, residual=norm)

    solution = solve_component_ccsd(
        static, policy=policy, compute_rank6=False, accepted_step_callback=accepted
    )
    fresh = component_static_residual(static, solution.amplitudes)
    residual = float(np.linalg.norm(fresh.retained))
    metrics = dict(
        N=N,
        U=U,
        W=W,
        beta=beta,
        eta=eta,
        cc_residual=residual,
        cc_wall_seconds=solution.wall_seconds,
        cc_jvp_calls=solution.jvp_evaluations,
        cc_history=[asdict(x) for x in solution.history],
    )
    (folder / f"cc_N{N}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if not solution.converged or residual > 1e-8:
        raise RuntimeError(f"N={N}: CCSD not converged; residual={residual}")
    print(f"  CCSD residual={residual:.3e}; computing S13 response", flush=True)
    eom = eom_ccsd_problem(static.oracle, response_spin=0, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(problem, solution.amplitudes)
    sources = component_eom_sources(context)
    operator = ComponentSigmaOperator(context)
    response = component_retarded_matrix_green_arnoldi(
        operator,
        [sources],
        omega,
        eta,
        maximum_krylov_dimension=min(PARAMETERS["response"]["maximum_krylov_dimension"], operator.dimension),
        residual_tolerance=PARAMETERS["response"]["residual_tolerance"],
        check_interval=32,
    )
    G = response.retarded[:, 0, 0]
    A = -G.imag / np.pi
    # Reapply the actual action at three representative shifts, not just H_m.
    true = []
    for projection, b in [
        (response.addition_projections[0], sources.addition),
        (response.removal_projections[0], sources.removal),
    ]:
        for z, x in zip(
            np.array([-5.0, 0.0, 5.0]) + 1j * eta,
            projection.solutions(np.array([-5.0, 0.0, 5.0]) + 1j * eta),
        ):
            true.append(
                float(
                    np.linalg.norm(z * x - operator.matvec(x) - b) / np.linalg.norm(b)
                )
            )
    if max(true) > 1e-8:
        raise RuntimeError(f"EOM true residual too large: {max(true)}")
    metrics.update(
        eom_true_residual=max(true),
        eom_maximum_projected_residual=float(
            max(
                np.max(response.addition_relative_residuals),
                np.max(response.removal_relative_residuals),
            )
        ),
        krylov_dimensions=[
            response.addition_projections[0].dimension,
            response.removal_projections[0].dimension,
        ],
        equal_time_defect=float(
            abs(sources.sink @ (sources.addition + sources.removal) - 1)
        ),
        finite_window_integral=float(np.trapezoid(A, omega)),
        minimum_A=float(A.min()),
        particle_hole_spectral_defect=float(
            np.linalg.norm(A - A[::-1]) / np.linalg.norm(A)
        ),
        total_wall_seconds=perf_counter() - start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    (folder / f"spectrum_N{N}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(
        folder / f"spectrum_N{N}.csv",
        np.column_stack([omega, G.real, G.imag, A]),
        delimiter=",",
        header="omega,Re_G_up,Im_G_up,A_up",
        comments="",
    )
    print(
        f"  done: EOM residual={max(true):.2e}; integral={metrics['finite_window_integral']:.6f}; "
        f"{metrics['total_wall_seconds']:.1f}s",
        flush=True,
    )


def save_convergence(folder):
    """Dump adjacent-N spectral differences on the common frequency grid."""
    spectra = {}
    for N in range(1, 7):
        path = folder / f"spectrum_N{N}.csv"
        if path.exists():
            spectra[N] = np.loadtxt(path, delimiter=",", skiprows=1)[:, 3]
    rows = [
        (N, np.linalg.norm(spectra[N] - spectra[N - 1]) / np.linalg.norm(spectra[N]))
        for N in range(2, 7)
        if N in spectra and N - 1 in spectra
    ]
    np.savetxt(
        folder / "spectral_convergence.csv",
        np.asarray(rows).reshape(-1, 2),
        delimiter=",",
        header="N,relative_L2_adjacent_spectra",
        comments="",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes", nargs="+", type=int, choices=range(1, 7), default=PARAMETERS["modes"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "siam_modes" / "results/cc_reference",
    )
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    check_parameters(args.output)
    for N in args.modes:
        H, D, g = fit_bath(N, args.output)
        if not args.fit_only and not (args.output / f"spectrum_N{N}.csv").exists():
            spectrum(N, H, D, g, args.output)
    if not args.fit_only:
        save_convergence(args.output)

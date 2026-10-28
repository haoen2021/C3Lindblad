"""Full-Coulomb self-consistent 2B and charge-RPA GW on continuum reservoirs.

Spin restricted. U_ijkl=<ij|kl>, reconstructed from opposite-spin terms.
2B includes both direct and exchange diagrams. GW uses the full spatial
Coulomb tensor in a symmetric product basis, with exact instantaneous HF.
No physical regulator or display eta enters the self-consistency equations.
"""

import argparse
from ..parameters import parameters
from pathlib import Path
import time
import numpy as np
from types import SimpleNamespace
from .model import load_input, case_directory, BIASES
from ..shared.cc_io import save_npz, write_json


def continuum_model(phi, beta):
    from c3l.interactions import SparseTwoBodyInteraction

    data = load_input()
    return SimpleNamespace(
        E=np.asarray(data["one_body_matrix"], complex),
        phi=phi,
        beta=beta,
        interaction=SparseTwoBodyInteraction(
            8, tuple(tuple(term) for term in data["canonical_two_body_terms"])
        ),
    )


def coulomb_tensor(model):
    return np.array(
        [
            model.interaction.coefficient(2 * i, 2 * j + 1, 2 * k, 2 * l + 1)
            for i, j, k, l in np.ndindex(4, 4, 4, 4)
        ]
    ).reshape(4, 4, 4, 4)


def screening_vertices(u):
    pair = u.transpose(0, 2, 1, 3).reshape(16, 16)
    values, vectors = np.linalg.eigh(pair)
    if values.min() < -1e-12:
        raise ValueError("Coulomb product matrix is not positive semidefinite")
    q = (vectors[:, values > 1e-12] * np.sqrt(values[values > 1e-12])).T.reshape(
        -1, 4, 4
    )
    np.testing.assert_allclose(q, q.swapaxes(-1, -2), atol=1e-12)
    return q


def hf_selfenergy(rho, u, xp=np):
    """Spin-restricted Hartree (factor 2) minus Fock self-energy."""
    return 2 * xp.einsum("ikjl,lk->ij", u, rho) - xp.einsum("iklj,lk->ij", u, rho)


def born_contraction(a, b, u, xp=np):
    """Second-Born direct minus exchange contraction in the time domain."""
    return 2 * xp.einsum(
        "iklm,npjq,tln,tmp,tqk->tij", u, u, a, a, b, optimize="optimal"
    ) - xp.einsum("iklm,npjq,tlp,tmn,tqk->tij", u, u, a, a, b, optimize="optimal")


def adjoint(a):
    return a.conj().swapaxes(-1, -2)


def embedding(model, z):
    from scipy.special import expit

    data = load_input()
    us = np.asarray(data["leads"]["u_substrate"])
    ut = np.asarray(data["leads"]["tip_geometries"]["center"])
    dr = np.zeros((len(z), 4, 4), complex)
    dl = np.zeros_like(dr)
    dg = np.zeros_like(dr)
    p = parameters("junction")
    width = data["leads"]["half_bandwidth"]
    for shift, strength, vector in ((0.0, p["substrate_coupling_squared"], us),
                                     (model.phi, p["tip_coupling_squared"], ut)):
        zz = np.asarray(z, complex) - shift
        dr += (
            strength
            * ((zz - np.sqrt(zz - width) * np.sqrt(zz + width)) / (2 * width**2))[:, None, None]
            * np.outer(vector, vector.conj())
        )
        x = z.real - shift
        gamma = strength * np.sqrt(np.maximum(0.0, 1 - x * x / width**2)) / width
        f = expit(-model.beta * x)
        projector = np.outer(vector, vector.conj())
        dl += 1j * (gamma * f)[:, None, None] * projector
        dg += -1j * (gamma * (1 - f))[:, None, None] * projector
    return dr, dl, dg


class Solver:
    """Self-consistent Dyson iteration on a uniform frequency grid.

    State = (Sigma_HF, Sigma_R, Sigma_<, Sigma_>).
    dr, dl, dg denote the retarded, lesser, and greater reservoir self-energies.
    The class keeps FFT grids and compiled contractions together between iterations.
    """

    def __init__(self, model, method, n=16384, window=16.0, backend="cpu"):
        import jax

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_platform_name", backend)
        import jax.numpy as xp

        self.xp = xp
        self.method = method
        self.n = n
        self.padding = 3
        self.dw = 2 * window / n
        self.omega = np.linspace(-window, window, n, endpoint=False)
        self.u = xp.asarray(coulomb_tensor(model))
        self.q = xp.asarray(screening_vertices(np.asarray(self.u)))
        self.h = xp.asarray(model.E)
        self.initial_hf = hf_selfenergy(xp.eye(4) * 0.5, self.u, xp)
        self.dr, self.dl, self.dg = map(
            xp.asarray, embedding(model, self.omega.astype(complex))
        )
        self.eye = xp.eye(4)
        self.w = xp.asarray(self.omega)
        m = n * 3
        t = np.arange(m)
        self.reverse = xp.asarray((-t) % m)
        self.theta = xp.asarray(np.where(t < m // 2, 1.0, 0.0))
        self.theta = self.theta.at[0].set(0.5).at[m // 2].set(0.5)
        self.step = jax.jit(self._step)

    def to_time(self, a):
        """f(t) = integral dω exp(-iωt) f(ω)/(2π), with zero padding."""
        xp = self.xp
        pad = self.n
        return (
            xp.fft.fft(
                xp.fft.ifftshift(
                    xp.pad(a, ((pad, pad), *((0, 0) for _ in a.shape[1:]))), axes=0
                ),
                axis=0,
            )
            * self.dw
            / (2 * np.pi)
        )

    def to_frequency(self, a):
        """Inverse transform on the original frequency window."""
        xp = self.xp
        return (
            xp.fft.fftshift(xp.fft.ifft(a, axis=0), axes=0) * (2 * np.pi / self.dw)
        )[self.n : 2 * self.n]

    def causal(self, less, greater):
        return self.to_frequency((greater - less) * self.theta[:, None, None])

    def dyson(self, state):
        """G_R = [ω - h - Δ_R - Σ_HF - Σ_R]^-1; G_</> = G_R(Δ_</> + Σ_</>)G_A."""
        xp = self.xp
        hf, sr, sl, sg = state
        gr = xp.linalg.inv(
            self.w[:, None, None] * self.eye - self.h - self.dr - hf - sr
        )
        gl = gr @ (self.dl + sl) @ adjoint(gr)
        gg = gr @ (self.dg + sg) @ adjoint(gr)
        return gr, (gl - adjoint(gl)) / 2, (gg - adjoint(gg)) / 2

    def correlation(self, gl, gg):
        """Evaluate the 2ndB or GW correlation diagrams from the current G."""
        xp = self.xp
        lt, gt = self.to_time(gl), self.to_time(gg)
        if self.method == "HF":
            return (xp.zeros_like(gl),) * 3
        if self.method == "2B":
            slt = born_contraction(lt, gt[self.reverse], self.u, xp)
            sgt = born_contraction(gt, lt[self.reverse], self.u, xp)
        else:
            q = self.q

            def polar(a, b):
                return -2j * xp.einsum(
                    "aij,tjk,bkl,tli->tab", q, a, q, b, optimize="optimal"
                )

            plt, pgt = polar(lt, gt[self.reverse]), polar(gt, lt[self.reverse])
            pr = self.causal(plt, pgt)
            pl, pg = self.to_frequency(plt), self.to_frequency(pgt)
            # Screen the Coulomb interaction: W_R = (1 - P_R)^-1 in the q basis.
            wr = xp.linalg.inv(xp.eye(q.shape[0]) - pr)
            wl, wg = wr @ pl @ adjoint(wr), wr @ pg @ adjoint(wr)
            slt = 1j * xp.einsum(
                "aik,tkl,blj,tab->tij", q, lt, q, self.to_time(wl), optimize="optimal"
            )
            sgt = 1j * xp.einsum(
                "aik,tkl,blj,tab->tij", q, gt, q, self.to_time(wg), optimize="optimal"
            )
        sr = self.causal(slt, sgt)
        sl, sg = self.to_frequency(slt), self.to_frequency(sgt)
        return sr, (sl - adjoint(sl)) / 2, (sg - adjoint(sg)) / 2

    def _step(self, state):
        xp = self.xp
        gr, gl, gg = self.dyson(state)
        rho = -1j * xp.sum(gl, axis=0) * self.dw / (2 * np.pi)
        hf = hf_selfenergy((rho + adjoint(rho)) / 2, self.u, xp)
        proposed = (hf, *self.correlation(gl, gg))
        defects = xp.array([xp.max(xp.abs(a - b)) for a, b in zip(proposed, state)])
        return proposed, defects

    def solve(self, output, maxiter=2000, mixing=0.1, initial=None):
        xp = self.xp
        state = (
            (self.initial_hf, *(xp.zeros((self.n, 4, 4), complex) for _ in range(3)))
            if initial is None
            else tuple(map(xp.asarray, initial))
        )
        history = []
        for iteration in range(maxiter):
            proposed, defects = self.step(state)
            defects = np.asarray(defects)
            error = float(defects.max())
            history.append(error)
            if iteration % 25 == 0:
                print(
                    f"{self.method} iteration={iteration} defect={error:.9e}",
                    flush=True,
                )
                write_json(
                    output / "progress.json", dict(iteration=iteration, defect=error)
                )
                save_npz(
                    output / f"checkpoint_{self.method}.npz",
                    omega=self.omega,
                    HF=np.asarray(state[0]),
                    Sigma_R=np.asarray(state[1]),
                    Sigma_lesser=np.asarray(state[2]),
                    Sigma_greater=np.asarray(state[3]),
                    iteration=iteration,
                    defect=error,
                )
            if not np.isfinite(error):
                raise FloatingPointError("Nonfinite self-consistency defect")
            if error < 1e-8:
                break
            state = tuple(
                (1 - mixing) * a + mixing * b for a, b in zip(state, proposed)
            )
        _, defects = self.step(state)
        return (
            tuple(map(np.asarray, state)),
            tuple(map(np.asarray, self.dyson(state))),
            np.asarray(defects),
            history,
        )


def equilibrium_green(omega, gr, out):
    """LUMO imaginary-time data from the unbroadened continuum spectrum."""
    p = parameters("junction")
    beta = p["beta"]
    wn = (2 * np.arange(p["response"]["positive_matsubara"]) + 1) * np.pi / beta
    tau = np.linspace(0, beta, p["response"]["tau_points"])
    dw = omega[1] - omega[0]
    spectral = -(gr - adjoint(gr)) / (2j * np.pi)
    giw = np.einsum(
        "nw,wij->nij", dw / (1j * wn[:, None] - omega), spectral, optimize=True
    )
    kernel = -np.exp(-tau[:, None] * omega - np.logaddexp(0, -beta * omega)) * dw
    gtau = np.einsum("tw,wij->tij", kernel, spectral, optimize=True)
    lum = np.asarray(load_input()["fixed_MO_basis"]["columns"])[:, 2]
    save_npz(
        out / "greens.npz",
        wn=wn,
        tau=tau,
        G_iw_LUMO=np.einsum("i,nij,j->n", lum.conj(), giw, lum),
        G_tau_LUMO=np.einsum("i,tij,j->t", lum.conj(), gtau, lum),
        beta=parameters("junction")["beta"],
    )


def tip_current(omega, gl, gg, phi):
    """Spin-summed continuum extraction current, Eq. S89."""
    from scipy.special import expit

    tip = np.asarray(load_input()["leads"]["tip_geometries"]["center"])
    p = parameters("junction")
    width = p["model"]["leads"]["half_bandwidth"]
    gamma = p["tip_coupling_squared"] * np.sqrt(np.maximum(0.0, 1 - (omega - phi)**2 / width**2)) / width
    f = expit(-p["beta"] * (omega - phi))
    ar, aa = -1j * gl / (2 * np.pi), 1j * gg / (2 * np.pi)
    removal = np.einsum("i,tij,j->t", tip.conj(), ar, tip).real
    addition = np.einsum("i,tij,j->t", tip.conj(), aa, tip).real
    return float(np.trapezoid(2 * gamma * ((1 - f) * removal - f * addition), omega))


def run(args):
    start = time.monotonic()
    out = case_directory(args.phi) / args.method
    out.mkdir(parents=True, exist_ok=False)
    model = continuum_model(args.phi, args.beta)
    # Start from Hartree-Fock, then iterate the correlation self-energy.
    hfsolver = Solver(model, "HF", args.n, args.window, args.backend)
    hfstate, _, hfdefects, _ = hfsolver.solve(out, args.maxiter, args.mixing)
    if hfdefects.max() >= 1e-8:
        raise RuntimeError("Continuum HF initialization did not converge")
    solver = Solver(model, args.method, args.n, args.window, args.backend)
    initial = hfstate
    if args.initial:
        with np.load(args.initial) as old:
            initial = [old["HF"]]
            for name in ("Sigma_R", "Sigma_lesser", "Sigma_greater"):
                initial.append(
                    np.stack(
                        [
                            np.interp(
                                solver.omega, old["omega"], old[name][:, i, j].real
                            )
                            + 1j
                            * np.interp(
                                solver.omega, old["omega"], old[name][:, i, j].imag
                            )
                            for i, j in np.ndindex(4, 4)
                        ],
                        axis=1,
                    ).reshape(-1, 4, 4)
                )
    state, greens, defects, history = solver.solve(
        out, args.maxiter, args.mixing, initial
    )
    hf, sr, sl, sg = state
    gr, gl, gg = greens
    a = -(gr - adjoint(gr)) / (2j * np.pi)
    rho = -1j * np.sum(gl, axis=0) * solver.dw / (2 * np.pi)
    keldysh = float(np.max(np.abs(gg - gl - gr + adjoint(gr))))
    collision = float(
        abs(
            np.sum(np.trace(sl @ gg - sg @ gl, axis1=-2, axis2=-1))
            * solver.dw
            / (2 * np.pi)
        )
    )
    checks = dict(
        converged=bool(defects.max() < 1e-8),
        fixed_point_defect=float(defects.max()),
        spectral_sum_rule_error=float(
            np.max(np.abs(a.sum(axis=0) * solver.dw - np.eye(4)))
        ),
        minimum_spectral_eigenvalue=float(np.linalg.eigvalsh(a).min()),
        keldysh_defect=keldysh,
        collision_integral_abs=collision,
        impurity_occupation=float(2 * np.trace(rho).real),
    )
    # Save the physical, unbroadened Green functions needed by the current integral.
    save_npz(
        out / "spectra.npz",
        omega=solver.omega,
        G_R=gr,
        G_lesser=gl,
        G_greater=gg,
        HF=hf,
        Sigma_R=sr,
        Sigma_lesser=sl,
        Sigma_greater=sg,
        beta=args.beta,
        phi=args.phi,
    )
    write_json(
        out / "result.json",
        dict(
            method=args.method,
            beta=args.beta,
            phi=args.phi,
            n=args.n,
            window=args.window,
            padding=3,
            mixing=args.mixing,
            iterations=len(history),
            interaction="full spatial Coulomb tensor; spin-restricted",
            bath="analytic continuum semicircular reservoirs; no fit",
            wall_seconds=time.monotonic() - start,
            **checks,
        ),
    )
    save_npz(
        out / "paper.npz",
        phi=args.phi,
        beta=parameters("junction")["beta"],
        I_tip=tip_current(solver.omega, gl, gg, args.phi),
        converged=checks["converged"],
    )
    if args.phi == 0:
        equilibrium_green(solver.omega, gr, out)
    print(checks, flush=True)
    if not checks["converged"]:
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["GW", "2B"], required=True)
    p.set_defaults(beta=parameters("junction")["beta"])
    p.add_argument("--phi", type=float, choices=BIASES, default=0.0)
    p.add_argument("--n", type=int, default=parameters("junction")["mbpt"]["n"])
    p.add_argument("--window", type=float, default=parameters("junction")["mbpt"]["window"])
    p.add_argument("--device", "--backend", dest="backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--maxiter", type=int, default=parameters("junction")["mbpt"]["maxiter"])
    p.add_argument("--mixing", type=float, default=parameters("junction")["mbpt"]["mixing"])
    p.add_argument("--initial", type=Path)
    run(p.parse_args())

"""Equilibrium CT-HYB/NCA/OCA data for Fig. 4b,c; optional TRIQS dependencies."""

from ..parameters import parameters

import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from .model import ROOT
from .observables import strengths

BASE = ROOT
BETA = parameters("junction")["beta"]


def model():
    data = parameters("junction")["model"]
    h = np.array(data["one_body_matrix"], float)
    us = np.array(data["leads"]["u_substrate"], float)
    ut = np.array(data["leads"]["tip_geometries"]["center"], float)
    # Rotate to the contact eigenbasis, without truncating the four-orbital interaction.
    p = parameters("junction")
    contact = (p["substrate_coupling_squared"] * np.outer(us, us)
               + p["tip_coupling_squared"] * np.outer(ut, ut))
    values, rotation = np.linalg.eigh(contact)
    values[abs(values) < 1e-13] = 0
    assert values.min() >= 0
    return data, h, values, rotation


def operators(data, h, rotation):
    """Construct H0, Hint, densities, and spins in the chosen TRIQS basis."""
    from triqs.operators import c, c_dag, dagger, Operator, n

    spins = ("up", "dn")
    ann = []
    for i in range(4):
        for s in spins:
            ann.append(
                sum((float(rotation[i, a]) * c(s, a) for a in range(4)), Operator())
            )
    create = [dagger(x) for x in ann]
    hint = sum(
        (
            float(v) * create[p] * create[q] * ann[s] * ann[r]
            for p, q, r, s, v in data["canonical_two_body_terms"]
        ),
        Operator(),
    )
    hr = rotation.T @ h @ rotation
    h0 = sum(
        (
            float(hr[i, j]) * c_dag(s, i) * c(s, j)
            for i, j in np.ndindex(4, 4)
            for s in spins
        ),
        Operator(),
    )
    occ = [create[p] * ann[p] for p in range(8)]
    sz = [(occ[2 * i] - occ[2 * i + 1]) / 2 for i in range(4)]
    sp = [create[2 * i] * ann[2 * i + 1] for i in range(4)]
    sm = [dagger(x) for x in sp]
    spin = [
        [sz[i] * sz[j] + (sp[i] * sm[j] + sm[i] * sp[j]) / 2 for j in range(4)]
        for i in range(4)
    ]
    quantum = [sum((n(s, a) for a in range(4)), Operator()) for s in spins]
    return hint, h0, occ, sz, spin, quantum


def bath_quadrature(size=512):
    """Semicircular reservoir: energies and Gauss-Chebyshev integration weights."""
    theta = np.arange(1, size + 1) * np.pi / (size + 1)
    width = parameters("junction")["model"]["leads"]["half_bandwidth"]
    return width * np.cos(theta), 0.5 / (size + 1) * np.sin(theta) ** 2


def run_cthyb(args):
    from triqs_cthyb import Solver
    from triqs.atom_diag import trace_rho_op
    import triqs.utility.mpi as mpi
    import sys, traceback
    from mpi4py import MPI

    def abort_on_error(kind, value, tb):
        traceback.print_exception(kind, value, tb)
        MPI.COMM_WORLD.Abort(1)

    sys.excepthook = abort_on_error
    data, h, weights, rot = model()
    q, _ = np.linalg.qr(np.random.default_rng(27).normal(size=(4, 4)))
    contact = q.T @ np.diag(weights) @ q
    rot = rot @ q
    # Mix the measurement basis so every orbital couples to the bath.
    assert np.min(np.diag(contact)) > 1e-4
    folder = args.output_root / f"group_{args.group:02d}"
    if mpi.is_master_node():
        folder.mkdir(parents=True, exist_ok=False)
    mpi.barrier()
    hint, h0, occ, sz, spin, quantum = operators(data, h, rot)
    s = Solver(
        beta=BETA,
        gf_struct=[("up", 4), ("dn", 4)],
        n_iw=256,
        n_tau=args.n_tau,
        delta_interface=True,
    )
    tau = np.linspace(0, BETA, args.n_tau)
    e, w = bath_quadrature()
    delta = -(w * np.exp(-tau[:, None] * e - np.logaddexp(0, -BETA * e))).sum(axis=1)
    for sp in ("up", "dn"):
        s.Delta_tau[sp].data[:] = delta[:, None, None] * contact
    wn = (2 * np.arange(256) + 1) * np.pi / BETA
    dt = tau[1] - tau[0]
    quad = np.ones(len(tau)) * dt
    quad[[0, -1]] *= 0.5
    # Independent batches retain the statistical information for the paper curves.
    for batch in range(args.batches):
        start = time.monotonic()
        seed = 68852231 + args.group * 1000003 + batch * 10007 + mpi.rank * 97
        s.solve(
            h_int=hint,
            h_loc0=h0,
            partition_method="quantum_numbers",
            quantum_numbers=quantum,
            n_cycles=args.cycles,
            n_warmup_cycles=5000,
            length_cycle=50,
            random_seed=seed,
            max_time=args.seconds,
            measure_density_matrix=True,
            use_norm_as_weight=True,
            measure_G_tau=True,
            measure_pert_order=True,
            move_double=True,
            verbosity=1 if mpi.is_master_node() else 0,
        )
        if mpi.is_master_node():
            gt = np.stack(
                [
                    np.einsum("ia,tab,jb->tij", rot, s.G_tau[sp].data, rot)
                    for sp in ("up", "dn")
                ]
            )
            gi = []
            for g in gt:
                chunks = []
                for i in range(0, len(wn), 32):
                    chunks.append(
                        np.einsum(
                            "nt,tij,t->nij",
                            np.exp(1j * np.outer(wn[i : i + 32], tau)),
                            g,
                            quad,
                            optimize=True,
                        )
                    )
                gi.append(np.concatenate(chunks))
            gi = np.array(gi)
            rho = s.density_matrix
            ad = s.h_loc_diagonalization
            corr = float(np.real(trace_rho_op(rho, spin[0][2], ad)))
            trace = float(sum(np.trace(x).real for x in rho))
            lum = np.asarray(data["fixed_MO_basis"]["columns"])[:, 2]
            np.savez_compressed(
                folder / f"batch_{batch:03d}.npz",
                tau=tau,
                G_tau_LUMO=np.einsum("i,tij,j->t", lum.conj(), gt.mean(axis=0), lum),
                wn=wn,
                G_iw_LUMO=np.einsum("i,nij,j->n", lum.conj(), gi.mean(axis=0), lum),
                beta=BETA,
            )
            record = dict(
                beta=BETA,
                group=args.group,
                batch=batch,
                mpi_ranks=mpi.size,
                solve_status=int(s.solve_status),
                seconds=time.monotonic() - start,
                sign=float(np.real(s.average_sign)),
                trace=trace,
                spin=corr,
                measurements=int(
                    s.perturbation_order["up"].n_data_pts
                    + s.perturbation_order["up"].n_lost_pts
                ),
                dense_measurement_basis=True,
                sampling_cap_seconds=args.seconds,
                seed_base=seed - mpi.rank * 97,
                sampling_core_hours=(time.monotonic() - start) * mpi.size / 3600,
            )

            def mean(op):
                return float(np.real(trace_rho_op(rho, op, ad)))

            nh = occ[0] + occ[1]
            nl = occ[4] + occ[5]
            record.update(strengths(mean(nh), mean(nl), mean(nh * nl), corr))
            (folder / f"batch_{batch:03d}.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            print(
                f"CT-HYB batch {batch}: sign={record['sign']:.5g}, R={record['R']:.6g}",
                flush=True,
            )
        mpi.barrier()
    if mpi.is_master_node():
        print("GROUP_COMPLETE", folder, flush=True)


def run_xca(a):
    from triqs_xca.block_sparse_solver import BlockSparseSolver
    import triqs.utility.mpi as mpi
    from triqs.atom_diag import atomic_density_matrix, trace_rho_op
    from h5 import HDFArchive

    class ContinuumSolver(BlockSparseSolver):
        def eval_pseudo_particle_self_energy(
            self, G, max_order, connected=True, verbose=False
        ):
            # Save the accepted iterate before an expensive OCA diagram evaluation.
            if mpi.is_master_node():
                temporary = self.output / "checkpoint.tmp.h5"
                with HDFArchive(str(temporary), "w") as f:
                    f["G_pp"] = G
                    f["eta"] = self.eta
                    f["order"] = max_order
                os.replace(temporary, self.output / "checkpoint.h5")
            return super().eval_pseudo_particle_self_energy(
                G, max_order, connected, verbose
            )

        def fit_hybridization(self, tol=None, compression=False, verbose=True):
            # Deterministic integration of the analytic continuum; no fitted bath.
            e, w = bath_quadrature(self.quadrature)
            self.set_hybridization_poles_and_coefficients(
                e, (w[:, None, None] * self.contact[None]).astype(complex)
            )
            self.hyb.tol = 0.0
            self.hyb.compression = False
            self.hyb.fit_error = 0.0
            if mpi.is_master_node():
                print(
                    "Analytic semicircle quadrature nodes", self.quadrature, flush=True
                )

    data, h, weights, rotation = model()
    hint, h0, occ, sz, spin, qn = operators(data, h, rotation)
    folder = ROOT / "results" / ("nca" if a.order == 1 else "oca")
    if mpi.is_master_node():
        folder.mkdir(parents=True, exist_ok=True)
    mpi.barrier()
    s = ContinuumSolver(
        H_loc=h0 + hint,
        beta=BETA,
        w_max=20.0,
        eps=a.eps,
        gf_struct=[("up", 4), ("dn", 4)],
        conserved_operators=qn,
    )
    s.output = folder
    s.quadrature = a.quadrature
    s.contact = np.kron(np.eye(2), np.diag(weights))
    # Check semicircle quadrature independently against the analytic Matsubara bath.
    e, w = bath_quadrature(a.quadrature)
    z = 1j * (2 * np.arange(256) + 1) * np.pi / BETA
    width = data["leads"]["half_bandwidth"]
    error = float(
        np.max(
            abs(
                (w / (z[:, None] - e)).sum(axis=1)
                - 0.5 / (z + np.sqrt(z - width) * np.sqrt(z + width))
            )
        )
    )
    assert error < 1e-10, error
    for block, g in s.Delta_tau:
        tau = np.array([float(t) for t in g.mesh])
        delta = -(w * np.exp(-tau[:, None] * e - np.logaddexp(0, -BETA * e))).sum(
            axis=1
        )
        g.data[:] = delta[:, None, None] * np.diag(weights)[None]
    rho_atomic = atomic_density_matrix(s.atom_diag, BETA)
    exact = float(np.real(trace_rho_op(rho_atomic, spin[0][2], s.atom_diag)))
    observed = float(np.real(s.expectation_value(spin[0][2])))
    assert abs(observed - exact) < 100 * a.eps, (observed, exact)
    if mpi.is_master_node():
        print("ATOMIC_VALIDATION", exact, observed, "bath_error", error, flush=True)
    if a.resume:
        from triqs.gfs import make_gf_dlr

        with HDFArchive(a.resume, "r") as f:
            assert (
                float(f["result"]["beta"]) == BETA and float(f["result"]["phi"]) == 0.0
            )
            saved = f["G_pp"]
            s.eta = f["eta"]
        for block, g in s.G:
            old = make_gf_dlr(saved[block])
            for k, t in enumerate(g.mesh):
                g.data[k] = old(float(t))
        if mpi.is_master_node():
            print("RESUMED", a.resume, flush=True)
    s.solve(
        max_order=a.order,
        tol=a.tol,
        maxiter=a.maxiter,
        mix=0.7,
        hyb_comp=False,
        spgf_max_order=a.order,
    )
    if mpi.is_master_node():
        from triqs.gfs import make_gf_dlr, make_gf_imfreq, make_gf_imtime

        gt = []
        gi = []
        for sp in ("up", "dn"):
            dlr = make_gf_dlr(s.G_tau[sp])
            gtime = make_gf_imtime(dlr, n_tau=501)
            gfreq = make_gf_imfreq(dlr, n_iw=256)
            tau = np.array([float(t) for t in gtime.mesh])
            frequencies = np.array([complex(z).imag for z in gfreq.mesh])
            positive = frequencies > 0
            wn = frequencies[positive]
            gt.append(np.einsum("ia,tab,jb->tij", rotation, gtime.data, rotation))
            gi.append(
                np.einsum("ia,nab,jb->nij", rotation, gfreq.data[positive], rotation)
            )
        lum = np.asarray(data["fixed_MO_basis"]["columns"])[:, 2]
        np.savez_compressed(
            folder / "greens.npz",
            tau=tau,
            G_tau_LUMO=np.einsum("i,tij,j->t", lum.conj(), np.mean(gt, axis=0), lum),
            wn=wn,
            G_iw_LUMO=np.einsum("i,nij,j->n", lum.conj(), np.mean(gi, axis=0), lum),
            beta=BETA,
            spgf_order=a.order,
        )

        def mean(op):
            return float(np.real(s.expectation_value(op)))

        corr = mean(spin[0][2])
        nh, nl = occ[0] + occ[1], occ[4] + occ[5]
        result = dict(
            beta=BETA,
            phi=0.0,
            method="NCA" if a.order == 1 else "OCA",
            converged=bool(s.diff_G < a.tol),
            residual=float(s.diff_G),
            iterations=s.n_ppsc_iter,
            **strengths(mean(nh), mean(nl), mean(nh * nl), corr),
        )
        with HDFArchive(str(folder / "state.h5"), "w") as f:
            f["G_pp"] = s.G
            f["eta"] = s.eta
            f["result"] = result
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"{result['method']}: residual={result['residual']:.3e}, R={result['R']:.6g}",
            flush=True,
        )
    if s.diff_G >= a.tol:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="method", required=True)
    p = sub.add_parser("cthyb")
    p.add_argument("--group", type=int, default=0)
    p.add_argument("--cycles", type=int, default=100000000)
    p.add_argument("--seconds", type=int, default=3400)
    p.add_argument("--batches", type=int, default=2)
    p.add_argument("--n-tau", type=int, default=16001)
    p.add_argument("--output-root", type=Path, default=ROOT / "results/cthyb")
    for name, order in (("nca", 1), ("oca", 2)):
        p = sub.add_parser(name)
        p.set_defaults(order=order)
        p.add_argument("--eps", type=float, default=1e-9)
        p.add_argument("--tol", type=float, default=1e-8)
        p.add_argument("--quadrature", type=int, default=256)
        p.add_argument("--maxiter", type=int, default=100)
        p.add_argument("--resume")
    args = parser.parse_args()
    if args.method == "cthyb":
        run_cthyb(args)
    else:
        run_xca(args)

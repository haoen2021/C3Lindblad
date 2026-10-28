"""Run longer CT-HYB Green-function batches with explicit sampling and runtime controls."""

import sys, json, time, argparse, os
from pathlib import Path
import numpy as np
from .common import (
    ROOT,
    BETA,
    BATH_MATRIX,
    ROT,
    bath_quadrature,
    operators,
    expectations,
)
from triqs.atom_diag import trace_rho_op
from triqs_cthyb import Solver
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from mpi4py import MPI
import traceback


def abort_on_error(kind, value, tb):
    traceback.print_exception(kind, value, tb)
    MPI.COMM_WORLD.Abort(1)


sys.excepthook = abort_on_error
parser = argparse.ArgumentParser()
parser.add_argument("group", type=int)
parser.add_argument(
    "--output-root", type=Path, default=ROOT / "results/greens/CTHYB_production"
)
parser.add_argument("--batches", type=int, default=4)
parser.add_argument("--cycles", type=int, default=10000000)
parser.add_argument("--seconds", type=int, default=1800)
args = parser.parse_args()
hint, h0, obs, qn = operators()

e, w = bath_quadrature()
e2, w2 = bath_quadrature(2048)
z = 1j * (2 * np.arange(8) + 1) * np.pi / BETA
np.testing.assert_allclose(
    (w / (z[:, None] - e)).sum(axis=1), (w2 / (z[:, None] - e2)).sum(axis=1), atol=1e-10
)
solver = Solver(
    beta=BETA,
    gf_struct=[("up", 2), ("dn", 2)],
    n_iw=256,
    n_tau=16001,
    delta_interface=True,
)
tau = np.linspace(0, BETA, 16001)
delta = -(w * np.exp(-tau[:, None] * e - np.logaddexp(0, -BETA * e))).sum(axis=1)
for s in ("up", "dn"):
    solver.Delta_tau[s].data[:] = delta[:, None, None] * BATH_MATRIX
group = args.group
out = args.output_root / f"group_{group:02d}"
if mpi.is_master_node():
    out.mkdir(parents=True, exist_ok=True)
mpi.barrier()
for batch in range(args.batches):
    p = out / f"batch_{batch:03d}.json"
    if p.exists():
        raise FileExistsError(p)
    start = time.monotonic()
    solver.solve(
        h_int=hint,
        h_loc0=h0,
        partition_method="quantum_numbers",
        quantum_numbers=qn,
        n_cycles=args.cycles,
        n_warmup_cycles=10000,
        length_cycle=50,
        random_seed=851173 + 1000003 * group + 10007 * batch + 97 * mpi.rank,
        max_time=args.seconds,
        measure_density_matrix=True,
        use_norm_as_weight=True,
        measure_G_tau=True,
        measure_pert_order=True,
        move_double=True,
        verbosity=0,
    )
    rho = solver.density_matrix
    ad = solver.h_loc_diagonalization
    if mpi.is_master_node():
        result = dict(
            method="CTHYB",
            beta=BETA,
            phi=0.0,
            group=group,
            batch=batch,
            convention="decoupling_corrected",
            solve_status=int(solver.solve_status),
            measurements=int(
                solver.perturbation_order["up"].n_data_pts
                + solver.perturbation_order["up"].n_lost_pts
            ),
            sign=float(np.real(solver.average_sign)),
            trace=float(sum(np.trace(x).real for x in rho)),
            seconds=time.monotonic() - start,
            observables=expectations(
                lambda x: float(np.real(trace_rho_op(rho, x, ad))), obs
            ),
        )
        gt = np.mean(
            [
                np.einsum("ia,tab,jb->tij", ROT, solver.G_tau[sp].data, ROT)
                for sp in ("up", "dn")
            ],
            axis=0,
        )
        wn = (2 * np.arange(256) + 1) * np.pi / BETA
        quad = np.full(len(tau), tau[1] - tau[0])
        quad[[0, -1]] *= 0.5
        gi = np.concatenate(
            [
                np.einsum(
                    "nt,tij,t->nij",
                    np.exp(1j * np.outer(wn[k : k + 32], tau)),
                    gt,
                    quad,
                    optimize=True,
                )
                for k in range(0, len(wn), 32)
            ]
        )
        np.savez_compressed(
            out / f"batch_{batch:03d}.npz", tau=tau, G_tau=gt, wn=wn, G_iw=gi, beta=BETA
        )
        result.update(
            mpi_ranks=mpi.size,
            requested_cycles_per_rank=args.cycles,
            maximum_seconds_per_batch=args.seconds,
            tau_points=len(tau),
            sampling="production; independent MPI seeds and groups",
            seed_base=851173 + 1000003 * group + 10007 * batch,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            expected_batches=args.batches,
        )
        coarse_tau = tau[::2]
        coarse_quad = np.full(len(coarse_tau), coarse_tau[1] - coarse_tau[0])
        coarse_quad[[0, -1]] *= 0.5
        coarse_iw = np.einsum(
            "nt,tij,t->nij",
            np.exp(1j * np.outer(wn[:32], coarse_tau)),
            gt[::2],
            coarse_quad,
            optimize=True,
        )
        result["first32_fourier_mesh_difference"] = float(
            np.linalg.norm(gi[:32] - coarse_iw) / np.linalg.norm(gi[:32])
        )
        p.write_text(json.dumps(result, indent=2) + "\n")
        with HDFArchive(str(out / f"batch_{batch:03d}.h5"), "w") as f:
            f["density_matrix"] = rho
            f["atom_diag"] = ad
        print("RESULT", json.dumps(result), flush=True)
    mpi.barrier()

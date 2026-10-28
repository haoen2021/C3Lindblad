"""Generate independent CT-HYB batches of static occupations and spin observables."""

import sys, json, time
import numpy as np
from .common import (
    ROOT,
    BETA,
    BATH_MATRIX,
    bath_quadrature,
    operators,
    expectations,
)
from triqs.atom_diag import trace_rho_op
from triqs_cthyb import Solver
import triqs.utility.mpi as mpi
from h5 import HDFArchive

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
    n_tau=1001,
    delta_interface=True,
)
tau = np.linspace(0, BETA, 1001)
delta = -(w * np.exp(-tau[:, None] * e - np.logaddexp(0, -BETA * e))).sum(axis=1)
for s in ("up", "dn"):
    solver.Delta_tau[s].data[:] = delta[:, None, None] * BATH_MATRIX
group = int(sys.argv[1])
out = ROOT / f"results/cthyb/group_{group:02d}"
if mpi.is_master_node():
    out.mkdir(parents=True, exist_ok=True)
mpi.barrier()
for batch in range(2):
    p = out / f"batch_{batch:03d}.json"
    if p.exists():
        raise FileExistsError(p)
    start = time.monotonic()
    solver.solve(
        h_int=hint,
        h_loc0=h0,
        partition_method="quantum_numbers",
        quantum_numbers=qn,
        n_cycles=100000,
        n_warmup_cycles=10000,
        length_cycle=50,
        random_seed=51173 + 1000003 * group + 10007 * batch + 97 * mpi.rank,
        max_time=240,
        measure_density_matrix=True,
        use_norm_as_weight=True,
        measure_G_tau=False,
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
        p.write_text(json.dumps(result, indent=2) + "\n")
        with HDFArchive(str(out / f"batch_{batch:03d}.h5"), "w") as f:
            f["density_matrix"] = rho
            f["atom_diag"] = ad
        print("RESULT", json.dumps(result), flush=True)
    mpi.barrier()

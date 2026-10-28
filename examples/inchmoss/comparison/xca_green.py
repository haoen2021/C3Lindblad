"""Generate continuum NCA/OCA Green functions and observables using the optional TRIQS XCA solver."""

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
from triqs_xca.block_sparse_solver import BlockSparseSolver
import triqs.utility.mpi as mpi


class Continuum(BlockSparseSolver):
    def fit_hybridization(self, tol=None, compression=False, verbose=True):
        e, w = bath_quadrature(1024)
        self.set_hybridization_poles_and_coefficients(
            e, (w[:, None, None] * np.kron(np.eye(2), BATH_MATRIX)[None]).astype(complex)
        )
        self.hyb.tol = 0.0
        self.hyb.compression = False
        self.hyb.fit_error = 0.0


hint, h0, obs, qn = operators()
order = int(sys.argv[1])
method = "NCA" if order == 1 else "OCA"
out = ROOT / f"results/greens/{method}"
if mpi.is_master_node():
    out.mkdir(parents=True, exist_ok=True)
mpi.barrier()
start = time.monotonic()
s = Continuum(
    H_loc=h0 + hint,
    beta=BETA,
    w_max=40.0,
    eps=1e-10,
    gf_struct=[("up", 2), ("dn", 2)],
    conserved_operators=qn,
)
e, w = bath_quadrature(1024)
for _, g in s.Delta_tau:
    tau = np.array([float(x) for x in g.mesh])
    delta = -(w * np.exp(-tau[:, None] * e - np.logaddexp(0, -BETA * e))).sum(axis=1)
    g.data[:] = delta[:, None, None] * BATH_MATRIX

s.solve(
    max_order=order,
    tol=1e-9,
    maxiter=100,
    mix=0.7,
    hyb_comp=False,
    spgf_max_order=order,
)
if mpi.is_master_node():
    from triqs.gfs import make_gf_dlr, make_gf_imfreq, make_gf_imtime
    from .common import ROT
    from h5 import HDFArchive

    gt = []
    gi = []
    for sp in ("up", "dn"):
        dlr = make_gf_dlr(s.G_tau[sp])
        gtime = make_gf_imtime(dlr, n_tau=501)
        gfreq = make_gf_imfreq(dlr, n_iw=256)
        tau = np.array([float(t) for t in gtime.mesh])
        freq = np.array([complex(z).imag for z in gfreq.mesh])
        positive = freq > 0
        wn = freq[positive]
        gt.append(np.einsum("ia,tab,jb->tij", ROT, gtime.data, ROT))
        gi.append(np.einsum("ia,nab,jb->nij", ROT, gfreq.data[positive], ROT))
    np.savez_compressed(
        out / "greens.npz",
        tau=tau,
        G_tau=np.mean(gt, axis=0),
        wn=wn,
        G_iw=np.mean(gi, axis=0),
        beta=BETA,
    )
    with HDFArchive(str(out / "state.h5"), "w") as archive:
        archive["G_tau"] = s.G_tau
        archive["G_pp"] = s.G
        archive["eta"] = s.eta
    result = dict(
        method=method,
        beta=BETA,
        phi=0.0,
        converged=bool(s.diff_G < 1e-9),
        residual=float(s.diff_G),
        seconds=time.monotonic() - start,
        observables=expectations(lambda x: float(np.real(s.expectation_value(x))), obs),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
if s.diff_G >= 1e-9:
    raise SystemExit(2)

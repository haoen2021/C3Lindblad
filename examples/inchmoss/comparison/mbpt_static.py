"""Equal-time cumulants from saved continuum spectra; no repeated Dyson solves."""

import json
import numpy as np
from scipy.special import expit
from .common import ROOT, ROT

from ...shared.spin_cumulant import cumulant, spin_components, adjoint

source = ROOT.parents[1] / "results/inchmoss_fig6_mbpt_v1"
(ROOT / "results").mkdir(parents=True, exist_ok=True)
u = np.zeros((2, 2, 2, 2))
u[1, 1, 1, 1] = 8.0
for method in ("2B", "GW"):
    with np.load(source / f"{method}_refined.npz") as z:
        w = z["omega"]
        g = ROT @ z["G_R"] @ ROT.T
    f = expit(-0.1 * w)[:, None, None]
    disc = g - adjoint(g)
    gl = -f * disc
    gg = (1 - f) * disc
    rho = -1j * gl.sum(axis=0) * (w[1] - w[0]) / (2 * np.pi)
    direct, checks = cumulant(w, gl, gg, u, method, padding=8)
    result = dict(
        method=method, beta=0.1, bath="continuum", observables={}, checks=checks
    )
    for label, rotation in [("rotated", np.eye(2)), ("original", ROT)]:
        density = rotation @ rho @ rotation.T
        tensor = np.einsum(
            "ia,jb,kc,ld,abcd->ijkl", rotation, rotation, rotation, rotation, direct
        )
        v = spin_components(density, tensor, method, i=0, j=1)
        result["observables"][label] = dict(
            occupations=(2 * density.diagonal().real).tolist(),
            double_occupations=[
                float((density[i, i] ** 2 + tensor[i, i, i, i]).real) for i in range(2)
            ],
            spin_correlation=float(v["dot"].real),
            spin_zz=float(v["zz"].real),
            spin_density_order=float(v["dot_density_order"].real),
            spin_isotropy_defect=float(v["spin_isotropy_defect"].real),
        )
    (ROOT / f"results/{method}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)

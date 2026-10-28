"""Imaginary-axis integrals of saved unbroadened continuum MBPT spectra."""

import json
import numpy as np
from .common import ROOT, BETA

wn = (2 * np.arange(256) + 1) * np.pi / BETA
tau = np.linspace(0, BETA, 501)
for name in ("2B", "GW"):
    path = ROOT.parents[1] / f"results/inchmoss_fig6_mbpt_v1/{name}_refined.npz"
    with np.load(path) as f:
        w = f["omega"].copy()
        g = f["G_R"].copy()
        defect = f["defect_history"][-1].copy()
    A = -(g - g.conj().swapaxes(-1, -2)) / (2j * np.pi)
    dw = w[1] - w[0]
    gi = np.concatenate(
        [
            np.einsum(
                "nw,wij->nij", dw / (1j * wn[k : k + 16, None] - w), A, optimize=True
            )
            for k in range(0, len(wn), 16)
        ]
    )
    gt = np.concatenate(
        [
            np.einsum(
                "tw,wij->tij",
                -dw * np.exp(-tau[k : k + 16, None] * w - np.logaddexp(0, -BETA * w)),
                A,
                optimize=True,
            )
            for k in range(0, len(tau), 16)
        ]
    )
    out = ROOT / f"results/greens/{name}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "greens.npz", wn=wn, G_iw=gi, tau=tau, G_tau=gt, beta=BETA
    )
    result = dict(
        source=str(path),
        sum_rule_error=float(abs(A.sum(axis=0) * dw - np.eye(2)).max()),
        last_defect=defect.tolist(),
        method="Equilibrium spectral integrals; original unbroadened continuum data; window [-64,64)",
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, result, flush=True)

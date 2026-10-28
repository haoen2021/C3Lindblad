"""InchMOSS imaginary-axis response: saved S135 recurrence, rebuilt S13."""

import argparse
import json, time
from pathlib import Path
from dataclasses import replace
import numpy as np
from .common import ROOT, BETA
from ...shared.matsubara import matsubara_to_tau
from ...shared.response import reduced_gmres
from .. import run as benchmark
from c3l.cc.factorized import factorized_cc_problem
from c3l.cc.component_tensor import (
    ComponentStaticProblem,
    component_static_residual,
    ComponentEOMProblem,
    component_eom_context,
    component_eom_sources,
)
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.component_jax import JaxRetainedEOMOperator
from c3l.green.resident_arnoldi import ResidentArnoldi
import jax
import jax.numpy as jnp
from ...shared.saved_reference import restore_modes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
args = parser.parse_args()
jax.config.update("jax_enable_x64", True)

out = ROOT / "results/greens"
out.mkdir(parents=True, exist_ok=True)
start = time.monotonic()
wn = (2 * np.arange(256) + 1) * np.pi / BETA
tau = np.linspace(0, BETA, 501)
shifts = 1j * wn
archive = Path("results/multiorbital_full_q5_gpu_v1/inchmoss_corrected")
for rank in ("S135", "S13"):
    folder = out / rank
    folder.mkdir(exist_ok=True)
    G = np.zeros((256, 2, 2), complex)
    records = []
    if rank == "S13":
        print("STAGE restoring validated fresh static state", flush=True)
        base = benchmark.build_cc_problem(corrected_convention=True)
        with np.load(ROOT / "results/cc_latest.npz") as f:
            amp = f["amplitudes"].copy()
            ref = replace(
                base.reference,
                up=restore_modes(base.reference.up, f["reference_V_up"]),
                down=restore_modes(base.reference.down, f["reference_V_down"]),
            )
        oracle = factorized_cc_problem(
            base.model, "CCSD", reference=ref, build_next_rank_masks=False
        )
        static = ComponentStaticProblem.compile(oracle)
        residual = float(
            np.linalg.norm(
                oracle.tilde.pack(
                    component_static_residual(static, amp).retained, check=False
                )
            )
        )
        assert residual < 1e-8, residual
        ep = eom_ccsd_problem(oracle, response_spin=0, build_next_rank_masks=False)
        context = component_eom_context(
            ComponentEOMProblem.compile(ep, static, charge=1), amp
        )
        sources = tuple(component_eom_sources(context, site=i) for i in (0, 1))
        op = JaxRetainedEOMOperator.from_cpu(context, platform=args.device, production=True)
        sinks = np.vstack([s.sink for s in sources])
        print("STATIC", residual, "dimension", op.dimension, flush=True)
    for i in (0, 1):
        if rank == "S135":
            path = archive / f"sources/q135_source_{i}.npz"
            with np.load(path) as f:
                h = f["hessenberg"].copy()
                sink = f["sink_projection"].copy()
                norm = float(f["source_norm"])
        else:
            rhs = sources[i].addition + sources[i].removal
            norm = float(np.linalg.norm(rhs))
            path = folder / f"source_{i}.npz"
            cap = min(256, op.dimension)
            chain = ResidentArnoldi(op.device_matvec, rhs, sinks, cap, device=op.device)
            for target in sorted({min(128, cap), cap}):
                chain.advance(target)
                h = chain.hessenberg[:chain.current + 1, :chain.current]
                sink = chain.projected[:, :chain.current]
                _, estimated = reduced_gmres(h, norm, shifts)
                if np.max(estimated) <= 1e-12 or chain.breakdown:
                    break
        m = h.shape[1]
        y, res = reduced_gmres(h, norm, shifts)
        G[:, :, i] = y @ sink[:, :m].T
        small = min(128, m)
        ys, rs = reduced_gmres(h[: small + 1, :small], norm, shifts)
        prefix = ys @ sink[:, :small].T
        rec = dict(
            source=i,
            m=m,
            source_norm=norm,
            max_reduced_relative_residual=float(res.max()),
            prefix_change=float(
                np.linalg.norm(G[:, :, i] - prefix) / np.linalg.norm(G[:, :, i])
            ),
            input=str(path),
        )
        if rank == "S13":
            index = int(np.argmax(res))
            vector = chain.reconstruct(y[index])
            rec["true_relative_residual"] = float(
                jnp.linalg.norm(jax.device_put(rhs, chain.device) - (shifts[index] * vector - op.device_matvec(vector)))
            ) / norm
            rec["orthogonalization_backend"] = f"{chain.device.platform}_resident_cgs2"
            np.savez_compressed(
                folder / f"source_{i}.npz",
                hessenberg=h,
                sink_projection=sink[:, :m],
                source_norm=norm,
            )
            del chain, vector
        records.append(rec)
        print(rank, rec, flush=True)
    gt = matsubara_to_tau(wn, G, BETA, tau)
    coarse = matsubara_to_tau(wn[:128], G[:128], BETA, tau)
    np.savez_compressed(
        folder / "greens.npz", wn=wn, G_iw=G, tau=tau, G_tau=gt, beta=BETA
    )
    metadata = dict(
        rank=rank,
        records=records,
        tau_cutoff_change=float(abs(gt - coarse).max()),
        seconds=time.monotonic() - start,
        static_origin="archived accepted static state; no reload required"
        if rank == "S135"
        else "fresh validated static checkpoint cc_latest.npz",
        interpretation="Equilibrium KMS reconstruction of finite-PM retarded response; not a directly measured thermal CC correlator.",
    )
    (folder / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("DONE", rank, metadata["tau_cutoff_change"], flush=True)

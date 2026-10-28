"""Half-filled Kanamori CCSD and GPU-resident S13/S135 with cap 1024.

Run through the release launcher, with data in an external working directory:

    python /path/to/C3L/examples/run.py --work-dir /path/to/work \
        kanamori.half_filling --beta 8 --bath /path/to/bath_fit.npz \
        --output results/kanamori_half_filling_beta8

Use --beta 64 with its matching five-mode bath for the low-temperature case.
Both response spaces run by default; --rank S13 or --rank S135 selects one.
The default impurity chemical potential is (3U-5J)/2, equal to 2.5 for
U=2 and J=0.2. The shared model parameters and other examples are unchanged.

"""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import run as model_builder
from ..inchmoss.gpu_timing import build_response, source_response
from ..shared.cc_io import save_npz, write_json


def build_model(bath_path, convention="particle_hole_symmetric"):
    from c3l.pseudomode import PseudomodeRealization, matrix_pseudomode_model

    with np.load(bath_path, allow_pickle=False) as saved:
        branches = tuple(
            PseudomodeRealization(saved[f"H_{branch}"], saved[f"damping_{branch}"], saved[f"g_{branch}"])
            for branch in ("lesser", "greater")
        )
    if any(branch.size != 5 for branch in branches):
        raise ValueError("Expected the archived five-mode bath branches")
    return matrix_pseudomode_model(
        model_builder.impurity_one_body(convention),
        model_builder.orbital_hybridization_matrix(),
        *branches, model_builder.interaction(), units="t",
    )


def reconstruct(model, raw, shifts):
    from c3l.spectra.reconstruction import matrix_ordered_dyson_reconstruction

    identity = np.eye(2)
    one_body = model.E[:2, :2]
    bath_matrix = model.E[2:, 2:] - 1j * (model.gamma_loss + model.gamma_gain)[2:, 2:]
    coupling = model.E[:2, 2:]
    bath_identity = np.eye(bath_matrix.shape[0])
    delta_pm = np.asarray([
        coupling @ np.linalg.solve(shift * bath_identity - bath_matrix, coupling.conj().T)
        for shift in shifts
    ])
    delta_continuum = model_builder.bethe_retarded(shifts)[:, None, None] * model_builder.orbital_hybridization_matrix()
    result = matrix_ordered_dyson_reconstruction(raw, delta_pm, delta_continuum)
    self_energy = shifts[:, None, None] * identity - one_body - delta_pm - np.linalg.inv(raw)
    return result.retarded, self_energy


def load_static_checkpoint(path, reference_fock, reference_gamma):
    with np.load(path, allow_pickle=False) as saved:
        if not bool(saved["converged"]):
            raise ValueError("Static checkpoint is not converged")
        for key, expected in (("reference_fock", reference_fock), ("reference_gamma", reference_gamma)):
            if not np.allclose(saved[key], expected, atol=1e-10, rtol=1e-10):
                raise ValueError(f"Static checkpoint reference mismatch: {key}")
        return saved["amplitudes"].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=int, choices=(8, 64), required=True)
    parser.add_argument("--bath", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--one-body-convention", choices=("chemical_potential", "particle_hole_symmetric"),
                        default="particle_hole_symmetric")
    parser.add_argument("--rank", nargs="+", choices=("S13", "S135"), default=["S13", "S135"])
    parser.add_argument("--cap", type=int, choices=(1024,), default=1024,
                        help="fixed reproduction cap per source (1024)")
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume-static", action="store_true")
    checkpoint.add_argument("--static-from", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume_static)
    import jax
    from c3l.cc.component_jax import solve_component_ccsd_jax
    from c3l.cc.component_tensor import ComponentStaticProblem, component_static_residual
    from c3l.cc.factorized import factorized_cc_problem
    from c3l.cc.newton_krylov import NewtonKrylovPolicy
    from c3l.cc.observables import factorized_ccsd_covariance
    from ..shared.matsubara import matsubara_to_tau
    from ..shared.multiorbital_response import spectral_matrix

    jax.config.update("jax_enable_x64", True)
    device = jax.devices("gpu")[0]
    started = perf_counter()
    model = build_model(args.bath, convention=args.one_body_convention)
    reference = model_builder._regulated_reference(model, 0.05)
    static = ComponentStaticProblem.compile(factorized_cc_problem(
        model, "CCSD", reference=reference, build_next_rank_masks=False,
    ))
    policy = NewtonKrylovPolicy(
        tolerance=1e-10, maximum_newton_iterations=60, gmres_restart=320,
        gmres_maximum_restart_cycles=4, maximum_line_search_steps=16,
        minimum_gmres_relative_tolerance=1e-8, maximum_gmres_relative_tolerance=.5,
        initial_gmres_relative_tolerance=.1, forcing_strategy="eisenstat_walker",
    )
    record = dict(
        state="STATIC", beta=args.beta, U=2., J=.2, mu=float(-model.E[0, 0].real),
        one_body_convention=args.one_body_convention,
        interaction="full_kanamori_spin_exchange_and_pair_hopping",
        bath=str(args.bath.resolve()), modes_per_branch=5,
        device=device.device_kind, basis_storage="gpu_only", policy=policy.to_dict(),
        setup_seconds=perf_counter() - started, static_dimension=static.oracle.num_amplitudes,
        ranks={},
    )
    write_json(args.output / "summary.json", record)
    print(f"beta={args.beta}: {device.device_kind}, static dimension={static.oracle.num_amplitudes}", flush=True)
    static_started = perf_counter()
    if args.resume_static or args.static_from:
        source = args.static_from or args.output / "static.npz"
        amplitudes = load_static_checkpoint(source, reference.fock, reference.gamma)
        record["static"] = dict(reused=True, checkpoint=str(source.resolve()), seconds=perf_counter() - static_started)
        if args.static_from:
            save_npz(args.output / "static.npz", amplitudes=amplitudes, converged=True,
                     reference_fock=reference.fock, reference_gamma=reference.gamma)
    else:
        def checkpoint(iteration, amplitudes, residual):
            save_npz(args.output / "static_latest.npz", amplitudes=amplitudes, residual=residual)
            print(f"Newton {iteration}: residual={residual:.12e}", flush=True)

        solution = solve_component_ccsd_jax(
            static, policy=policy, platform="gpu", production=True,
            accepted_step_callback=checkpoint,
        )
        amplitudes = solution.amplitudes
        record["static"] = dict(
            converged=bool(solution.converged), residual=solution.packed_residual_norm,
            iterations=solution.iterations, seconds=perf_counter() - static_started,
            history=[asdict(step) for step in solution.history],
        )
        save_npz(args.output / "static.npz", amplitudes=amplitudes,
                 converged=solution.converged, residual=solution.packed_residual_norm,
                 reference_fock=reference.fock, reference_gamma=reference.gamma)
        write_json(args.output / "summary.json", record)
        if not solution.converged or not np.isfinite(solution.packed_residual_norm) or solution.packed_residual_norm > 1e-10:
            raise RuntimeError("Static solve did not converge; response is not launched")
    retained = component_static_residual(static, amplitudes).retained
    fresh_residual = float(np.linalg.norm(static.oracle.tilde.pack(retained, check=False)))
    if not np.isfinite(fresh_residual) or fresh_residual > 1e-8:
        raise RuntimeError(f"Independent static residual failed: {fresh_residual}")
    covariance = factorized_ccsd_covariance(reference, static.oracle.cluster_basis, amplitudes)
    record["fresh_static_residual"] = fresh_residual
    record["impurity_occupations"] = np.diag(covariance)[:4].real.tolist()
    record["covariance_eigenvalues"] = np.linalg.eigvalsh((covariance + covariance.conj().T) / 2).tolist()
    record["state"] = "RESPONSE"
    write_json(args.output / "summary.json", record)
    omega = np.linspace(-4., 4., 801)
    matsubara = (2 * np.arange(256) + 1) * np.pi / args.beta
    shifts = np.r_[omega + .05j, 1j * matsubara]
    tau = np.linspace(0., args.beta, 1001)
    for rank in args.rank:
        rank_started = perf_counter()
        operator, sources = build_response(static, amplitudes, rank, device="gpu")
        setup_seconds = perf_counter() - rank_started
        cap = min(args.cap, operator.dimension)
        sinks = np.stack([source.sink for source in sources])
        columns, checks = [], []
        for site, source in enumerate(sources):
            print(f"Computing {rank} source {site} to cap {cap}; diagnostics follow projected solves", flush=True)
            column, metrics = source_response(
                operator, source.addition + source.removal, sinks, shifts,
                len(omega), cap, args.output / f"{rank}_source{site}.npz",
            )
            columns.append(column)
            checks.append(metrics)
        raw = np.stack(columns, axis=2)
        reconstructed, self_energy = reconstruct(model, raw, shifts)
        converged = all(check["converged"] for check in checks)
        save_npz(
            args.output / f"{rank}.npz", omega=omega, eta=.05, omega_n=matsubara, tau=tau,
            G_R_raw_PM=raw[:len(omega)], G_R=reconstructed[:len(omega)],
            A_raw_PM=spectral_matrix(raw[:len(omega)]), A=spectral_matrix(reconstructed[:len(omega)]),
            G_iw_raw_PM=raw[len(omega):], G_iw=reconstructed[len(omega):],
            G_tau=matsubara_to_tau(matsubara, reconstructed[len(omega):], args.beta, tau),
            Sigma_R=self_energy[:len(omega)], Sigma_iw=self_energy[len(omega):],
            converged=converged,
        )
        record["ranks"][rank] = dict(
            converged=converged, dimension=operator.dimension, cap=cap, sources=checks,
            setup_seconds=setup_seconds, seconds=perf_counter() - rank_started,
        )
        write_json(args.output / "summary.json", record)
        print(f"{rank} completed: converged={converged}", flush=True)
        del operator, sources, sinks
        gc.collect()
    record["state"] = "COMPLETE" if all(item["converged"] for item in record["ranks"].values()) else "RESPONSE_CAP_REACHED"
    record["total_seconds"] = perf_counter() - started
    write_json(args.output / "summary.json", record)
    print(f"{record['state']}: {record['total_seconds']:.3f}s", flush=True)


if __name__ == "__main__":
    main()

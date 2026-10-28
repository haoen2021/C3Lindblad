"""Sweep SIAM interactions with JAX CCSD and device-resident S13/S135 response."""

import argparse
from dataclasses import asdict
import gc
import logging
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from . import run
from ..shared.cc_io import save_npz, write_json


def static_policy(restart):
    from c3l.cc.newton_krylov import NewtonKrylovPolicy

    return NewtonKrylovPolicy(
        tolerance=1e-10, maximum_newton_iterations=20, gmres_restart=restart,
        gmres_maximum_restart_cycles=2, maximum_line_search_steps=16,
        forcing_strategy="eisenstat_walker", initial_gmres_relative_tolerance=.1,
        maximum_gmres_relative_tolerance=.3, minimum_gmres_relative_tolerance=1e-8,
    )


def load_seed(path):
    from c3l.cc.component_tensor import component_static_residual
    from ..shared.anderson_response import portable_checkpoint_amplitudes

    with np.load(path, allow_pickle=False) as saved:
        interaction = float(saved["U"])
        amplitudes = np.asarray(saved["amplitudes"], dtype=complex)
    problem = run.build_static_problem(interaction)
    if amplitudes.shape != (problem.oracle.num_amplitudes,) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("Invalid seed amplitudes")
    residual = float(np.linalg.norm(component_static_residual(problem, amplitudes).retained))
    if residual > 1e-8:
        aligned = portable_checkpoint_amplitudes(problem, amplitudes)
        if not aligned.resolved:
            raise RuntimeError("Cannot align the archived seed coordinates")
        amplitudes = aligned.amplitudes
        residual = float(np.linalg.norm(component_static_residual(problem, amplitudes).retained))
    if not np.isfinite(residual) or residual > 1e-8:
        raise RuntimeError(f"Seed does not pass its fresh static residual check: {residual}")
    print(f"Seed U={interaction:g}: fresh residual={residual:.3e}", flush=True)
    return problem, amplitudes, dict(U=interaction, residual=residual, path=str(path.resolve()))


def continue_static(seed_problem, amplitudes, interactions, policy, output, *, device="gpu", solver=None):
    if solver is None:
        from c3l.cc.component_jax import solve_component_ccsd_jax
        solver = solve_component_ccsd_jax
    records = []
    for interaction in interactions:
        folder = output / f"U{interaction:g}"
        folder.mkdir()
        started = perf_counter()
        problem = run.build_static_problem(interaction)
        for spin in ("up", "down"):
            np.testing.assert_allclose(
                getattr(problem.oracle.reference, spin).V,
                getattr(seed_problem.oracle.reference, spin).V, atol=1e-12, rtol=0,
                err_msg="Continuation requires the same half-filled reference coordinates",
            )

        def accepted(iteration, values, residual):
            save_npz(folder / "static_latest.npz", U=interaction, amplitudes=values, residual=residual)
            print(f"U={interaction:g}, Newton {iteration}: residual={residual:.3e}", flush=True)

        solution = solver(
            problem, amplitudes, policy=policy, platform=device, production=True,
            accepted_step_callback=accepted,
        )
        residual = solution.packed_residual_norm
        converged = bool(
            solution.converged and np.isfinite(residual) and residual <= policy.tolerance
            and np.isfinite(solution.residual_norm) and solution.residual_norm <= 1e-8
        )
        record = dict(
            U=interaction, seconds=perf_counter() - started, converged=converged,
            packed_residual=residual, retained_residual=solution.residual_norm,
            iterations=solution.iterations, effective_gmres_restart=min(policy.gmres_restart, problem.oracle.num_variables),
            history=[asdict(step) for step in solution.history],
        )
        write_json(folder / "static.json", record)
        if not converged:
            raise RuntimeError(f"Static solve failed at U={interaction:g}: {residual}; try smaller interaction steps")
        amplitudes = solution.amplitudes.copy()
        save_npz(folder / "static.npz", U=interaction, amplitudes=amplitudes, residual=residual)
        records.append(record)
        print(f"U={interaction:g}: static converged in {record['seconds']:.3f}s", flush=True)
    return problem, amplitudes, records


def build_response(static, amplitudes, rank, *, device="gpu"):
    from c3l.cc.component_tensor import ComponentEOMProblem, component_eom_context, component_eom_sources
    from c3l.cc.component_jax import JaxRetainedEOMOperator
    from c3l.cc.eom import eom_ccsd_problem
    from c3l.cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator
    from c3l.cc.s135_jax import JaxFullS135ResponseOperator

    eom = eom_ccsd_problem(static.oracle, response_spin=0, build_next_rank_masks=False)
    context = component_eom_context(ComponentEOMProblem.compile(eom, static, charge=1), amplitudes)
    source = component_eom_sources(context, site=static.oracle.model.impurity_site)
    if rank == "S13":
        return JaxRetainedEOMOperator.from_cpu(context, platform=device, production=True, projection_tile_size=16384), source
    native = FullS135ResponseOperator(CompiledFullS135Action(context), s5_tile_size=16384)
    operator = JaxFullS135ResponseOperator.from_cpu(native, platform=device, production=True)
    embedded = {}
    for name in ("addition", "removal", "sink"):
        embedded[name] = np.zeros(operator.dimension, dtype=complex)
        embedded[name][:native.p_dimension] = getattr(source, name)
    return operator, SimpleNamespace(**embedded)


def resident_response(operator, source, shifts, cap, output):
    import jax
    import jax.numpy as jnp
    import scipy.linalg
    from c3l.green.resident_arnoldi import ResidentArnoldi

    started = perf_counter()
    rhs = source.addition + source.removal
    norm = np.linalg.norm(rhs)
    capacity = min(cap, operator.dimension)
    resident = ResidentArnoldi(
        operator.device_matvec, rhs, source.sink[None, :], capacity,
        device=getattr(operator, "device", None),
    )
    device_rhs = jax.device_put(rhs, resident.device)
    previous, history = None, []
    for target in sorted({min(capacity, value) for value in (64, 128, 256, 512, capacity)}):
        resident.advance(target)
        dimension = resident.current
        hessenberg = resident.hessenberg[:dimension + 1, :dimension]
        triangular, unitary = scipy.linalg.schur(hessenberg[:dimension], output="complex", check_finite=False)
        reduced_rhs = unitary[0].conj() * norm
        identity = np.eye(dimension)
        coefficients = np.array([
            unitary @ scipy.linalg.solve_triangular(shift * identity - triangular, reduced_rhs, check_finite=False)
            for shift in shifts
        ])
        green = coefficients @ resident.projected[0, :dimension]
        estimated = abs(hessenberg[dimension, dimension - 1] * coefficients[:, -1]) / norm
        if not np.all(np.isfinite(green)) or not np.all(np.isfinite(estimated)):
            raise FloatingPointError("Nonfinite EOM response or residual")
        indices = sorted({int(np.argmax(estimated)), *np.linspace(0, len(shifts) - 1, min(5, len(shifts)), dtype=int)})
        true_residuals = []
        for index in indices:
            vector = resident.reconstruct(coefficients[index])
            residual = float(jnp.linalg.norm(device_rhs - (shifts[index] * vector - operator.device_matvec(vector))) / norm)
            if not np.isfinite(residual):
                raise FloatingPointError("Nonfinite true EOM residual")
            true_residuals.append(residual)
        change = None if previous is None else float(np.linalg.norm(green - previous) / max(np.linalg.norm(green), 1e-30))
        absolute = None if previous is None else float(np.max(abs(green - previous)))
        stable = resident.breakdown or dimension == operator.dimension or change is not None and max(change, absolute) < 2e-4
        converged = bool(stable and max(float(estimated.max()), max(true_residuals)) <= 1e-8)
        history.append(dict(
            dimension=dimension, reduced_residual=float(estimated.max()),
            true_residual=max(true_residuals), true_residual_indices=indices,
            relative_change=change, absolute_change=absolute, converged=converged,
        ))
        print(f"{output.name}: {dimension} vectors, true residual={max(true_residuals):.3e}, converged={converged}", flush=True)
        if converged or resident.breakdown:
            break
        previous = green
    record = dict(
        dimension=dimension, operator_dimension=operator.dimension, capacity=capacity,
        seconds=perf_counter() - started, converged=converged, basis_storage=f"{resident.device.platform}_only",
        action_seconds=sum(resident.action_seconds), orthogonalization_seconds=sum(resident.orthogonalization_seconds),
        equal_time_defect=float(abs(source.sink @ rhs - 1)), history=history,
    )
    write_json(output.with_suffix(".json"), record)
    save_npz(
        output.with_suffix(".npz"), omega=shifts.real, eta=shifts.imag,
        retarded=green, spectral=-green.imag / np.pi, converged=converged,
        hessenberg=hessenberg, sink_projections=resident.projected[:, :dimension],
    )
    if not converged:
        raise RuntimeError(f"EOM did not converge at {capacity} vectors; diagnostics saved in {output}.json")
    return green, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--U", type=float, nargs="+", default=list(run.case().U), help="Increasing interactions; static and EOM run at every point (default: siam_ed.U in parameters.json)")
    parser.add_argument("--seed", type=Path, default=run.ROOT / "checkpoints/U2.npz")
    parser.add_argument("--rank", nargs="+", choices=("S13", "S135"), default=["S13", "S135"])
    parser.add_argument("--gmres-restart", type=int, default=320)
    parser.add_argument("--krylov-cap", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.all(np.isfinite(args.U)) or min(args.U) < 0 or np.any(np.diff(args.U) <= 0):
        parser.error("--U must contain finite nonnegative, strictly increasing values")
    if min(args.gmres_restart, args.krylov_cap) < 1:
        parser.error("Krylov dimensions must be positive")
    if len(set(args.rank)) != len(args.rank):
        parser.error("Specify each response rank only once")
    import jax

    jax.config.update("jax_enable_x64", True)
    device = jax.devices(args.device)[0]
    args.output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("jax")
    logger.addHandler(logging.FileHandler(args.output / "compilation.log"))
    logger.propagate = False
    jax.config.update("jax_log_compiles", True)
    started = perf_counter()
    seed_problem, initial, seed = load_seed(args.seed)
    if args.U[0] < seed["U"]:
        parser.error("The first continuation point must be at least the seed interaction")
    policy = static_policy(args.gmres_restart)
    record = dict(
        beta=run.case().beta, modes_per_correlation=run.case().N, seed=seed,
        bath=str((run.ROOT / "data/fit.npz").resolve()), device=device.device_kind,
        interactions=args.U, ranks=args.rank, policy=policy.to_dict(), static=[], responses={},
    )
    write_json(args.output / "summary.json", record)
    frequency = run.config()["frequency"]
    omega = np.linspace(frequency["minimum"], frequency["maximum"], frequency["points"])
    shifts = omega + 1j * frequency["eta"]
    problem, amplitudes = seed_problem, initial
    for interaction in args.U:
        problem, amplitudes, static_records = continue_static(
            problem, amplitudes, [interaction], policy, args.output, device=args.device,
        )
        record["static"].extend(static_records)
        token = f"U{interaction:g}"
        record["responses"][token] = {}
        write_json(args.output / "summary.json", record)
        for rank in args.rank:
            setup_started = perf_counter()
            operator, source = build_response(problem, amplitudes, rank, device=args.device)
            setup_seconds = perf_counter() - setup_started
            _, response = resident_response(operator, source, shifts, args.krylov_cap, args.output / token / rank)
            record["responses"][token][rank] = dict(setup_seconds=setup_seconds, **response)
            write_json(args.output / "summary.json", record)
            del operator, source
            gc.collect()
    record["total_seconds"] = perf_counter() - started
    write_json(args.output / "summary.json", record)
    print(f"Completed in {record['total_seconds']:.3f}s; results: {args.output}", flush=True)


if __name__ == "__main__":
    main()

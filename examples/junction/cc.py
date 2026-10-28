"""Fig. 4: stationary CCSD state, then its one-electron EOM response.

static: Hartree-Fock reference -> CCSD amplitudes -> R and tip current.
eom:    saved CCSD state -> (z - A)x = b -> G_LUMO(tau) or addition/removal spectra.
The Hamiltonian is in parameters.json; fitted baths and results are external.
"""

from ..parameters import parameters as paper_parameters

import argparse
import hashlib
import json
import time
import numpy as np
from .model import (
    ROOT,
    assemble_model,
    contact_currents,
    case_directory,
    load_input,
    input_fingerprint,
    BIASES,
    RESPONSE_BIASES,
)
from .observables import cc_observables, strengths
from ..shared.cc_io import fresh_residual, write_json, save_npz
from ..shared.saved_reference import build_problem
from c3l.ci.reference import full_hartree_fock_reference
from c3l.gaussian_mean_field import solve_ghf
from c3l.cc.component_jax import solve_component_ccsd_jax
from c3l.cc.newton_krylov import NewtonKrylovPolicy


# Numerical parameters used for the paper.
PARAMETERS = paper_parameters("junction")
beta = PARAMETERS["beta"]
cc_tolerance = PARAMETERS["static"]["tolerance"]
cc_acceptance = PARAMETERS["static"]["acceptance"]  # Relaxed threshold only after the final recovery attempt.
eom_tolerance = PARAMETERS["response"]["tolerance"]
response_change = PARAMETERS["response"]["change_tolerance"]
# (GMRES vectors, initial relative tolerance), tried in this order.
gmres_attempts = PARAMETERS["static"]["gmres_attempts"]
interaction_steps = PARAMETERS["static"]["interaction_steps"]


def run_static(phi, *, device="gpu"):
    start = time.monotonic()
    config = paper_parameters("junction")
    b = config["bath_files"]["4"]
    bath = ROOT / b["path"]
    assert hashlib.sha256(bath.read_bytes()).hexdigest() == b["sha256"]
    out = case_directory(phi) / "cc_N4"
    out.mkdir(parents=True, exist_ok=True)
    # A restart must refer to the same Hamiltonian, bias, and fitted bath.
    parameters = dict(
        beta=beta, phi=phi, input_sha256=input_fingerprint(), bath_sha256=b["sha256"]
    )
    if (out / "configuration.json").exists():
        saved = json.loads((out / "configuration.json").read_text())
        for key, value in parameters.items():
            assert saved[key] == value, f"Restart input changed: {key}"
    if (out / "solve.json").exists() and (out / "observables.json").exists():
        saved_solve = json.loads((out / "solve.json").read_text())
        if (
            saved_solve.get("converged")
            and json.loads((out / "observables.json").read_text())["checks"]["passed"]
        ):
            original_execution = saved_solve.get(
                "execution", dict(backend="unknown", device="unknown")
            )
            print(
                "Reusing cached saved static state "
                f"(original backend={original_execution['backend']}, "
                f"device={original_execution['device']}, "
                f"seconds={saved_solve.get('seconds', 'unknown')}); "
                f"requested device={device}: no new device timing. "
                "Response will independently verify it",
                flush=True,
            )
            return dict(saved_solve, execution=original_execution, cached=True)
    execution = dict(backend="jax", device=device)
    if not (out / "configuration.json").exists():
        write_json(out / "configuration.json", dict(parameters, execution=execution))
    physical = assemble_model(bath, phi)
    # 1. Find the spin-restricted Hartree-Fock reference and keep its orbital basis.
    reference_file = out / "reference.npz"
    if reference_file.exists():
        with np.load(reference_file) as f:
            fock = f["fock"].copy()
            gamma_ref = f["gamma_reference"].copy()
    else:
        hf = solve_ghf(
            np.kron(physical.E, np.eye(2)),
            physical.interaction,
            np.kron(physical.gamma_loss, np.eye(2)),
            np.kron(physical.gamma_gain, np.eye(2)),
            restricted=True,
            tolerance=1e-10,
            max_iterations=600,
            damping=0.5,
            terminal_least_squares=True,
        )
        assert hf.converged and hf.self_consistency_residual < 1e-8
        fock, gamma_ref = hf.fock, hf.gamma
        reference = full_hartree_fock_reference(physical, fock, gamma_ref)
        save_npz(
            reference_file,
            fock=fock,
            gamma_reference=gamma_ref,
            reference_V_up=reference.up.V,
            reference_V_down=reference.down.V,
            beta=beta,
            phi=phi,
        )

    # 2. Solve R(T) = 0 at a given interaction strength, with a saved restart.
    def solve_stage(scale, route, initial):
        """Scale the interaction by s, keeping the HF orbital coordinates fixed."""
        stage = out / f"{route}/s{int(round(100 * scale)):03d}"
        stage.mkdir(parents=True, exist_ok=True)
        print(f"CCSD: {route}, interaction scale = {scale:g}", flush=True)
        p, component = build_problem(
            physical, fock, gamma_ref, scale=scale, checkpoint=reference_file
        )
        if (stage / "accepted.npz").exists():
            with np.load(stage / "accepted.npz") as f:
                initial = f["amplitudes"].copy()
            res = fresh_residual(p, component, initial)
            assert res < 1e-8
            return (
                True,
                initial,
                p,
                component,
                json.loads((stage / "result.json").read_text()),
            )
        accepted = False
        attempts = []
        for attempt, (budget, forcing) in enumerate(gmres_attempts):
            folder = stage / f"attempt_{attempt}"
            folder.mkdir(exist_ok=True)
            if (folder / "latest.npz").exists():
                with np.load(folder / "latest.npz") as f:
                    for key, actual in (
                        ("reference_V_up", p.reference.up.V),
                        ("reference_V_down", p.reference.down.V),
                    ):
                        np.testing.assert_array_equal(f[key], actual)
                    assert float(f["scale"]) == scale
                    initial = f["amplitudes"].copy()
            policy = NewtonKrylovPolicy(
                tolerance=cc_tolerance,
                maximum_newton_iterations=16,
                gmres_restart=budget,
                gmres_maximum_restart_cycles=1,
                minimum_gmres_relative_tolerance=1e-8,
                maximum_gmres_relative_tolerance=0.5,
                initial_gmres_relative_tolerance=forcing,
                forcing_strategy="eisenstat_walker",
                maximum_line_search_steps=12,
            )

            def checkpoint(iteration, amplitudes, residual):
                save_npz(
                    folder / "latest.npz",
                    amplitudes=amplitudes,
                    residual=residual,
                    iteration=iteration,
                    reference_V_up=p.reference.up.V,
                    reference_V_down=p.reference.down.V,
                    scale=scale,
                    beta=beta,
                    phi=phi,
                )
                print(f"  iteration {iteration}: |R| = {residual:.3e}", flush=True)

            solution = solve_component_ccsd_jax(
                component,
                initial,
                policy=policy,
                platform=device,
                production=True,
                accepted_step_callback=checkpoint,
            )
            initial = solution.amplitudes
            res = fresh_residual(p, component, initial)
            record = dict(
                scale=scale,
                attempt=attempt,
                budget=budget,
                residual=res,
                iterations=solution.iterations,
            )
            attempts.append(record)
            accepted = res < cc_tolerance or (
                attempt == len(gmres_attempts) - 1 and res < cc_acceptance
            )
            if accepted:
                break
        summary = dict(
            scale=scale,
            converged=accepted,
            strict_converged=bool(res < cc_tolerance),
            acceptance_tolerance=cc_tolerance if res < cc_tolerance else cc_acceptance,
            residual=res,
            attempts=attempts,
        )
        write_json(stage / "result.json", summary)
        if accepted:
            save_npz(stage / "accepted.npz", amplitudes=initial, scale=scale)
        return accepted, initial, p, component, summary

    # Equilibrium uses continuation; at finite bias try the full interaction first.
    records = []
    accepted = False
    if phi != 0:
        accepted, initial, p, component, record = solve_stage(1.0, "direct", None)
        records.append(dict(route="direct", **record))
    if not accepted:
        initial = None
        for scale in interaction_steps:
            accepted, initial, p, component, record = solve_stage(
                scale, "homotopy", initial
            )
            records.append(dict(route="homotopy", **record))
            if not accepted:
                write_json(
                    out / "solve.json",
                    dict(
                        converged=False,
                        continuation=records,
                        residual=record["residual"],
                        execution=execution,
                    ),
                )
                raise RuntimeError(
                    f"CCSD did not converge at interaction scale {scale:g}"
                )
    # 3. Save the stationary amplitudes, then evaluate the paper observables.
    res = fresh_residual(p, component, initial)
    save_npz(
        out / "static.npz",
        amplitudes=initial,
        reference_V_up=p.reference.up.V,
        reference_V_down=p.reference.down.V,
        fock=fock,
        gamma_reference=gamma_ref,
        beta=beta,
        phi=phi,
    )
    solve_record = dict(
        converged=True,
        strict_converged=bool(res < cc_tolerance),
        acceptance_tolerance=cc_tolerance if res < cc_tolerance else cc_acceptance,
        residual=res,
        continuation=records,
        seconds=time.monotonic() - start,
        execution=execution,
    )
    write_json(out / "solve.json", solve_record)
    gamma, obs = cc_observables(p, initial)
    curr = contact_currents(physical, gamma)
    # Pauli bounds, number fluctuations, and steady-state charge conservation.
    cv = gamma[:8, :8]
    ev = np.linalg.eigvalsh((cv + cv.conj().T) / 2)
    nn = np.asarray(obs["density_pairs"])
    di = np.diag(cv)
    cov = nn - di[:, None] * di[None, :]
    ce = np.linalg.eigvalsh((cov + cov.conj().T) / 2)
    continuity = max(
        abs(curr["I_tip"] + curr["I_substrate"]),
        abs(curr["I_tip"] - curr["F_tip"]),
        abs(curr["I_substrate"] - curr["F_substrate"]),
    )
    checks = dict(
        one_particle_min=float(ev.min()),
        one_particle_max=float(ev.max()),
        number_covariance_min=float(ce.min()),
        continuity_error=continuity,
    )
    checks["passed"] = bool(
        ev.min() >= -1e-8
        and ev.max() <= 1 + 1e-8
        and ce.min() >= -1e-8
        and continuity < 1e-8
        and obs["maximum_observable_imaginary_defect"] < 1e-8
    )
    save_npz(out / "state.npz", gamma=gamma)
    nn_hl = nn[np.ix_([0, 1], [4, 5])].sum().real
    transfer = strengths(
        obs["site_occupations"][0].real,
        obs["site_occupations"][2].real,
        nn_hl,
        obs["h1_l1_spin_correlation"].real,
    )
    save_npz(
        out / "paper.npz",
        beta=beta,
        phi=phi,
        R=transfer["R"],
        I_tip=-curr["I_tip"],
        I_substrate=-curr["I_substrate"],
        converged=checks["passed"],
        residual=res,
    )
    write_json(
        out / "observables.json",
        dict(
            beta=beta,
            phi=phi,
            currents=curr,
            checks=checks,
            residual=res,
            seconds=time.monotonic() - start,
        ),
    )
    print(
        f"CCSD residual = {res:.3e}; physical checks passed: {checks['passed']}",
        flush=True,
    )
    if not checks["passed"]:
        raise RuntimeError(
            "The CCSD state fails the occupation or current-conservation checks"
        )
    return dict(solve_record, cached=False)


def solve_source(operator, rhs, sinks, shifts, folder, cap):
    """Solve shifted systems with a resident basis and check a full residual."""
    import jax
    import jax.numpy as jnp
    from ..shared.response import reduced_gmres
    from c3l.green.resident_arnoldi import ResidentArnoldi

    cap = min(cap, operator.dimension)
    prefixes = sorted({min(prefix, cap) for prefix in (32, 64, 128, 256, 384, 512, 768, 1024, cap)})
    norm = float(np.linalg.norm(rhs))
    if norm == 0:
        return np.zeros((len(shifts), len(sinks)), complex), dict(converged=True, dimension=0)
    chain = ResidentArnoldi(operator.device_matvec, rhs, sinks, cap,
                            device=getattr(operator, "device", None))
    device_rhs = jax.device_put(rhs, chain.device)
    previous = None
    history = []
    for target in prefixes:
        chain.advance(target)
        dimension = chain.current
        coefficients, estimated = reduced_gmres(
            chain.hessenberg[:dimension + 1, :dimension], norm, shifts
        )
        column = coefficients @ chain.projected[:, :dimension].T
        index = int(np.argmax(estimated))
        vector = chain.reconstruct(coefficients[index])
        true = float(jnp.linalg.norm(
            device_rhs - (shifts[index] * vector - operator.device_matvec(vector))
        )) / norm
        if not np.isfinite(true) or not np.all(np.isfinite(estimated)) or not np.all(np.isfinite(column)):
            raise FloatingPointError("Nonfinite resident response or full residual")
        relative, absolute = None, None
        closed = chain.breakdown or dimension == operator.dimension
        stable = closed
        if previous is not None:
            change = column - previous
            relative = float(np.linalg.norm(change) / max(np.linalg.norm(column), 1e-30))
            absolute = float(np.max(abs(change)))
            stable = closed or (relative < response_change and absolute < response_change)
        good = bool(max(float(estimated.max()), true) <= eom_tolerance and stable)
        history.append(dict(
            dimension=dimension, reduced_residual=float(estimated.max()),
            true_residual=true, relative_change=relative, absolute_change=absolute,
            converged=good,
        ))
        print(f"{folder.name}: m={dimension}, residual={max(float(estimated.max()), true):.3e}, converged={good}", flush=True)
        if good or closed:
            break
        previous = column
    return column, dict(converged=good, dimension=dimension, history=history,
                        orthogonalization_backend=f"{chain.device.platform}_resident_cgs2")


def run_eom(args):
    """Use the same CCSD state for all four orbital sources and both EOM ranks."""
    from ..shared.response import response_context, response_operator
    from ..shared.matsubara import matsubara_to_tau

    base = case_directory(args.phi)
    case = base / "cc_N4"
    out = (base / "greens" if args.phi == 0 else base) / f"CC_N4_{args.rank}"
    out.mkdir(parents=True, exist_ok=True)
    config = paper_parameters("junction")
    b = config["bath_files"]["4"]
    bath = ROOT / b["path"]
    assert hashlib.sha256(bath.read_bytes()).hexdigest() == b["sha256"]
    # 1. Reload the state and verify its residual in the saved HF coordinates.
    saved_inputs = json.loads((case / "configuration.json").read_text())
    if saved_inputs["input_sha256"] != input_fingerprint():
        raise ValueError("Saved CC state belongs to different junction parameters")
    acceptance = json.loads((case / "solve.json").read_text())
    assert acceptance["converged"]
    assert json.loads((case / "observables.json").read_text())["checks"]["passed"]
    with np.load(case / "static.npz") as saved:
        amp, fock, gamma = (
            saved[k].copy() for k in ("amplitudes", "fock", "gamma_reference")
        )
        assert float(saved["phi"]) == args.phi and float(saved["beta"]) == beta
    physical = assemble_model(bath, args.phi)
    cc_problem, static = build_problem(
        physical, fock, gamma, checkpoint=case / "static.npz"
    )
    residual = fresh_residual(cc_problem, static, amp)
    assert residual < acceptance["acceptance_tolerance"], residual
    context, sources = response_context(cc_problem, static, amp)
    operator, sources = response_operator(context, sources, getattr(args, "device", "gpu"), args.rank)
    cap = min(args.cap or PARAMETERS["response"]["krylov_caps"][args.rank], operator.dimension)
    # 2. Equilibrium: Matsubara frequencies. Bias: the physical real axis (eta=0).
    if args.phi == 0:
        omega = (2 * np.arange(PARAMETERS["response"]["positive_matsubara"]) + 1) * np.pi / beta
        shifts = 1j * omega
        branches = ("total",)
    else:
        omega = np.unique(
            np.round(
                np.concatenate([np.arange(*segment) for segment in
                                PARAMETERS["response"]["real_frequency_segments"]]),
                12,
            )
        )
        shifts = omega.astype(complex)
        branches = ("addition", "removal")
    parameters = dict(
        phi=args.phi,
        beta=beta,
        rank=args.rank,
        cap=cap,
        static_sha256=hashlib.sha256((case / "static.npz").read_bytes()).hexdigest(),
        bath_sha256=b["sha256"],
        residual_tolerance=eom_tolerance,
        grid_sha256=hashlib.sha256(shifts.tobytes()).hexdigest(),
    )
    record_path = out / "configuration.json"
    if record_path.exists():
        assert json.loads(record_path.read_text()) == parameters
    else:
        write_json(record_path, parameters)
    # 3. Build each source response; addition/removal stay separate for currents.
    sinks = np.vstack([s.sink for s in sources])
    responses = {branch: np.zeros((len(shifts), 4, 4), complex) for branch in branches}
    records = []
    for si, source in enumerate(sources):
        for branch in branches:
            rhs = (
                source.addition + source.removal
                if branch == "total"
                else getattr(source, branch)
            )
            column, record = solve_source(
                operator,
                rhs,
                sinks,
                shifts,
                out / "sources" / f"source_{si}_{branch}",
                cap,
            )
            responses[branch][:, :, si] = column
            records.append(dict(source=si, branch=branch, **record))
    # 4. Save only the imaginary-time LUMO curve or the transport response.
    converged = all(record["converged"] for record in records)
    if args.phi == 0:
        tau = np.linspace(0, beta, PARAMETERS["response"]["tau_points"])
        giw = responses["total"]
        gtau = matsubara_to_tau(omega, giw, beta, tau)
        lum = np.asarray(load_input()["fixed_MO_basis"]["columns"])[:, 2]
        save_npz(
            out / "response.npz",
            wn=omega,
            tau=tau,
            G_iw_LUMO=np.einsum("i,nij,j->n", lum.conj(), giw, lum),
            G_tau_LUMO=np.einsum("i,tij,j->t", lum.conj(), gtau, lum),
            converged=converged,
            beta=beta,
            phi=0.0,
        )
    else:
        save_npz(
            out / "branches.npz",
            omega=omega,
            R_add=responses["addition"],
            R_rem=responses["removal"],
            converged=converged,
            beta=beta,
            phi=args.phi,
        )
    write_json(
        out / "result.json",
        dict(
            converged=converged,
            physical_converged=converged,
            static_residual=residual,
            sources=records,
        ),
    )
    if not converged:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("static", "eom"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--phi", type=float, choices=BIASES, default=0.0)
    parser.add_argument("--rank", choices=("S13", "S135"), default="S13")
    parser.add_argument("--cap", type=int)
    args = parser.parse_args()
    if args.cap is not None and args.cap < 1:
        parser.error("--cap must be positive")
    if args.stage == "eom" and args.phi not in RESPONSE_BIASES:
        parser.error("EOM uses the paper response biases 0, -0.2, ..., -1.6")
    if args.stage == "static":
        run_static(args.phi, device=args.device)
    else:
        run_eom(args)


if __name__ == "__main__":
    main()

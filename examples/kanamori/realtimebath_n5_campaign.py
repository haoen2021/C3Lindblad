"""Beta-8 Kanamori EOM-CC with the RealTimeBath five-mode sectors."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from ..parameters import parameters
import shutil
from time import perf_counter
from typing import Any, Literal

import numpy as np

from c3l.cc.component_tensor import ComponentStaticProblem
from c3l.cc.factorized import factorized_cc_problem
from c3l.pseudomode import PseudomodeRealization, matrix_pseudomode_model

from . import run as benchmark
from ..shared.multiorbital_response import CaseSpec


ROOT = Path.cwd()
FIT_ROOT = ROOT / "kanamori/inputs/beta8_N5"
FIT_PATH = FIT_ROOT / "bath_fit.npz"
OUTPUT = ROOT / "results/kanamori_full_mu1_beta8_realtimebath_N5_cc_v2"
SCRATCH = OUTPUT / "scratch"
STATIC_TOLERANCE = 1.0e-8
Space = Literal["S13", "S135"]


@dataclass(frozen=True, slots=True)
class RealtimeBathFit:
    path: Path
    lesser: PseudomodeRealization
    greater: PseudomodeRealization


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_realtimebath_fit(path: Path = FIT_PATH) -> RealtimeBathFit:
    """Load the validated C3L-convention RealTimeBath export."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as saved:
        lesser = PseudomodeRealization(
            np.asarray(saved["H_lesser"], dtype=np.complex128),
            np.asarray(saved["damping_lesser"], dtype=np.complex128),
            np.asarray(saved["g_lesser"], dtype=np.complex128),
        )
        greater = PseudomodeRealization(
            np.asarray(saved["H_greater"], dtype=np.complex128),
            np.asarray(saved["damping_greater"], dtype=np.complex128),
            np.asarray(saved["g_greater"], dtype=np.complex128),
        )
    if lesser.size != 5 or greater.size != 5:
        raise RuntimeError("the campaign requires five modes in each sector")
    return RealtimeBathFit(source, lesser, greater)


def build_model():
    """Build the paper-matched beta-8 finite model with the new bath."""

    fit = load_realtimebath_fit()
    return matrix_pseudomode_model(
        benchmark.impurity_one_body("chemical_potential"),
        benchmark.orbital_hybridization_matrix(),
        fit.lesser,
        fit.greater,
        benchmark.interaction(),
        units="t",
    )


def build_static_problem() -> ComponentStaticProblem:
    """Compile the CCSD equations for the RealTimeBath model."""

    model = build_model()
    reference = benchmark._regulated_reference(model, 0.05)
    problem = factorized_cc_problem(
        model,
        "CCSD",
        build_next_rank_masks=False,
        reference=reference,
    )
    return ComponentStaticProblem.compile(problem)


def campaign_spec() -> CaseSpec:
    """Return the GitHub beta-8 real-axis grid with the new checkpoint."""

    omega = np.linspace(*parameters("kanamori")["frequency"])
    eta = 0.05
    return CaseSpec(
        name="kanamori_full_mu1_beta8_realtimebath_N5",
        family="kanamori",
        beta=parameters("kanamori")["beta"][0],
        modes_per_correlation=parameters("kanamori")["modes_per_branch"],
        omega=omega,
        eta=eta,
        checkpoint=OUTPUT / "static.npz",
        retained_reference=None,
        run_q13=True,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_static(*, device: str = "gpu") -> dict[str, Any]:
    """Solve the physical endpoint directly with the production JAX solver."""

    from c3l.cc.component_jax import solve_component_ccsd_jax
    from ..shared.multiorbital_response import configure_jax

    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_jax(OUTPUT / "cache/static", device=device)
    started = perf_counter()
    problem = build_static_problem()

    def checkpoint(iteration: int, amplitudes: np.ndarray, residual: float) -> None:
        np.savez_compressed(
            OUTPUT / "static_latest.npz",
            amplitudes=amplitudes,
            iteration=np.asarray(iteration),
            packed_residual=np.asarray(residual),
        )
        print(f"{device.upper()} Newton step {iteration}: residual={residual:.12e}", flush=True)

    solution = solve_component_ccsd_jax(
        problem,
        policy=benchmark.direct_static_policy(),
        platform=device,
        production=True,
        accepted_step_callback=checkpoint,
    )
    if not solution.converged or solution.packed_residual_norm > STATIC_TOLERANCE:
        raise RuntimeError(
            f"direct s=1 {device.upper()} static solve failed: {solution.packed_residual_norm:.6e}"
        )
    np.savez_compressed(
        OUTPUT / "static.npz",
        amplitudes=solution.amplitudes,
        variables=solution.variables,
        converged=np.asarray(solution.converged),
        packed_residual=np.asarray(solution.packed_residual_norm),
        residual_norm=np.asarray(solution.residual_norm),
        reference_residual=np.asarray(solution.reference_residual),
    )
    fit_record = json.loads((FIT_ROOT / "fit.json").read_text(encoding="utf-8"))
    record = {
        "state": "COMPLETE",
        "route": f"direct_s1_from_fresh_full_hf_{device}",
        "production_backend": device,
        "beta": 8.0,
        "modes_per_sector": 5,
        "realtimebath_commit": fit_record["realtimebath_commit"],
        "realtimebath_method": fit_record["method"],
        "realtimebath_optimize": fit_record["optimize"],
        "bath_sha256": _sha256(FIT_PATH),
        "converged": bool(solution.converged),
        "packed_residual": float(solution.packed_residual_norm),
        "residual_norm": float(solution.residual_norm),
        "iterations": int(solution.iterations),
        "residual_evaluations": int(solution.residual_evaluations),
        "jvp_evaluations": int(solution.jvp_evaluations),
        "solver_wall_seconds": float(solution.wall_seconds),
        "end_to_end_seconds": float(perf_counter() - started),
        "complex_amplitudes": int(problem.oracle.num_amplitudes),
        "independent_real_variables": int(problem.oracle.num_variables),
        "policy": benchmark.direct_static_policy().to_dict(),
        "history": [asdict(step) for step in solution.history],
    }
    _write_json(OUTPUT / "static.json", record)
    print(json.dumps(record), flush=True)
    return record


def _load_context():
    from c3l.cc.component_tensor import (
        ComponentEOMProblem,
        component_eom_context,
        component_eom_sources,
        component_static_residual,
    )
    from c3l.cc.eom import eom_ccsd_problem

    problem = build_static_problem()
    with np.load(OUTPUT / "static.npz", allow_pickle=False) as saved:
        if not bool(saved["converged"].item()):
            raise RuntimeError("the static checkpoint is not converged")
        amplitudes = np.asarray(saved["amplitudes"], dtype=np.complex128)
    residual = component_static_residual(problem, amplitudes)
    packed = float(
        np.linalg.norm(problem.oracle.tilde.pack(residual.retained, check=False))
    )
    if packed > STATIC_TOLERANCE:
        raise RuntimeError(f"fresh static residual failed: {packed:.6e}")
    eom = eom_ccsd_problem(
        problem.oracle,
        response_spin=0,
        build_next_rank_masks=False,
    )
    response = ComponentEOMProblem.compile(eom, problem, charge=1)
    context = component_eom_context(response, amplitudes)
    sources = tuple(component_eom_sources(context, site=site) for site in (0, 1))
    return problem, context, sources, packed


def run_eom(space: Space, *, device: str = "gpu") -> dict[str, Any]:
    """Construct one response-space recurrence for both orbital sources."""

    from ..shared.multiorbital_response import (
        _build_full_operator,
        _resource_gate,
        _run_combined_source,
        _source_metadata,
        configure_jax,
    )

    if space not in ("S13", "S135"):
        raise ValueError("space must be S13 or S135")
    spec = campaign_spec()
    stage = space.lower()
    scratch = SCRATCH / stage
    scratch.mkdir(parents=True, exist_ok=True)
    configure_jax(OUTPUT / "cache" / stage, device=device)
    started = perf_counter()
    _, context, sources, static_residual = _load_context()
    setup_started = perf_counter()
    compiled = None
    if space == "S13":
        from c3l.cc.component_jax import JaxRetainedEOMOperator

        operator = JaxRetainedEOMOperator.from_cpu(
            context,
            platform=device,
            production=True,
        )
        dimensions = {"total": int(operator.dimension)}
    else:
        operator, compiled, _ = _build_full_operator(context, device=device)
        dimensions = {
            "P": int(operator.p_dimension),
            "Q5": int(operator.s5_dimension),
            "total": int(operator.dimension),
        }
    setup_seconds = float(perf_counter() - setup_started)
    device = getattr(getattr(operator, "device", None), "platform", device)
    resources = _resource_gate(operator.dimension, device=getattr(operator, "device", None))
    records_list: list[dict[str, Any]] = []
    for source in sources:
        source_root = scratch / f"source_{source.site}_{device}"
        result = _run_combined_source(
            operator, source, sources, spec, source_root,
            provenance=f"kanamori-full-mu1-beta8-rtb-n5:{stage}:{device}",
        )
        records_list.append(result)
    records = tuple(records_list)
    metadata = [_source_metadata(item) for item in records]
    if not all(item["observable_converged"] for item in metadata):
        raise RuntimeError(f"{space} did not meet observable convergence")
    destination = OUTPUT / "sources"
    destination.mkdir(parents=True, exist_ok=True)
    for source, item in zip(sources, records, strict=True):
        origin = scratch / f"source_{source.site}_{device}" / "compact.npz"
        shutil.copy2(origin, destination / f"{stage}_source_{source.site}.npz")
    runtime = asdict(operator.runtime)
    memory = operator.device_memory_record()
    record = {
        "state": "COMPLETE",
        "space": space,
        "static_packed_residual": static_residual,
        "dimensions": dimensions,
        "compiled_group_count": None if compiled is None else int(compiled.group_count),
        "compiled_storage_bytes": None
        if compiled is None
        else int(compiled.storage_bytes),
        "setup_seconds": setup_seconds,
        "sources": metadata,
        "resources": resources,
        "runtime": runtime,
        "production_backend": device,
        "peak_device_memory": memory,
        "wall_seconds": float(perf_counter() - started),
    }
    _write_json(OUTPUT / f"{stage}.json", record)
    print(json.dumps(record), flush=True)
    return record


def _load_source(space: Space, site: int) -> dict[str, np.ndarray]:
    path = OUTPUT / "sources" / f"{space.lower()}_source_{site}.npz"
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in saved.files}


def _projected_matrix(space: Space, shifts: np.ndarray) -> np.ndarray:
    from ..shared.reconstruction import projected_transfer

    matrix = np.empty((shifts.size, 2, 2), dtype=np.complex128)
    for site in (0, 1):
        source = _load_source(space, site)
        dimension = int(source["final_m"].item())
        matrix[:, :, site] = projected_transfer(
            source["hessenberg"],
            source["sink_projection"],
            float(source["source_norm"].item()),
            shifts,
            dimension,
        )
    return matrix


def _reconstruction_inputs(shifts: np.ndarray):
    from c3l.gaussian_mean_field import quadratic_frequency_response
    from c3l.quadratic_ness import finite_pm_covariance

    model = build_model()
    covariance, lyapunov = finite_pm_covariance(
        model.E, model.gamma_loss, model.gamma_gain
    )
    g0_pm = quadratic_frequency_response(
        model.E,
        model.gamma_loss,
        model.gamma_gain,
        covariance,
        shifts,
        orbitals=(0, 1),
    ).retarded
    identity = np.eye(2, dtype=np.complex128)
    one_body = benchmark.impurity_one_body("chemical_potential")
    hybridization = benchmark.orbital_hybridization_matrix()
    g0_continuum = np.asarray(
        [
            np.linalg.solve(
                shift * identity
                - one_body
                - hybridization * benchmark.bethe_retarded(shift),
                identity,
            )
            for shift in shifts
        ]
    )
    inverse_pm = np.asarray([np.linalg.solve(value, identity) for value in g0_pm])
    inverse_continuum = np.asarray(
        [np.linalg.solve(value, identity) for value in g0_continuum]
    )
    delta_pm = shifts[:, None, None] * identity - one_body - inverse_pm
    delta_continuum = shifts[:, None, None] * identity - one_body - inverse_continuum
    return delta_pm, delta_continuum, float(lyapunov)


def _reconstruct(
    raw: np.ndarray, shifts: np.ndarray
) -> tuple[np.ndarray, float, float]:
    from c3l.spectra.reconstruction import matrix_ordered_dyson_reconstruction

    delta_pm, delta_continuum, lyapunov = _reconstruction_inputs(shifts)
    result = matrix_ordered_dyson_reconstruction(raw, delta_pm, delta_continuum)
    return (
        np.asarray(result.retarded),
        float(np.max(result.ordering_difference)),
        lyapunov,
    )


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / np.linalg.norm(right))


def assemble() -> dict[str, Any]:
    """Evaluate both recurrences and save real/Matsubara/time data."""

    from ..shared.matsubara import matsubara_to_tau
    from ..shared.multiorbital_response import spectral_matrix

    spec = campaign_spec()
    real_shifts = spec.omega.astype(np.complex128) + 1j * spec.eta
    raw_real = {
        space: _projected_matrix(space, real_shifts) for space in ("S13", "S135")
    }
    reconstructed_real: dict[str, np.ndarray] = {}
    real_reconstruction: dict[str, Any] = {}
    spectra: dict[str, np.ndarray] = {}
    for space in ("S13", "S135"):
        reconstructed, ordering, lyapunov = _reconstruct(raw_real[space], real_shifts)
        reconstructed_real[space] = reconstructed
        spectra[space] = spectral_matrix(reconstructed)
        real_reconstruction[space] = {
            "ordering_difference_maximum": ordering,
            "finite_PM_G0_lyapunov_residual": lyapunov,
        }

    omega_n = (2.0 * np.arange(256) + 1.0) * np.pi / 8.0
    matsubara_shifts = 1j * omega_n
    tau = np.linspace(0.0, parameters("kanamori")["beta"][0], 1001)
    tau_over_beta = tau / 8.0
    raw_iw = {
        space: _projected_matrix(space, matsubara_shifts) for space in ("S13", "S135")
    }
    reconstructed_iw: dict[str, np.ndarray] = {}
    green_tau: dict[str, np.ndarray] = {}
    matsubara_reconstruction: dict[str, Any] = {}
    for space in ("S13", "S135"):
        reconstructed, ordering, lyapunov = _reconstruct(
            raw_iw[space], matsubara_shifts
        )
        reconstructed_iw[space] = reconstructed
        green_tau[space] = matsubara_to_tau(omega_n, reconstructed, 8.0, tau)
        matsubara_reconstruction[space] = {
            "ordering_difference_maximum": ordering,
            "finite_PM_G0_lyapunov_residual": lyapunov,
        }

    np.savez_compressed(
        OUTPUT / "compact.npz",
        omega=spec.omega,
        eta=np.asarray(spec.eta),
        G_R_S13_raw_PM=raw_real["S13"],
        G_R_S135_raw_PM=raw_real["S135"],
        G_R_S13=reconstructed_real["S13"],
        G_R_S135=reconstructed_real["S135"],
        A_S13=spectra["S13"],
        A_S135=spectra["S135"],
        omega_n=omega_n,
        z=matsubara_shifts,
        G_iw_S13_raw_PM=raw_iw["S13"],
        G_iw_S135_raw_PM=raw_iw["S135"],
        G_iw_S13=reconstructed_iw["S13"],
        G_iw_S135=reconstructed_iw["S135"],
        tau=tau,
        tau_over_beta=tau_over_beta,
        G_tau_S13=green_tau["S13"],
        G_tau_S135=green_tau["S135"],
    )

    record = {
        "state": "COMPLETE",
        "beta": 8.0,
        "modes_per_sector": 5,
        "eta": spec.eta,
        "frequency_window": [float(spec.omega[0]), float(spec.omega[-1])],
        "frequency_points": int(spec.omega.size),
        "matsubara_frequencies": int(omega_n.size),
        "S13_to_S135_real_axis_relative_L2": _relative_l2(
            reconstructed_real["S13"], reconstructed_real["S135"]
        ),
        "S13_to_S135_matsubara_relative_L2": _relative_l2(
            reconstructed_iw["S13"], reconstructed_iw["S135"]
        ),
        "real_axis_reconstruction": real_reconstruction,
        "matsubara_reconstruction": matsubara_reconstruction,
    }
    _write_json(OUTPUT / "record.json", record)
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("static", "eom", "assemble"))
    parser.add_argument("--space", choices=("S13", "S135"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    arguments = parser.parse_args()
    if arguments.command == "static":
        run_static(device=arguments.device)
    elif arguments.command == "eom":
        if arguments.space is None:
            parser.error("eom requires --space")
        run_eom(arguments.space, device=arguments.device)
    else:
        assemble()


if __name__ == "__main__":
    main()

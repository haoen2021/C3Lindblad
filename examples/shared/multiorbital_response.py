#!/usr/bin/env python3
"""JAX Q1+Q3 and exact Q1+Q3+Q5 spectra for selected multiorbital cases."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
from ..parameters import parameters
import resource
import shutil
import subprocess
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
PREFIXES = tuple(parameters("response")["multiorbital"]["prefixes"])
OBSERVABLE_TOLERANCE = parameters("response")["multiorbital"]["observable_tolerance"]
STATIC_TOLERANCE = parameters("response")["multiorbital"]["static_tolerance"]
REPOSITORY_ROOT = Path.cwd()
DEFAULT_RESULTS = REPOSITORY_ROOT / "results/multiorbital_full_q5_gpu_v1"
DEFAULT_SCRATCH = REPOSITORY_ROOT / "scratch" / "multiorbital_response"


@dataclass(frozen=True, slots=True)
class CaseSpec:
    name: str
    family: str
    beta: float
    modes_per_correlation: int
    omega: RealArray
    eta: float
    checkpoint: Path
    retained_reference: Path | None
    run_q13: bool


def case_specs() -> dict[str, CaseSpec]:
    """Paper endpoints; beta=8 uses kanamori.realtimebath_n5_campaign."""
    return {
        "kanamori_beta64": CaseSpec(
            name="kanamori_full_mu1_beta64",
            family="kanamori",
            beta=parameters("kanamori")["beta"][1],
            modes_per_correlation=parameters("kanamori")["modes_per_branch"],
            omega=np.linspace(*parameters("kanamori")["frequency"]),
            eta=parameters("kanamori")["eta"],
            checkpoint=REPOSITORY_ROOT / "inputs/kanamori_full_mu1_beta64/static.npz",
            retained_reference=None,
            run_q13=True,
        ),
        "inchmoss_corrected": CaseSpec(
            name="inchmoss_corrected",
            family="inchmoss",
            beta=parameters("inchmoss")["beta"],
            modes_per_correlation=parameters("inchmoss")["modes_per_branch"],
            omega=np.linspace(*parameters("inchmoss")["frequency"]),
            eta=parameters("inchmoss")["eta"],
            checkpoint=REPOSITORY_ROOT / "inputs/inchmoss/static_cpu_direct.npz",
            retained_reference=REPOSITORY_ROOT
            / "inputs/inchmoss/comparison_compact.npz",
            run_q13=True,
        ),
    }


def spectral_matrix(retarded: ComplexArray) -> ComplexArray:
    """Return ``i (G-G^dagger)/(2 pi)`` at every frequency."""

    values = np.asarray(retarded, dtype=np.complex128)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("retarded response must be a square matrix grid")
    return np.asarray(
        1j * (values - values.conj().swapaxes(1, 2)) / (2.0 * np.pi),
        dtype=np.complex128,
    )


def assemble_source_columns(
    sources: tuple[dict[str, Any], ...],
    grid: RealArray,
    sink_order: tuple[int, ...],
) -> ComplexArray:
    """Assemble directly computed sink/source columns after convergence."""

    expected_grid = np.asarray(grid, dtype=np.float64)
    expected_sinks = np.asarray(sink_order, dtype=np.int64)
    if len(sources) != len(sink_order):
        raise ValueError("one directly computed source is required per sink site")
    matrix = np.empty(
        (expected_grid.size, len(sink_order), len(sink_order)),
        dtype=np.complex128,
    )
    seen: set[int] = set()
    for source in sources:
        site = int(np.asarray(source["source_index"]).item())
        if site not in sink_order or site in seen:
            raise RuntimeError("source ordering is incomplete or duplicated")
        seen.add(site)
        if not bool(np.asarray(source["converged_observable"]).item()):
            raise RuntimeError("a source did not pass observable convergence")
        if not np.array_equal(np.asarray(source["sink_order"]), expected_sinks):
            raise RuntimeError("source sink ordering changed")
        if not np.array_equal(np.asarray(source["grid"]), expected_grid):
            raise RuntimeError("source frequency grid changed")
        final_m = int(np.asarray(source["final_m"]).item())
        column = np.asarray(source[f"G_full_m{final_m}"], dtype=np.complex128)
        if column.shape != (expected_grid.size, len(sink_order)):
            raise RuntimeError("source column has the wrong shape")
        if not np.all(np.isfinite(column)):
            raise RuntimeError("source column is nonfinite")
        matrix[:, :, sink_order.index(site)] = column
    return matrix


def _atomic_savez(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _git_head() -> str:
    """Identify the installed solver checkout when available."""
    import c3l

    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(c3l.__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def configure_jax(scratch: Path, *, device: str = "gpu") -> Path:
    cache = scratch / f"jax_compilation_cache_{device}"
    configured = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    defaults = tuple(str(scratch / f"jax_compilation_cache_{platform}") for platform in ("cpu", "gpu"))
    if configured is not None and configured not in defaults:
        cache = Path(configured)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    return cache


def _resource_gate(operator_dimension: int, *, device: Any = None) -> dict[str, int | None]:
    from c3l.green.resident_arnoldi import check_resident_memory, resident_workspace_bytes

    if device is None:
        import jax

        device = jax.devices()[0]
    cap = min(PREFIXES[-1], operator_dimension)
    required = ((cap + 32) // 32) * 32 * operator_dimension * 16
    available = check_resident_memory(device, required)
    workspace = resident_workspace_bytes(device, available)
    return {"required_device_bytes": required + workspace, "available_device_bytes": available}


def _verify_source_compact(
    path: Path,
    grid: RealArray,
    source_index: int,
    sink_order: tuple[int, ...],
    operator_dimension: int,
    *,
    eta: float,
    provenance: str,
) -> None:
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["schema"].item()) != "cclens.multiorbital.q135.source.v1":
            raise RuntimeError("source compact schema changed")
        if int(saved["source_index"].item()) != source_index:
            raise RuntimeError("source compact index changed")
        if not np.array_equal(saved["sink_order"], np.asarray(sink_order)):
            raise RuntimeError("source compact sink order changed")
        if not np.array_equal(saved["grid"], np.asarray(grid)):
            raise RuntimeError("source compact grid changed")
        if int(saved["operator_dimension"].item()) != operator_dimension:
            raise RuntimeError("source compact operator dimension changed")
        if float(saved["eta"]) != eta:
            raise RuntimeError("source compact eta changed")
        if str(saved["provenance"].item()) != provenance:
            raise RuntimeError("source compact provenance changed")
        prefixes = np.asarray(saved["prefixes"], dtype=np.int64)
        changes = np.asarray(saved["observable_change_relative_l2"], dtype=np.float64)
        if (
            prefixes.ndim != 1
            or prefixes.size < 1
            or (prefixes.size < 2 and not bool(saved.get("krylov_closed", False)))
            or changes.shape != prefixes.shape
            or np.any(np.diff(prefixes) <= 0)
            or int(saved["final_m"].item()) != int(prefixes[-1])
        ):
            raise RuntimeError("source compact prefixes are invalid")
        for prefix in prefixes:
            column = np.asarray(saved[f"G_full_m{int(prefix)}"])
            if column.shape != (len(grid), len(sink_order)) or not np.all(
                np.isfinite(column)
            ):
                raise RuntimeError("source compact contains an invalid column")


def _observable_change(old: ComplexArray, new: ComplexArray) -> float:
    return float(
        np.linalg.norm((new - old).ravel())
        / max(np.linalg.norm(new.ravel()), np.finfo(np.float64).tiny)
    )


def _run_combined_source(
    operator: Any,
    physical_source: Any,
    sink_sources: tuple[Any, ...],
    spec: CaseSpec,
    root: Path,
    *,
    provenance: str,
) -> dict[str, Any]:
    """Run one device-resident chain for the combined physical source."""

    from c3l.green.resident_arnoldi import ResidentArnoldi
    from .response import resident_curve

    root.mkdir(parents=True, exist_ok=True)
    compact = root / "compact.npz"
    sink_order = tuple(int(item.site) for item in sink_sources)
    cap = min(PREFIXES[-1], operator.dimension)
    if compact.exists():
        _verify_source_compact(
            compact, spec.omega, int(physical_source.site), sink_order,
            int(operator.dimension), eta=spec.eta, provenance=provenance,
        )
        with np.load(compact, allow_pickle=False) as saved:
            existing = {name: np.asarray(saved[name]) for name in saved.files}
        if bool(existing["converged_observable"]) or int(existing["final_m"]) >= cap:
            return existing

    retained_source = np.asarray(physical_source.addition + physical_source.removal, complex)
    source = np.zeros(operator.dimension, complex)
    source[:retained_source.size] = retained_source
    sinks = np.zeros((len(sink_sources), operator.dimension), complex)
    sinks[:, :retained_source.size] = np.vstack([item.sink for item in sink_sources])
    norm = float(np.linalg.norm(source))
    chain = ResidentArnoldi(
        operator.device_matvec, source, sinks, cap, device=getattr(operator, "device", None)
    )
    arrays = {
        "schema": np.asarray("cclens.multiorbital.q135.source.v1"),
        "source_index": np.asarray(physical_source.site, dtype=np.int64),
        "sink_order": np.asarray(sink_order, dtype=np.int64),
        "grid": np.asarray(spec.omega),
        "eta": np.asarray(spec.eta),
        "operator_dimension": np.asarray(operator.dimension, dtype=np.int64),
        "provenance": np.asarray(provenance),
        "source_convention": np.asarray("addition_plus_removal_direct"),
    }
    completed = []
    changes = []
    prior = None
    for target in sorted({min(prefix, cap) for prefix in PREFIXES}):
        chain.advance(target)
        actual = chain.current
        column = resident_curve(chain, norm, spec.omega, "frequency", eta=spec.eta)
        completed.append(actual)
        change = np.nan if prior is None else _observable_change(prior, column)
        changes.append(change)
        prior = column
        closed = chain.breakdown or actual == operator.dimension
        arrays[f"G_full_m{actual}"] = column
        arrays["prefixes"] = np.asarray(completed, dtype=np.int64)
        arrays["observable_change_relative_l2"] = np.asarray(changes)
        arrays["final_m"] = np.asarray(actual, dtype=np.int64)
        arrays["krylov_closed"] = np.asarray(closed)
        _atomic_savez(root / "progress.npz", arrays)
        print(json.dumps(dict(event="prefix_complete", case=spec.name,
                              source=int(physical_source.site), m=actual,
                              observable_change=change)), flush=True)
        if closed or change <= OBSERVABLE_TOLERANCE:
            break
    arrays.update(
        converged_observable=np.asarray(closed or changes[-1] <= OBSERVABLE_TOLERANCE),
        hessenberg=chain.hessenberg[:actual + 1, :actual],
        sink_projection=chain.projected[:, :actual],
        source_norm=np.asarray(norm),
        action_wall_seconds=np.asarray(chain.action_seconds),
        orthogonalization_wall_seconds=np.asarray(chain.orthogonalization_seconds),
        orthogonalization_backend=np.asarray(f"{chain.device.platform}_resident_cgs2"),
        peak_rss_kib=np.asarray(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )
    _atomic_savez(compact, arrays)
    _verify_source_compact(
        compact, spec.omega, int(physical_source.site), sink_order,
        int(operator.dimension), eta=spec.eta, provenance=provenance,
    )
    (root / "progress.npz").unlink(missing_ok=True)
    return arrays


def _source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    action = np.asarray(source["action_wall_seconds"], dtype=np.float64)
    orthogonalization = np.asarray(
        source["orthogonalization_wall_seconds"], dtype=np.float64
    )
    changes = np.asarray(source["observable_change_relative_l2"], dtype=np.float64)
    warm = action[1:] if action.size > 1 else action
    return {
        "source_index": int(np.asarray(source["source_index"]).item()),
        "final_m": int(np.asarray(source["final_m"]).item()),
        "observable_change_relative_l2": float(changes[-1]),
        "observable_converged": bool(np.asarray(source["converged_observable"]).item()),
        "cold_action_wall_seconds": float(action[0]),
        "warm_action_wall_seconds_median": float(np.median(warm)),
        "action_wall_seconds_total": float(np.sum(action)),
        "orthogonalization_wall_seconds_total": float(np.sum(orthogonalization)),
        "orthogonalization_backend": str(
            np.asarray(source["orthogonalization_backend"]).item()
        ),
    }


def _load_case(spec: CaseSpec):
    from c3l.cc.component_tensor import (
        ComponentEOMProblem,
        component_eom_context,
        component_eom_sources,
        component_static_residual,
    )
    from c3l.cc.eom import eom_ccsd_problem

    if spec.family == "kanamori":
        from ..kanamori import run as benchmark

        static = benchmark.build_static_problem(
            spec.beta,
            spec.modes_per_correlation,
            "chemical_potential",
        )
    else:
        from ..inchmoss import run as benchmark

        static = benchmark.build_static_problem(corrected_convention=True)
    with np.load(spec.checkpoint, allow_pickle=False) as saved:
        if "converged" in saved and not bool(saved["converged"].item()):
            raise RuntimeError("selected static checkpoint is not converged")
        amplitudes = np.asarray(saved["amplitudes"], dtype=np.complex128)
    if amplitudes.shape != (static.oracle.num_amplitudes,):
        raise RuntimeError("selected static amplitudes have the wrong shape")
    residual = component_static_residual(static, amplitudes)
    packed = float(
        np.linalg.norm(static.oracle.tilde.pack(residual.retained, check=False))
    )
    if packed > STATIC_TOLERANCE:
        raise RuntimeError("selected static checkpoint misses its fresh gate")
    eom = eom_ccsd_problem(static.oracle, response_spin=0, build_next_rank_masks=False)
    response = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(response, amplitudes)
    sites = (0, 1)
    sources = tuple(component_eom_sources(context, site=site) for site in sites)
    return static, amplitudes, context, sources, packed


def _build_full_operator(context: Any, *, device: str = "gpu"):
    from c3l.cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator

    started = perf_counter()
    compiled = CompiledFullS135Action(context, materialization_batch_size=16)
    cpu = FullS135ResponseOperator(compiled, s5_tile_size=16384)
    from c3l.cc.s135_jax import JaxFullS135ResponseOperator

    operator = JaxFullS135ResponseOperator.from_cpu(cpu, platform=device, production=True)
    return operator, compiled, float(perf_counter() - started)


def _kanamori_reconstruction_inputs(spec: CaseSpec):
    from ..kanamori.run import (
        bethe_retarded,
        build_model,
        impurity_one_body,
        orbital_hybridization_matrix,
    )
    from c3l.gaussian_mean_field import quadratic_frequency_response
    from c3l.quadratic_ness import finite_pm_covariance

    model = build_model(
        spec.beta,
        spec.modes_per_correlation,
        "chemical_potential",
    )
    covariance, lyapunov = finite_pm_covariance(
        model.E, model.gamma_loss, model.gamma_gain
    )
    shifts = spec.omega.astype(np.complex128) + 1j * spec.eta
    g0_pm = quadratic_frequency_response(
        model.E,
        model.gamma_loss,
        model.gamma_gain,
        covariance,
        shifts,
        orbitals=(0, 1),
    ).retarded
    identity = np.eye(2, dtype=np.complex128)
    one_body = impurity_one_body("chemical_potential")
    hybridization = orbital_hybridization_matrix()
    g0_continuum = np.asarray(
        [
            np.linalg.solve(
                shift * identity - one_body - hybridization * bethe_retarded(shift),
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
    return g0_pm, g0_continuum, delta_pm, delta_continuum, float(lyapunov)


def _reconstruct(
    raw: ComplexArray,
    delta_pm: ComplexArray,
    delta_continuum: ComplexArray,
) -> tuple[ComplexArray, float]:
    from c3l.spectra.reconstruction import matrix_ordered_dyson_reconstruction

    reconstructed = matrix_ordered_dyson_reconstruction(raw, delta_pm, delta_continuum)
    return np.asarray(reconstructed.retarded), float(
        np.max(reconstructed.ordering_difference)
    )


def _relative_l2(left: ComplexArray, right: ComplexArray) -> float:
    return float(
        np.linalg.norm((left - right).ravel())
        / max(np.linalg.norm(right.ravel()), np.finfo(np.float64).tiny)
    )


def _response_checks(
    omega: RealArray, retarded: ComplexArray, spectral: ComplexArray
) -> dict[str, Any]:
    diagonal = np.diagonal(spectral, axis1=1, axis2=2).real
    return {
        "diagonal_sum_rules": np.trapezoid(diagonal, x=omega, axis=0).tolist(),
        "minimum_diagonal_spectrum": float(np.min(diagonal)),
        "reciprocity_relative_l2": float(
            np.linalg.norm(retarded - retarded.swapaxes(1, 2))
            / max(np.linalg.norm(retarded), np.finfo(np.float64).tiny)
        ),
        "reflection_relative_l2": float(
            np.linalg.norm(spectral - spectral[::-1])
            / max(np.linalg.norm(spectral), np.finfo(np.float64).tiny)
        ),
    }


def _retained_reference(spec: CaseSpec) -> ComplexArray | None:
    if spec.retained_reference is None:
        return None
    with np.load(spec.retained_reference, allow_pickle=False) as saved:
        key = "retarded" if spec.family == "kanamori" else "G_R_eta0p05"
        return np.asarray(saved[key], dtype=np.complex128)


def _verify_case_artifact(path: Path, spec: CaseSpec) -> None:
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["schema"].item()) != "cclens.multiorbital.q135.case.v1":
            raise RuntimeError("case artifact schema changed")
        if not np.array_equal(saved["omega"], spec.omega):
            raise RuntimeError("case artifact frequency grid changed")
        for key in (
            "G_R_Q13_raw_PM",
            "G_R_Q135_raw_PM",
            "G_R_Q13",
            "G_R_Q135",
            "A_Q13",
            "A_Q135",
        ):
            values = np.asarray(saved[key])
            if values.shape != (len(spec.omega), 2, 2) or not np.all(
                np.isfinite(values)
            ):
                raise RuntimeError(f"case artifact field {key} is invalid")


def run_case(
    spec: CaseSpec,
    results: Path,
    scratch: Path,
    *,
    device: str = "gpu",
) -> dict[str, Any]:
    """Run or reopen one selected endpoint."""

    output = results / spec.name
    artifact = output / "compact.npz"
    record_path = output / "record.json"
    if artifact.exists() and record_path.exists():
        with np.load(artifact, allow_pickle=False) as saved:
            completed_schema = str(saved["schema"].item())
        if completed_schema == "cclens.multiorbital.q135.case.v1":
            _verify_case_artifact(artifact, spec)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("production_backend", "gpu") == device:
                return record

    started = perf_counter()
    static, amplitudes, context, sources, static_residual = _load_case(spec)
    sink_order = tuple(int(source.site) for source in sources)
    if spec.run_q13:
        from c3l.cc.component_jax import JaxRetainedEOMOperator

        retained_started = perf_counter()
        retained = JaxRetainedEOMOperator.from_cpu(
            context, platform=device, production=True
        )
        device = getattr(getattr(retained, "device", None), "platform", device)
        retained_setup = float(perf_counter() - retained_started)
        retained_resources = _resource_gate(retained.dimension, device=getattr(retained, "device", None))
        q13_sources = tuple(
            _run_combined_source(
                retained,
                source,
                sources,
                spec,
                scratch / spec.name / "q13" / f"source_{source.site}_{device}",
                provenance=f"multiorbital-q135-v1:{spec.name}:q13:{device}",
            )
            for source in sources
        )
        q13_raw = assemble_source_columns(q13_sources, spec.omega, sink_order)
        q13_memory = retained.device_memory_record()
        del retained
        gc.collect()
    else:
        accepted_q13 = _retained_reference(spec)
        if accepted_q13 is None:
            raise RuntimeError("a reused Q13 result is missing")
        q13_raw = accepted_q13
        q13_sources = ()
        retained_setup = 0.0
        retained_resources = {}
        q13_memory = {}

    full, compiled, full_setup = _build_full_operator(context, device=device)
    device = getattr(getattr(full, "device", None), "platform", device)
    full_resources = _resource_gate(full.dimension, device=getattr(full, "device", None))
    q135_sources = tuple(
        _run_combined_source(
            full,
            source,
            sources,
            spec,
            scratch / spec.name / "q135" / f"source_{source.site}_{device}",
            provenance=(f"multiorbital-q135-v1:{spec.name}:q135:{device}"),
        )
        for source in sources
    )
    q135_raw = assemble_source_columns(q135_sources, spec.omega, sink_order)
    full_memory = full.device_memory_record()
    dimensions = {
        "P": int(full.p_dimension),
        "Q5": int(full.s5_dimension),
        "total": int(full.dimension),
    }
    runtime = asdict(full.runtime)
    del full
    gc.collect()

    reconstruction: dict[str, Any] = {"applied": False}
    q13 = q13_raw
    q135 = q135_raw
    auxiliary: dict[str, Any] = {}
    if spec.family == "kanamori":
        g0_pm, g0_cont, delta_pm, delta_cont, lyapunov = (
            _kanamori_reconstruction_inputs(spec)
        )
        q13, q13_order = _reconstruct(q13_raw, delta_pm, delta_cont)
        q135, q135_order = _reconstruct(q135_raw, delta_pm, delta_cont)
        auxiliary = {
            "G0_PM": g0_pm,
            "G0_continuum": g0_cont,
            "Delta_PM": delta_pm,
            "Delta_continuum": delta_cont,
        }
        reconstruction = {
            "applied": True,
            "finite_PM_G0_lyapunov_residual": lyapunov,
            "Q13_ordering_difference_maximum": q13_order,
            "Q135_ordering_difference_maximum": q135_order,
        }

    a13 = spectral_matrix(q13)
    a135 = spectral_matrix(q135)
    reference = _retained_reference(spec)
    reference_difference = (
        None if reference is None else _relative_l2(q13_raw, reference)
    )
    q13_metadata = [_source_metadata(source) for source in q13_sources]
    q135_metadata = [_source_metadata(source) for source in q135_sources]
    record = {
        "schema": "cclens.multiorbital.q135.case.v1",
        "case": spec.name,
        "family": spec.family,
        "git_head": _git_head(),
        "beta": spec.beta,
        "modes_per_correlation": spec.modes_per_correlation,
        "eta": spec.eta,
        "frequency_points": int(spec.omega.size),
        "frequency_window": [float(spec.omega[0]), float(spec.omega[-1])],
        "static_checkpoint": str(spec.checkpoint.relative_to(REPOSITORY_ROOT)),
        "fresh_static_packed_residual": static_residual,
        "dimensions": dimensions,
        "compiled_group_count": int(compiled.group_count),
        "compiled_storage_bytes": int(compiled.storage_bytes),
        "retained_setup_wall_seconds": retained_setup,
        "full_q5_setup_wall_seconds": full_setup,
        "runtime": runtime,
        "production_backend": device,
        "Q13_origin": f"{device}_production" if spec.run_q13 else "accepted_existing",
        "Q13_sources": q13_metadata,
        "Q135_sources": q135_metadata,
        "Q13_reference_relative_l2": reference_difference,
        "Q13_to_Q135_relative_l2": _relative_l2(q13, q135),
        "Q13_checks": _response_checks(spec.omega, q13, a13),
        "Q135_checks": _response_checks(spec.omega, q135, a135),
        "continuum_reconstruction": reconstruction,
        "resource_gates": {
            "Q13": retained_resources,
            "Q135": full_resources,
        },
        "peak_device_memory": {"Q13": q13_memory, "Q135": full_memory},
        "total_wall_seconds": float(perf_counter() - started),
    }
    arrays: dict[str, Any] = {
        "schema": np.asarray(record["schema"]),
        "omega": spec.omega,
        "eta": np.asarray(spec.eta),
        "sink_source_sites": np.asarray(sink_order, dtype=np.int64),
        "G_R_Q13_raw_PM": q13_raw,
        "G_R_Q135_raw_PM": q135_raw,
        "G_R_Q13": q13,
        "G_R_Q135": q135,
        "A_Q13": a13,
        "A_Q135": a135,
        "Q13_final_m_per_source": np.asarray(
            [item["final_m"] for item in q13_metadata], dtype=np.int64
        ),
        "Q135_final_m_per_source": np.asarray(
            [item["final_m"] for item in q135_metadata], dtype=np.int64
        ),
        "Q13_observable_change_per_source": np.asarray(
            [item["observable_change_relative_l2"] for item in q13_metadata]
        ),
        "Q135_observable_change_per_source": np.asarray(
            [item["observable_change_relative_l2"] for item in q135_metadata]
        ),
        **auxiliary,
    }
    _atomic_savez(artifact, arrays)
    _verify_case_artifact(artifact, spec)
    source_output = output / "sources"
    source_output.mkdir(parents=True, exist_ok=True)
    for label, records in (("q13", q13_sources), ("q135", q135_sources)):
        for source in records:
            site = int(np.asarray(source["source_index"]).item())
            origin = scratch / spec.name / label / f"source_{site}_{device}" / "compact.npz"
            shutil.copy2(origin, source_output / f"{label}_source_{site}.npz")
    _write_json(record_path, record)
    print(json.dumps({"event": "case_complete", **record}), flush=True)
    del static, amplitudes, context
    gc.collect()
    return record


def run_all(results: Path, scratch: Path, *, device: str = "gpu") -> list[dict[str, Any]]:
    configure_jax(scratch, device=device)
    results.mkdir(parents=True, exist_ok=True)
    specs = case_specs()
    order = ("kanamori_beta64", "inchmoss_corrected")
    records = [run_case(specs[name], results, scratch, device=device) for name in order]
    _write_json(
        results / "manifest.json",
        {
            "schema": "cclens.multiorbital.q135.manifest.v1",
            "git_head": _git_head(),
            "cases": [record["case"] for record in records],
            "records": records,
            "scratch": str(scratch),
        },
    )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "case"))
    parser.add_argument("--case", choices=tuple(case_specs()))
    parser.add_argument("--device", "--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    arguments = parser.parse_args()
    if arguments.command == "case" and arguments.case is None:
        parser.error("case command requires --case")
    return arguments


def main() -> None:
    arguments = parse_args()
    configure_jax(arguments.scratch, device=arguments.device)
    if arguments.command == "all":
        print(json.dumps(run_all(arguments.results, arguments.scratch, device=arguments.device), indent=2))
    else:
        spec = case_specs()[arguments.case]
        print(
            json.dumps(
                run_case(
                    spec,
                    arguments.results,
                    arguments.scratch,
                    device=arguments.device,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

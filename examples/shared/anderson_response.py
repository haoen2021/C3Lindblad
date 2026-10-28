#!/usr/bin/env python3
"""Exact JAX full-Q5 response gallery for the three accepted compact cases."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
from ..parameters import parameters
import resource
import subprocess
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from c3l.cc.component_tensor import component_static_residual
from c3l.green.checkpointed_arnoldi import eom_checkpoint_identity


ComplexArray = NDArray[np.complex128]
PREFIXES = tuple(parameters("response")["anderson"]["prefixes"])
OBSERVABLE_TOLERANCE = parameters("response")["anderson"]["observable_tolerance"]
STATIC_RESIDUAL_TOLERANCE = parameters("response")["anderson"]["static_tolerance"]
REPOSITORY_ROOT = Path.cwd()
DEFAULT_RESULTS = REPOSITORY_ROOT / "results/pseudomode_gallery_full_q5"
DEFAULT_SCRATCH = REPOSITORY_ROOT / "scratch" / "anderson_response"


@dataclass(frozen=True, slots=True)
class CheckpointAlignment:
    """One accepted CCSD vector expressed in this LAPACK mode gauge."""

    amplitudes: ComplexArray
    resolved: bool
    pair_code: int
    pair_count: int
    rank2_tilde_defect: float
    full_tilde_defect: float


def observable_change(old: ComplexArray, new: ComplexArray) -> float:
    """Return the established prefix relative-L2 observable change."""

    left = np.asarray(old, dtype=np.complex128)
    right = np.asarray(new, dtype=np.complex128)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("observable prefixes must have the same nonempty shape")
    return float(
        np.linalg.norm((right - left).ravel())
        / max(np.linalg.norm(right.ravel()), np.finfo(float).tiny)
    )


def _extension_time_grid(maximum_time: float, time_step: float) -> NDArray[np.float64]:
    if maximum_time <= 0.0 or time_step <= 0.0:
        raise ValueError("maximum time and time step must be positive")
    steps = int(round(float(maximum_time) / float(time_step)))
    return np.linspace(0.0, float(maximum_time), steps + 1)


def _ambiguous_mode_pairs(reference: Any) -> tuple[tuple[int, int], ...]:
    values = np.asarray(reference.eigenvalues, dtype=np.complex128)
    tolerance = max(
        1.0e-10,
        512.0 * np.finfo(np.float64).eps * max(float(np.linalg.norm(values)), 1.0),
    )
    pairs: list[tuple[int, int]] = []
    for half_plane in (-1, 1):
        indices = [
            int(index) for index in np.flatnonzero(values.imag * half_plane > 0.0)
        ]
        used: set[int] = set()
        for seed in indices:
            if seed in used:
                continue
            group = [
                index
                for index in indices
                if index not in used
                and abs(abs(values[index].imag) - abs(values[seed].imag)) <= tolerance
            ]
            used.update(group)
            if len(group) == 2:
                if group[1] != group[0] + 1:
                    raise RuntimeError("an ambiguous mode pair is not adjacent")
                pairs.append((group[0], group[1]))
            elif len(group) > 1:
                raise RuntimeError("a checkpoint gauge group is not binary")
    return tuple(pairs)


def _swap_mask_pair(
    mask: int, pair: tuple[int, int], local_modes: int
) -> tuple[int, int]:
    target = int(mask)
    sign = 1
    for offset in (0, local_modes):
        first = pair[0] + offset
        second = pair[1] + offset
        first_set = (target >> first) & 1
        second_set = (target >> second) & 1
        if first_set and second_set:
            sign = -sign
        if first_set != second_set:
            target ^= (1 << first) | (1 << second)
    return target, sign


def _pair_transforms(
    problem: Any,
    pairs: tuple[tuple[int, int], ...],
    *,
    creator_rank: int | None,
) -> tuple[NDArray[np.int64], tuple[tuple[NDArray[np.int64], NDArray[np.int8]], ...]]:
    masks = tuple(int(mask) for mask in problem.oracle.cluster_basis.masks)
    selected = np.asarray(
        [
            index
            for index, mask in enumerate(masks)
            if creator_rank is None or mask.bit_count() == creator_rank
        ],
        dtype=np.int64,
    )
    chosen = tuple(masks[index] for index in selected)
    index_by_mask = {mask: index for index, mask in enumerate(chosen)}
    local_modes = 2 * int(problem.oracle.model.n_sites)
    transforms: list[tuple[NDArray[np.int64], NDArray[np.int8]]] = []
    for pair in pairs:
        mapped = tuple(_swap_mask_pair(mask, pair, local_modes) for mask in chosen)
        transforms.append(
            (
                np.fromiter(
                    (index_by_mask[target] for target, _ in mapped),
                    dtype=np.int64,
                    count=len(mapped),
                ),
                np.fromiter(
                    (sign for _, sign in mapped),
                    dtype=np.int8,
                    count=len(mapped),
                ),
            )
        )
    return selected, tuple(transforms)


def _apply_pair_code(
    problem: Any,
    amplitudes: ComplexArray,
    pairs: tuple[tuple[int, int], ...],
    code: int,
) -> ComplexArray:
    _, transforms = _pair_transforms(problem, pairs, creator_rank=None)
    result = np.asarray(amplitudes, dtype=np.complex128).copy()
    for bit, (target, sign) in enumerate(transforms):
        if not (code >> bit) & 1:
            continue
        updated = np.empty_like(result)
        updated[target] = sign * result
        result = updated
    return result


def portable_checkpoint_amplitudes(
    problem: Any,
    amplitudes: ComplexArray,
    *,
    tilde_tolerance: float = 2.0e-10,
) -> CheckpointAlignment:
    """Recover LAPACK-ambiguous pair ordering from exact checkpoint symmetry.

    This performs only exterior-coordinate permutations of the accepted
    amplitudes.  It does not optimize an amplitude, change an equation, or
    alter the normal-mode vectors.
    """

    values = np.asarray(amplitudes, dtype=np.complex128)
    if values.shape != (problem.oracle.num_amplitudes,):
        raise ValueError("checkpoint amplitudes do not match the static problem")
    pairs = _ambiguous_mode_pairs(problem.oracle.reference.up)
    selected, transforms = _pair_transforms(problem, pairs, creator_rank=2)
    if selected.size == 0:
        raise RuntimeError("CCSD checkpoint has no rank-two coordinates")
    inverse = {int(full): local for local, full in enumerate(selected)}
    tilde = problem.oracle.tilde
    partner = np.fromiter(
        (inverse[int(tilde.partner[index])] for index in selected),
        dtype=np.int64,
        count=selected.size,
    )
    phase = np.asarray(tilde.phase, dtype=np.complex128)[selected]
    current = values[selected].copy()

    def defect(vector: ComplexArray) -> float:
        return float(np.max(abs(vector[partner] - phase * np.conj(vector))))

    best_defect = defect(current)
    best_code = 0
    resolved_code: int | None = 0 if best_defect <= tilde_tolerance else None
    if resolved_code is None:
        for gray_index in range(1, 1 << len(transforms)):
            changed_bit = (gray_index & -gray_index).bit_length() - 1
            target, sign = transforms[changed_bit]
            updated = np.empty_like(current)
            updated[target] = sign * current
            current = updated
            candidate = defect(current)
            gray_code = gray_index ^ (gray_index >> 1)
            if candidate < best_defect:
                best_defect = candidate
                best_code = gray_code
            if candidate <= tilde_tolerance:
                resolved_code = gray_code
                best_defect = candidate
                break

    if resolved_code is None:
        return CheckpointAlignment(
            amplitudes=values.copy(),
            resolved=False,
            pair_code=int(best_code),
            pair_count=len(pairs),
            rank2_tilde_defect=float(best_defect),
            full_tilde_defect=float(tilde.defect(values)),
        )
    aligned = _apply_pair_code(problem, values, pairs, resolved_code)
    full_defect = float(tilde.defect(aligned))
    if full_defect > tilde_tolerance:
        return CheckpointAlignment(
            amplitudes=values.copy(),
            resolved=False,
            pair_code=int(resolved_code),
            pair_count=len(pairs),
            rank2_tilde_defect=float(best_defect),
            full_tilde_defect=full_defect,
        )
    return CheckpointAlignment(
        amplitudes=aligned,
        resolved=True,
        pair_code=int(resolved_code),
        pair_count=len(pairs),
        rank2_tilde_defect=float(best_defect),
        full_tilde_defect=full_defect,
    )


def _fresh_static_residual(problem: Any, amplitudes: ComplexArray) -> float:
    retained = component_static_residual(problem, amplitudes).retained
    return float(np.linalg.norm(problem.oracle.tilde.pack(retained, check=False)))


def _atomic_savez(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _git_head() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            cwd=REPOSITORY_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _event(name: str, **values: Any) -> None:
    print(json.dumps({"event": name, **values}, default=str), flush=True)


def _embed_source_and_sinks(
    source: ComplexArray, sinks: ComplexArray, dimension: int
) -> tuple[ComplexArray, ComplexArray]:
    retained = np.asarray(source, dtype=np.complex128)
    rows = np.asarray(sinks, dtype=np.complex128)
    if retained.ndim != 1 or rows.ndim != 2 or rows.shape[1] != retained.size:
        raise ValueError("source and sinks have inconsistent retained coordinates")
    full_source = np.zeros(dimension, dtype=np.complex128)
    full_sinks = np.zeros((rows.shape[0], dimension), dtype=np.complex128)
    full_source[: retained.size] = retained
    full_sinks[:, : retained.size] = rows
    return full_source, full_sinks


def _verify_source_compact(
    path: Path,
    grid: NDArray[np.float64],
    sink_order: tuple[int, ...],
    source_index: int,
    *,
    kind: Literal["frequency", "time"],
    eta: float | None,
    provenance: str,
    operator_dimension: int,
) -> None:
    expected_grid = np.asarray(grid, dtype=np.float64)
    expected_sinks = np.asarray(sink_order, dtype=np.int64)
    expected_eta = np.nan if eta is None else float(eta)
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["schema"].item()) != "cclens.pseudomode_gallery_source.v1":
            raise RuntimeError("compact source schema changed")
        if int(saved["source_index"].item()) != source_index:
            raise RuntimeError("compact source ordering changed")
        if not np.array_equal(saved["sink_order"], expected_sinks):
            raise RuntimeError("compact sink ordering changed")
        if not np.array_equal(saved["grid"], expected_grid):
            raise RuntimeError("compact source grid changed")
        if str(saved["kind"].item()) != kind:
            raise RuntimeError("compact response kind changed")
        observed_eta = float(saved["eta"].item())
        if not (
            (np.isnan(expected_eta) and np.isnan(observed_eta))
            or observed_eta == expected_eta
        ):
            raise RuntimeError("compact response eta changed")
        if str(saved["provenance"].item()) != provenance:
            raise RuntimeError("compact source provenance changed")
        if int(saved["operator_dimension"].item()) != operator_dimension:
            raise RuntimeError("compact operator dimension changed")
        prefixes = np.asarray(saved["prefixes"], dtype=np.int64)
        changes = np.asarray(saved["observable_change_relative_l2"], dtype=np.float64)
        if (
            prefixes.ndim != 1
            or prefixes.size < 1
            or (prefixes.size < 2 and not bool(saved.get("krylov_closed", False)))
            or np.any(np.diff(prefixes) <= 0)
            or changes.shape != prefixes.shape
            or int(saved["final_m"].item()) != int(prefixes[-1])
        ):
            raise RuntimeError("compact source prefixes are invalid")
        expected_shape = (expected_grid.size, expected_sinks.size)
        for prefix in prefixes:
            addition = np.asarray(saved[f"G_addition_m{int(prefix)}"])
            removal = np.asarray(saved[f"G_removal_m{int(prefix)}"])
            full = np.asarray(saved[f"G_full_m{int(prefix)}"])
            if (
                addition.shape != expected_shape
                or removal.shape != expected_shape
                or full.shape != expected_shape
                or not np.all(np.isfinite(addition))
                or not np.all(np.isfinite(removal))
                or not np.all(np.isfinite(full))
            ):
                raise RuntimeError("compact source response is invalid")
            if not np.allclose(addition + removal, full, rtol=2.0e-13, atol=2.0e-13):
                raise RuntimeError("compact response channels do not reconstruct")


def _run_source(
    operator: Any,
    physical_source: Any,
    sink_sources: tuple[Any, ...],
    grid: NDArray[np.float64],
    kind: Literal["frequency", "time"],
    root: Path,
    *,
    eta: float | None,
    provenance: str,
) -> dict[str, Any]:
    from c3l.green.resident_arnoldi import ResidentArnoldi
    from .response import resident_curve

    root.mkdir(parents=True, exist_ok=True)
    compact_path = root / "compact.npz"
    sink_order = tuple(int(item.site) for item in sink_sources)
    cap = min(PREFIXES[-1], operator.dimension)
    if compact_path.exists():
        _verify_source_compact(
            compact_path, grid, sink_order, physical_source.site, kind=kind,
            eta=eta, provenance=provenance, operator_dimension=operator.dimension,
        )
        with np.load(compact_path, allow_pickle=False) as saved:
            existing = {name: np.asarray(saved[name]) for name in saved.files}
        if bool(existing["converged_observable"]) or int(existing["final_m"]) >= cap:
            return existing

    retained_sinks = np.vstack([item.sink for item in sink_sources])
    device = getattr(operator, "device", None)
    vectors = {}
    chains = {}
    norms = {}
    for branch in ("addition", "removal"):
        vectors[branch], sinks = _embed_source_and_sinks(
            getattr(physical_source, branch), retained_sinks, operator.dimension
        )
        norms[branch] = float(np.linalg.norm(vectors[branch]))
        chains[branch] = (
            ResidentArnoldi(
                operator.device_matvec, vectors[branch], sinks, cap, device=device
            )
            if norms[branch] else None
        )
    arrays: dict[str, Any] = {
        "schema": np.asarray("cclens.pseudomode_gallery_source.v1"),
        "source_index": np.asarray(physical_source.site, dtype=np.int64),
        "sink_order": np.asarray(sink_order, dtype=np.int64),
        "grid": grid,
        "kind": np.asarray(kind),
        "eta": np.asarray(np.nan if eta is None else eta, dtype=np.float64),
        "provenance": np.asarray(provenance),
        "operator_dimension": np.asarray(operator.dimension, dtype=np.int64),
    }
    completed = []
    changes = []
    prior = None
    for target in sorted({min(prefix, cap) for prefix in PREFIXES}):
        curves = {}
        for branch, chain in chains.items():
            if chain is None:
                curves[branch] = np.zeros((len(grid), len(sink_order)), complex)
            else:
                chain.advance(target)
                curves[branch] = resident_curve(chain, norms[branch], grid, kind, eta=eta)
        actual = max((chain.current for chain in chains.values() if chain is not None), default=0)
        closed = all(
            chain is None or chain.breakdown or chain.current == operator.dimension
            for chain in chains.values()
        )
        total = curves["addition"] + curves["removal"]
        arrays[f"G_addition_m{actual}"] = curves["addition"]
        arrays[f"G_removal_m{actual}"] = curves["removal"]
        arrays[f"G_full_m{actual}"] = total
        completed.append(actual)
        change = np.nan if prior is None else observable_change(prior, total)
        changes.append(change)
        prior = total
        arrays["prefixes"] = np.asarray(completed, dtype=np.int64)
        arrays["observable_change_relative_l2"] = np.asarray(changes, dtype=np.float64)
        arrays["final_m"] = np.asarray(actual, dtype=np.int64)
        arrays["krylov_closed"] = np.asarray(closed)
        _atomic_savez(root / "progress.npz", arrays)
        _event("prefix_complete", source=int(physical_source.site), kind=kind,
               m=actual, observable_change=change)
        if closed or change <= OBSERVABLE_TOLERANCE:
            break

    if device is None:
        import jax

        device = next((chain.device for chain in chains.values() if chain is not None), None)
        if device is None:
            device = jax.devices()[0]
    arrays.update(
        converged_observable=np.asarray(closed or changes[-1] <= OBSERVABLE_TOLERANCE),
        orthogonalization_backend=np.asarray(f"{device.platform}_resident_cgs2"),
        peak_rss_kib=np.asarray(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )
    for branch, chain in chains.items():
        dimension = 0 if chain is None else chain.current
        arrays[f"dimension_{branch}"] = np.asarray(dimension)
        arrays[f"hessenberg_{branch}"] = (
            np.zeros((1, 0), complex) if chain is None
            else chain.hessenberg[:dimension + 1, :dimension]
        )
        arrays[f"sink_projection_{branch}"] = (
            np.zeros((len(sink_order), 0), complex) if chain is None
            else chain.projected[:, :dimension]
        )
        arrays[f"source_norm_{branch}"] = np.asarray(norms[branch])
        arrays[f"action_wall_seconds_{branch}"] = np.asarray(
            [] if chain is None else chain.action_seconds
        )
        arrays[f"orthogonalization_wall_seconds_{branch}"] = np.asarray(
            [] if chain is None else chain.orthogonalization_seconds
        )
    _atomic_savez(compact_path, arrays)
    _verify_source_compact(
        compact_path, grid, sink_order, physical_source.site, kind=kind,
        eta=eta, provenance=provenance, operator_dimension=operator.dimension,
    )
    (root / "progress.npz").unlink(missing_ok=True)
    return arrays


def _build_operator(
    problem: Any, amplitudes: ComplexArray, sites: tuple[int, ...], *, device: str = "gpu"
):
    from c3l.cc.component_tensor import (
        ComponentEOMProblem,
        component_eom_context,
        component_eom_sources,
    )
    from c3l.cc.eom import eom_ccsd_problem
    from c3l.cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator
    from c3l.cc.s135_jax import JaxFullS135ResponseOperator

    started = perf_counter()
    eom = eom_ccsd_problem(problem.oracle, response_spin=0, build_next_rank_masks=False)
    response = ComponentEOMProblem.compile(eom, problem, charge=1)
    context = component_eom_context(response, amplitudes)
    sources = tuple(component_eom_sources(context, site=site) for site in sites)
    compiled = CompiledFullS135Action(context, materialization_batch_size=16)
    cpu_operator = FullS135ResponseOperator(compiled, s5_tile_size=16384)
    operator = JaxFullS135ResponseOperator.from_cpu(
        cpu_operator, platform=device, production=True
    )
    return operator, sources, compiled, float(perf_counter() - started)


def _retained_time_response(
    problem: Any,
    amplitudes: ComplexArray,
    time: NDArray[np.float64],
):
    from c3l.cc.component_response import (
        ComponentSigmaOperator,
        component_retarded_matrix_time,
    )
    from c3l.cc.component_tensor import (
        ComponentEOMProblem,
        component_eom_context,
        component_eom_sources,
    )
    from c3l.cc.eom import eom_ccsd_problem

    eom = eom_ccsd_problem(problem.oracle, response_spin=0, build_next_rank_masks=False)
    response = ComponentEOMProblem.compile(eom, problem, charge=1)
    context = component_eom_context(response, amplitudes)
    sources = (component_eom_sources(context),)
    return component_retarded_matrix_time(
        ComponentSigmaOperator(context),
        sources,
        np.asarray(time, dtype=np.float64),
        np.asarray([0.1j], dtype=np.complex128),
        maximum_krylov_dimension=160,
        residual_tolerance=1.0e-10,
        include_channels=True,
    )


def _smoke_once(operator: Any, source: Any, sinks: tuple[Any, ...]) -> dict[str, Any]:
    rows = np.vstack([item.sink for item in sinks])
    vector, _ = _embed_source_and_sinks(
        source.addition + source.removal, rows, operator.dimension
    )
    started = perf_counter()
    first = operator.matvec(vector)
    first_seconds = perf_counter() - started
    started = perf_counter()
    warm = operator.matvec(vector)
    warm_seconds = perf_counter() - started
    if first.shape != vector.shape or not np.all(np.isfinite(first)):
        raise RuntimeError("the sole full-Q5 action smoke is invalid")
    if warm.shape != vector.shape or not np.all(np.isfinite(warm)):
        raise RuntimeError("the warmed full-Q5 action smoke is invalid")
    return {
        "first_action_wall_seconds": float(first_seconds),
        "warm_action_wall_seconds": float(warm_seconds),
        "device_memory": operator.device_memory_record(),
    }


def _source_final(source: dict[str, Any]) -> ComplexArray:
    final_m = int(np.asarray(source["final_m"]).item())
    return np.asarray(source[f"G_full_m{final_m}"], dtype=np.complex128)


def _source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    addition_actions = np.asarray(source["action_wall_seconds_addition"])
    removal_actions = np.asarray(source["action_wall_seconds_removal"])
    actions = np.concatenate((addition_actions, removal_actions))
    orthogonalization = np.concatenate(
        (
            np.asarray(source["orthogonalization_wall_seconds_addition"]),
            np.asarray(source["orthogonalization_wall_seconds_removal"]),
        )
    )
    warm = np.concatenate((addition_actions[1:], removal_actions[1:]))
    if warm.size == 0:
        warm = actions
    changes = np.asarray(source["observable_change_relative_l2"])
    return {
        "final_m": int(np.asarray(source["final_m"]).item()),
        "observable_change_relative_l2": float(changes[-1]),
        "observable_converged": bool(np.asarray(source["converged_observable"]).item()),
        "orthogonalization_backend": str(
            np.asarray(source["orthogonalization_backend"]).item()
        ),
        "cold_action_wall_seconds_addition": float(addition_actions[0]) if addition_actions.size else 0.0,
        "cold_action_wall_seconds_removal": float(removal_actions[0]) if removal_actions.size else 0.0,
        "action_wall_seconds_total": float(np.sum(actions)),
        "warm_action_wall_seconds_median": float(np.median(warm)) if warm.size else 0.0,
        "orthogonalization_wall_seconds_total": float(np.sum(orthogonalization)),
    }


def _save_case_artifact(
    path: Path,
    arrays: dict[str, Any],
    required_shape: tuple[int, ...],
) -> None:
    _atomic_savez(path, arrays)
    with np.load(path, allow_pickle=False) as saved:
        full = np.asarray(saved["G_full_Q135"])
        if full.shape != required_shape or not np.all(np.isfinite(full)):
            raise RuntimeError("case artifact failed its compact response check")


def _static_alignment_record(alignment: CheckpointAlignment) -> dict[str, Any]:
    return {
        "checkpoint_gauge_resolved": bool(alignment.resolved),
        "checkpoint_pair_code": int(alignment.pair_code),
        "checkpoint_pair_count": int(alignment.pair_count),
        "rank2_tilde_defect": float(alignment.rank2_tilde_defect),
        "full_tilde_defect": float(alignment.full_tilde_defect),
    }


def run_n2(
    U: float,
    results: Path,
    scratch: Path,
    *,
    smoke: bool,
    device: str = "gpu",
) -> dict[str, Any]:
    from ..siam_ed import run as benchmark

    token = f"U{int(U)}"
    output = results / f"single_impurity_n2_{token}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as saved:
            record = json.loads(str(saved["record_json"].item()))
        if record.get("runtime", {}).get("platform", "gpu") == device:
            return record
    started = perf_counter()
    problem = benchmark.build_static_problem(U)
    alignment = portable_checkpoint_amplitudes(problem, benchmark.load_static(U))
    residual = _fresh_static_residual(problem, alignment.amplitudes)
    if not alignment.resolved or residual > STATIC_RESIDUAL_TOLERANCE:
        raise RuntimeError(f"N2 U={U:g} static checkpoint misses its fresh gate")
    operator, sources, compiled, operator_setup_seconds = _build_operator(
        problem, alignment.amplitudes, (problem.oracle.model.impurity_site,), device=device
    )
    device = getattr(getattr(operator, "device", None), "platform", device)
    checkpoint_identity = eom_checkpoint_identity(compiled.context)
    saved = benchmark.load_response(U)
    omega = np.asarray(saved["omega"], dtype=np.float64)
    eta = float(saved["eta"])
    smoke_record = _smoke_once(operator, sources[0], sources) if smoke else {}
    source = _run_source(
        operator,
        sources[0],
        sources,
        omega,
        "frequency",
        scratch / "single_impurity_n2" / token / f"source_0_{device}",
        eta=eta,
        provenance=f"gallery-full-q5:n2:{U:g}:{checkpoint_identity}:{device}",
    )
    full = _source_final(source)[:, 0]
    metadata = _source_metadata(source)
    calculation_provenance = {"git_head": _git_head()}
    record = {
        "schema": "cclens.pseudomode_gallery_full_q5.case.v1",
        "family": "single_impurity_n2",
        "U": float(U),
        "git_head": calculation_provenance["git_head"],
        "calculation_provenance": calculation_provenance,
        "static_packed_residual": residual,
        **_static_alignment_record(alignment),
        "dimensions": {
            "P": int(operator.p_dimension),
            "Q5": int(operator.s5_dimension),
            "total": int(operator.dimension),
        },
        "operator_setup_wall_seconds": operator_setup_seconds,
        "compiled_storage_bytes": int(compiled.storage_bytes),
        "runtime": asdict(operator.runtime),
        "production_backend": device,
        "full_q5_plan_fingerprint": checkpoint_identity,
        "source": metadata,
        "smoke": smoke_record,
        "total_wall_seconds": float(perf_counter() - started),
        "peak_device_memory": operator.device_memory_record(),
    }
    arrays: dict[str, Any] = {
        "schema": np.asarray(record["schema"]),
        "omega": omega,
        "eta": np.asarray(eta),
        "G_ED": np.asarray(saved["G_ED"]),
        "G_Q13": np.asarray(saved["G_EOM"]),
        "G_full_Q135": full,
        "G_addition_full_Q135": np.asarray(
            source[f"G_addition_m{metadata['final_m']}"]
        )[:, 0],
        "G_removal_full_Q135": np.asarray(source[f"G_removal_m{metadata['final_m']}"])[
            :, 0
        ],
        "prefixes": np.asarray(source["prefixes"]),
        "observable_change_relative_l2": np.asarray(
            source["observable_change_relative_l2"]
        ),
        "record_json": np.asarray(json.dumps(record)),
    }
    for prefix in np.asarray(source["prefixes"], dtype=np.int64):
        arrays[f"G_full_m{prefix}"] = np.asarray(source[f"G_full_m{prefix}"])[:, 0]
    _save_case_artifact(output, arrays, omega.shape)
    _event("case_complete", family="n2", U=U, **metadata)
    del operator
    gc.collect()
    return record


def run_n6(
    U: float,
    results: Path,
    scratch: Path,
    *,
    maximum_time: float | None = None,
    device: str = "gpu",
) -> dict[str, Any]:
    from ..siam_tdvp import run as benchmark

    token = benchmark.token(U)
    extension = "" if maximum_time is None else f"_t{benchmark.token(maximum_time)}"
    output = results / f"single_impurity_n6_U{token}{extension}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as saved:
            record = json.loads(str(saved["record_json"].item()))
        if record.get("runtime", {}).get("platform", "gpu") == device:
            return record
    started = perf_counter()
    problem = benchmark.build_static_problem(U)
    raw = benchmark.load_static(U)["amplitudes"]
    alignment = portable_checkpoint_amplitudes(problem, raw)
    residual = _fresh_static_residual(problem, alignment.amplitudes)
    if not alignment.resolved or residual > STATIC_RESIDUAL_TOLERANCE:
        raise RuntimeError(f"N6 U={U:g} static checkpoint misses its fresh gate")
    accepted = benchmark.load_retained_response(U)
    retained_result = None
    retained_metadata: dict[str, Any] = {}
    if maximum_time is None:
        time = np.asarray(accepted["time"], dtype=np.float64)
        retained_curve = np.asarray(accepted["G_retarded"])
        source_root = scratch / "single_impurity_n6" / f"U{token}" / "source_0"
    else:
        accepted_time = np.asarray(accepted["time"], dtype=np.float64)
        time = _extension_time_grid(
            maximum_time, float(accepted_time[1] - accepted_time[0])
        )
        if not np.array_equal(time[: accepted_time.size], accepted_time):
            raise RuntimeError("extended grid does not preserve the accepted grid")
        retained_started = perf_counter()
        retained_result = _retained_time_response(problem, alignment.amplitudes, time)
        retained_curve = np.asarray(retained_result.retarded)[:, 0, 0]
        retained_metadata = {
            "time_maximum": float(time[-1]),
            "time_step": float(time[1] - time[0]),
            "retained_wall_seconds": float(perf_counter() - retained_started),
            "retained_peak_rss_kib": int(retained_result.peak_rss_kib),
            "retained_maximum_relative_time_residual": float(
                np.max(retained_result.combined_relative_time_residuals)
            ),
            "retained_krylov_dimensions": [
                int(item.dimension) for item in retained_result.combined_projections
            ],
        }
        source_root = results / f"single_impurity_n6{extension}_sources" / f"U{token}"
    operator, sources, compiled, operator_setup_seconds = _build_operator(
        problem, alignment.amplitudes, (problem.oracle.model.impurity_site,), device=device
    )
    device = getattr(getattr(operator, "device", None), "platform", device)
    source_root = source_root / device
    checkpoint_identity = eom_checkpoint_identity(compiled.context)
    source = _run_source(
        operator,
        sources[0],
        sources,
        time,
        "time",
        source_root,
        eta=None,
        provenance=(f"gallery-full-q5:n6:{U:g}{extension}:{checkpoint_identity}:{device}"),
    )
    full = _source_final(source)[:, 0]
    metadata = _source_metadata(source)
    if maximum_time is not None:
        accepted_points = np.asarray(accepted["time"]).size
        if not np.allclose(
            retained_curve[:accepted_points],
            accepted["G_retarded"],
            rtol=5.0e-12,
            atol=5.0e-12,
        ):
            raise RuntimeError("extended retained curve changed the accepted window")
        with np.load(
            results / f"single_impurity_n6_U{token}.npz",
            allow_pickle=False,
        ) as prior:
            if not np.allclose(
                full[:accepted_points],
                prior["G_full_Q135"],
                rtol=5.0e-12,
                atol=5.0e-12,
            ):
                raise RuntimeError("extended full-Q5 curve changed the accepted window")
    calculation_provenance = {"git_head": _git_head()}
    record = {
        "schema": (
            "cclens.pseudomode_gallery_full_q5.case.v1"
            if maximum_time is None
            else "cclens.pseudomode_gallery_full_q5.time_extension.v1"
        ),
        "family": "single_impurity_n6",
        "U": float(U),
        "git_head": calculation_provenance["git_head"],
        "calculation_provenance": calculation_provenance,
        "static_packed_residual": residual,
        **_static_alignment_record(alignment),
        "dimensions": {
            "P": int(operator.p_dimension),
            "Q5": int(operator.s5_dimension),
            "total": int(operator.dimension),
        },
        "operator_setup_wall_seconds": operator_setup_seconds,
        "compiled_storage_bytes": int(compiled.storage_bytes),
        "runtime": asdict(operator.runtime),
        "production_backend": device,
        "full_q5_plan_fingerprint": checkpoint_identity,
        "source": metadata,
        "time_extension": retained_metadata,
        "source_compact": str(source_root / "compact.npz"),
        "total_wall_seconds": float(perf_counter() - started),
        "peak_device_memory": operator.device_memory_record(),
    }
    arrays: dict[str, Any] = {
        "schema": np.asarray(record["schema"]),
        "time": time,
        "G_Q13": retained_curve,
        "G_full_Q135": full,
        "G_addition_full_Q135": np.asarray(
            source[f"G_addition_m{metadata['final_m']}"]
        )[:, 0],
        "G_removal_full_Q135": np.asarray(source[f"G_removal_m{metadata['final_m']}"])[
            :, 0
        ],
        "prefixes": np.asarray(source["prefixes"]),
        "observable_change_relative_l2": np.asarray(
            source["observable_change_relative_l2"]
        ),
        "record_json": np.asarray(json.dumps(record)),
    }
    if retained_result is not None:
        arrays.update(
            {
                "G_Q13_addition": np.asarray(retained_result.addition)[:, 0, 0],
                "G_Q13_removal": np.asarray(retained_result.removal)[:, 0, 0],
                "Q13_combined_relative_time_residuals": np.asarray(
                    retained_result.combined_relative_time_residuals
                ),
                "Q13_addition_relative_time_residuals": np.asarray(
                    retained_result.addition_relative_time_residuals
                ),
                "Q13_removal_relative_time_residuals": np.asarray(
                    retained_result.removal_relative_time_residuals
                ),
                "Q13_krylov_dimension_combined": np.asarray(
                    [item.dimension for item in retained_result.combined_projections],
                    dtype=np.int64,
                ),
            }
        )
    for prefix in np.asarray(source["prefixes"], dtype=np.int64):
        arrays[f"G_full_m{prefix}"] = np.asarray(source[f"G_full_m{prefix}"])[:, 0]
    _save_case_artifact(output, arrays, time.shape)
    _event("case_complete", family="n6", U=U, extension=extension, **metadata)
    del operator
    gc.collect()
    return record


def run_dimer(results: Path, scratch: Path, *, device: str = "gpu") -> dict[str, Any]:
    from ..dimer import run as benchmark

    output = results / "hubbard_dimer_U2.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as saved:
            record = json.loads(str(saved["record_json"].item()))
        if record.get("runtime", {}).get("platform", "gpu") == device:
            return record
    problem = benchmark.build_static_problem()
    amplitudes = np.asarray(benchmark.load_static()["amplitudes"], dtype=np.complex128)
    if amplitudes.shape != (problem.oracle.num_amplitudes,):
        raise RuntimeError("accepted dimer amplitudes have the wrong shape")
    residual = _fresh_static_residual(problem, amplitudes)
    checkpoint_record = {
        "amplitudes": "dimer/checkpoints/U2.npz",
        "normal_mode_frame": ("dimer/checkpoints/U2_normal_modes.npz"),
        "coordinate_transform": "none",
    }
    if residual > STATIC_RESIDUAL_TOLERANCE:
        calculation_provenance = {"git_head": _git_head()}
        record = {
            "schema": "cclens.pseudomode_gallery_full_q5.gate_failure.v1",
            "family": "hubbard_dimer",
            "state": "STATIC_CHECKPOINT_GATE_FAILED",
            "git_head": calculation_provenance["git_head"],
            "calculation_provenance": calculation_provenance,
            "static_packed_residual": residual,
            "static_checkpoint": checkpoint_record,
            "action": "case_stopped_without_static_solve_or_response",
        }
        _write_json(results / "hubbard_dimer_GATE_FAILED.json", record)
        _event("case_gate_failed", **record)
        return record

    _event(
        "static_gate_passed",
        family="dimer",
        static_packed_residual=residual,
        coordinate_transform="none",
    )
    started = perf_counter()
    operator, sources, compiled, operator_setup_seconds = _build_operator(
        problem, amplitudes, (0, 1), device=device
    )
    device = getattr(getattr(operator, "device", None), "platform", device)
    checkpoint_identity = eom_checkpoint_identity(compiled.context)
    _event(
        "operator_ready",
        family="dimer",
        P=int(operator.p_dimension),
        S5=int(operator.s5_dimension),
        total=int(operator.dimension),
        operator_setup_wall_seconds=operator_setup_seconds,
    )
    saved = benchmark.load_retained_response()
    time = np.linspace(0.0, 3.0, 61)
    retained = benchmark.retained_response(time=time)
    columns = []
    for source in sources:
        columns.append(
            _run_source(
                operator,
                source,
                sources,
                time,
                "time",
                scratch / "hubbard_dimer" / f"source_{source.site}_{device}",
                eta=None,
                provenance=(
                    f"gallery-full-q5:dimer:source{source.site}:{checkpoint_identity}:{device}"
                ),
            )
        )
    full = np.stack([_source_final(column) for column in columns], axis=2)
    source_records = tuple(_source_metadata(column) for column in columns)
    calculation_provenance = {"git_head": _git_head()}
    record = {
        "schema": "cclens.pseudomode_gallery_full_q5.case.v1",
        "family": "hubbard_dimer",
        "U": 2.0,
        "t12": 0.5,
        "git_head": calculation_provenance["git_head"],
        "calculation_provenance": calculation_provenance,
        "static_packed_residual": residual,
        "static_checkpoint": checkpoint_record,
        "dimensions": {
            "P": int(operator.p_dimension),
            "Q5": int(operator.s5_dimension),
            "total": int(operator.dimension),
        },
        "operator_setup_wall_seconds": operator_setup_seconds,
        "compiled_storage_bytes": int(compiled.storage_bytes),
        "runtime": asdict(operator.runtime),
        "production_backend": device,
        "full_q5_plan_fingerprint": checkpoint_identity,
        "sources": source_records,
        "total_wall_seconds": float(perf_counter() - started),
        "peak_device_memory": operator.device_memory_record(),
    }
    arrays = {
        "schema": np.asarray(record["schema"]),
        "time": time,
        "sites": np.asarray((0, 1), dtype=np.int64),
        "G_Q13": np.asarray(retained.retarded),
        "G_full_Q135": full,
        "final_m_by_source": np.asarray(
            [item["final_m"] for item in source_records], dtype=np.int64
        ),
        "observable_change_by_source": np.asarray(
            [item["observable_change_relative_l2"] for item in source_records]
        ),
        "record_json": np.asarray(json.dumps(record)),
    }
    _save_case_artifact(output, arrays, (time.size, 2, 2))
    _event("case_complete", family="dimer", sources=source_records)
    del operator
    gc.collect()
    return record


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


def run_pilot(results: Path, scratch: Path, *, device: str = "gpu") -> dict[str, Any]:
    configure_jax(scratch, device=device)
    results.mkdir(parents=True, exist_ok=True)
    return run_n2(2.0, results, scratch, smoke=True, device=device)


def run_all(results: Path, scratch: Path, *, device: str = "gpu") -> list[dict[str, Any]]:
    configure_jax(scratch, device=device)
    results.mkdir(parents=True, exist_ok=True)
    records = [run_n2(2.0, results, scratch, smoke=True, device=device)]
    records.extend(run_n2(U, results, scratch, smoke=False, device=device) for U in (3.0, 4.0, 5.0))
    records.extend(
        run_n6(U, results, scratch, maximum_time=3.0, device=device) for U in (2.0, 2.5, 3.0)
    )
    records.append(run_dimer(results, scratch, device=device))
    calculation_provenance = {"git_head": _git_head()}
    _write_json(
        results / "run_manifest.json",
        {
            "schema": "cclens.pseudomode_gallery_full_q5.manifest.v1",
            "manifest_assembly_provenance": calculation_provenance,
            "record_provenance_note": (
                "Each record carries its own calculation provenance; cached "
                "legacy records may contain only their generation git_head."
            ),
            "records": records,
            "scratch": str(scratch),
        },
    )
    return records


def run_n6_t3(results: Path, scratch: Path, *, device: str = "gpu") -> list[dict[str, Any]]:
    """Extend all three accepted N=6 responses on the selected device."""

    configure_jax(scratch, device=device)
    results.mkdir(parents=True, exist_ok=True)
    records = [run_n6(U, results, scratch, maximum_time=3.0, device=device) for U in (2.0, 2.5, 3.0)]
    _write_json(
        results / "single_impurity_n6_t3_manifest.json",
        {
            "schema": "cclens.pseudomode_gallery_full_q5.n6_t3_manifest.v1",
            "calculation_provenance": {"git_head": _git_head()},
            "records": records,
            "source_compacts": [
                record.get("source_compact", str(
                    results
                    / "single_impurity_n6_t3_sources"
                    / f"U{token}"
                    / "compact.npz"
                ))
                for token, record in zip(("2", "2p5", "3"), records, strict=True)
            ],
            "scratch": str(scratch),
        },
    )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "all", "dimer-gate", "n6-t3"))
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "pilot":
        print(json.dumps(run_pilot(args.results, args.scratch, device=args.device), indent=2))
    elif args.command == "dimer-gate":
        args.results.mkdir(parents=True, exist_ok=True)
        configure_jax(args.scratch, device=args.device)
        print(json.dumps(run_dimer(args.results, args.scratch, device=args.device), indent=2))
    elif args.command == "n6-t3":
        print(json.dumps(run_n6_t3(args.results, args.scratch, device=args.device), indent=2))
    else:
        print(json.dumps(run_all(args.results, args.scratch, device=args.device), indent=2))


if __name__ == "__main__":
    main()

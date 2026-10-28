"""InchMOSS: cold/warm static CCSD and device-resident joint response."""

import argparse
import gc
import logging
from pathlib import Path
import time

import numpy as np

from ..shared.cc_io import write_json


def prepare(args):
    from . import run
    from c3l.cc.newton_krylov import NewtonKrylovPolicy

    fit = run.load_bath_fit(path=args.bath)
    problem = run.build_static_problem(fit, corrected_convention=True)
    policy = NewtonKrylovPolicy(
        tolerance=1e-10, maximum_newton_iterations=12, gmres_restart=160,
        gmres_maximum_restart_cycles=1, maximum_line_search_steps=12,
        minimum_gmres_relative_tolerance=1e-8, maximum_gmres_relative_tolerance=.5,
        initial_gmres_relative_tolerance=.5, forcing_strategy='eisenstat_walker',
    )
    return ((1., problem),), policy, run.case().beta


def build_response(static, amplitudes, rank, *, device='gpu'):
    from types import SimpleNamespace
    from c3l.cc.eom import eom_ccsd_problem
    from c3l.cc.component_tensor import ComponentEOMProblem, component_eom_context, component_eom_sources
    from c3l.cc.component_jax import JaxRetainedEOMOperator
    from c3l.cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator
    from c3l.cc.s135_jax import JaxFullS135ResponseOperator

    eom = eom_ccsd_problem(static.oracle, response_spin=0, build_next_rank_masks=False)
    context = component_eom_context(ComponentEOMProblem.compile(eom, static, charge=1), amplitudes)
    sources = tuple(component_eom_sources(context, site=site) for site in range(2))
    if rank == 'S13':
        return JaxRetainedEOMOperator.from_cpu(context, platform=device, production=True, projection_tile_size=16384), sources
    native = FullS135ResponseOperator(CompiledFullS135Action(context), s5_tile_size=16384)
    operator = JaxFullS135ResponseOperator.from_cpu(native, platform=device, production=True)
    embedded = []
    for source in sources:
        values = {}
        for name in ('addition', 'removal', 'sink'):
            values[name] = np.zeros(operator.dimension, dtype=complex)
            values[name][:native.p_dimension] = getattr(source, name)
        embedded.append(SimpleNamespace(**values))
    return operator, tuple(embedded)


def reduced_response(hessenberg, norm, shifts):
    import scipy.linalg

    dimension = hessenberg.shape[1]
    triangular, unitary = scipy.linalg.schur(hessenberg[:dimension], output='complex', check_finite=False)
    rhs = np.zeros(dimension, dtype=complex)
    rhs[0] = norm
    rhs = unitary.conj().T @ rhs
    identity = np.eye(dimension)
    coefficients = np.array([unitary @ scipy.linalg.solve_triangular(shift*identity-triangular, rhs, check_finite=False) for shift in shifts])
    residual = abs(hessenberg[dimension, dimension-1]*coefficients[:, -1]) / norm
    return coefficients, residual


def solve_static(problems, policy, *, device='gpu', solver=None):
    if solver is None:
        from c3l.cc.component_jax import solve_component_ccsd_jax
        solver = solve_component_ccsd_jax
    initial, records = None, []
    for scale, problem in problems:
        started = time.perf_counter()
        solution = solver(problem, initial, policy=policy, platform=device, production=True)
        residual = solution.packed_residual_norm
        if not solution.converged or not np.isfinite(residual) or residual > 1e-10:
            raise RuntimeError(f'Static solve failed at interaction scale {scale}: {residual}')
        initial = solution.amplitudes.copy()
        records.append(dict(scale=scale, seconds=time.perf_counter()-started,
                            residual=residual, iterations=solution.iterations))
        print(f'Static scale={scale:g}: residual={residual:.3e}', flush=True)
    return initial, records


def source_response(operator, rhs, sinks, shifts, real_count, cap, output):
    import jax
    import jax.numpy as jnp
    from c3l.green.resident_arnoldi import ResidentArnoldi

    started = time.perf_counter()
    resident = ResidentArnoldi(operator.device_matvec, rhs, sinks, cap,
                              device=getattr(operator, 'device', None))
    device_rhs = jax.device_put(rhs, resident.device)
    prefixes = sorted({min(cap, value) for value in (128, cap)})
    history, previous = [], None
    norm = np.linalg.norm(rhs)
    for target in prefixes:
        resident.advance(target)
        dimension = resident.current
        hessenberg = resident.hessenberg[:dimension+1, :dimension]
        coefficients, estimated = reduced_response(hessenberg, norm, shifts)
        column = coefficients @ resident.projected[:, :dimension].T
        axes = {}
        for name, start, stop in (('spectral', 0, real_count), ('matsubara', real_count, len(shifts))):
            index = start + int(np.argmax(estimated[start:stop]))
            vector = resident.reconstruct(coefficients[index])
            true = float(jnp.linalg.norm(device_rhs - (shifts[index]*vector - operator.device_matvec(vector))) / norm)
            if not np.isfinite(true) or not np.all(np.isfinite(estimated[start:stop])):
                raise FloatingPointError(f'Nonfinite response residual on the {name} axis')
            difference = None if previous is None else column[start:stop] - previous[start:stop]
            relative = None if difference is None else float(np.linalg.norm(difference) / max(np.linalg.norm(column[start:stop]), 1e-30))
            absolute = None if difference is None else float(np.max(abs(difference)))
            stable = resident.breakdown or relative is not None and relative < 2e-4 and absolute < 2e-4
            axes[name] = dict(reduced_residual=float(estimated[start:stop].max()), true_residual=true,
                              relative_change=relative, absolute_change=absolute,
                              converged=bool(max(estimated[start:stop].max(), true) <= 1e-8 and stable))
        history.append(dict(dimension=dimension, axes=axes, seconds=time.perf_counter()-started))
        write_json(output.with_suffix('.checks.json'), history)
        print(f'{output.stem}: dimension={dimension}, converged={all(axis["converged"] for axis in axes.values())}', flush=True)
        if resident.breakdown:
            break
        previous = column
    metrics = dict(wall_seconds=time.perf_counter()-started, history=history, dimension=dimension,
                   action_seconds=sum(resident.action_seconds),
                   orthogonalization_seconds=sum(resident.orthogonalization_seconds),
                   converged=all(axis['converged'] for axis in axes.values()))
    np.savez(output, response=column, hessenberg=hessenberg, sink_projections=resident.projected[:, :dimension+1])
    return column, metrics


def benchmark(args):
    import jax

    started = time.perf_counter()
    platform = getattr(args, 'device', 'gpu')
    device = jax.devices(platform)[0]
    problems, policy, beta = prepare(args)
    with np.load(args.reference) as reference:
        reference_amplitudes = reference['amplitudes'].copy()
    omega = np.linspace(-10, 10, 801)
    etas = np.array([.05])
    wn = (2*np.arange(256)+1)*np.pi/beta
    real_shifts = (omega[None, :] + 1j*etas[:, None]).ravel()
    shifts = np.r_[real_shifts, 1j*wn]
    cap = 256
    record = dict(model='inchmoss', rank=args.rank, beta=beta, cap=cap, policy=policy.to_dict(),
                  device=device.device_kind, basis_storage=f'{device.platform}_only',
                  model_setup_seconds=time.perf_counter()-started, runs=[])
    operator, operator_amplitudes = None, None
    for label in ('cold', 'warm'):
        logging.getLogger('jax').warning('Timing phase: %s', label)
        output = args.output / label
        output.mkdir()
        started = time.perf_counter()
        amplitudes, stages = solve_static(problems, policy, device=platform)
        static_seconds = time.perf_counter()-started
        np.testing.assert_allclose(amplitudes, reference_amplitudes, atol=1e-8, rtol=1e-7)
        reused = operator is not None and np.array_equal(amplitudes, operator_amplitudes)
        operator_seconds = 0.
        if not reused:
            setup_started = time.perf_counter()
            operator, sources = build_response(problems[-1][1], amplitudes, args.rank, device=platform)
            operator_amplitudes = amplitudes.copy()
            operator_seconds = time.perf_counter()-setup_started
        sinks = np.vstack([source.sink for source in sources])
        response_started = time.perf_counter()
        columns, source_records = [], []
        for index, source in enumerate(sources):
            column, metrics = source_response(operator, source.addition+source.removal, sinks, shifts,
                                               len(real_shifts), cap, output / f'source_{index}.npz')
            columns.append(column)
            source_records.append(dict(source=index, **metrics))
            gc.collect()
        response_seconds = time.perf_counter()-response_started
        metrics = dict(label=label, wall_seconds=time.perf_counter()-started,
                       static_seconds=static_seconds, static_stages=stages,
                       response_setup_seconds=operator_seconds, operator_reused=reused,
                       response_seconds=response_seconds, sources=source_records,
                       amplitude_max_error=float(np.max(abs(amplitudes-reference_amplitudes))),
                       converged=all(source['converged'] for source in source_records),
                       device_memory=device.memory_stats())
        green = np.stack(columns, axis=2)
        np.savez(output / 'static.npz', amplitudes=amplitudes)
        np.savez(output / 'response.npz', omega=omega, etas=etas, wn=wn,
                 retarded=green[:len(real_shifts)].reshape(len(etas), len(omega), len(sources), len(sources)),
                 G_iw=green[len(real_shifts):])
        write_json(output / 'result.json', metrics)
        record['runs'].append(metrics)
        record[f'{label}_total_seconds'] = metrics['wall_seconds'] + (record['model_setup_seconds'] if label == 'cold' else 0.)
        write_json(args.output / 'summary.json', record)
        print(f'{label}: static={static_seconds:.3f}s, response={response_seconds:.3f}s, total={record[f"{label}_total_seconds"]:.3f}s', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--rank', choices=['S13', 'S135'], required=True)
    parser.add_argument('--bath', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import jax

    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    jax.devices(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('jax')
    logger.addHandler(logging.FileHandler(args.output / 'compilation.log'))
    logger.propagate = False
    jax.config.update('jax_log_compiles', True)
    logger.warning('Timing phase: model setup')
    benchmark(args)


if __name__ == '__main__':
    main()

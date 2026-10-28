"""Equilibrium exchange junction: cold/warm static CCSD and device-resident joint response."""

import argparse
import gc
import logging
from pathlib import Path
import time

import numpy as np

from ..shared.cc_io import write_json


def prepare(args):
    from .model import assemble_model
    from ..parameters import parameters
    from ..shared.saved_reference import build_problem
    from c3l.gaussian_mean_field import solve_ghf
    from c3l.cc.newton_krylov import NewtonKrylovPolicy

    physical = assemble_model(args.bath, 0.)
    hf = solve_ghf(
        np.kron(physical.E, np.eye(2)), physical.interaction,
        np.kron(physical.gamma_loss, np.eye(2)), np.kron(physical.gamma_gain, np.eye(2)),
        restricted=True, tolerance=1e-10, max_iterations=600, damping=.5,
        terminal_least_squares=True,
    )
    if not hf.converged or not np.isfinite(hf.self_consistency_residual) or hf.self_consistency_residual >= 1e-8:
        raise RuntimeError('Hartree-Fock reference did not converge')
    problems = tuple((scale, build_problem(physical, hf.fock, hf.gamma, scale=scale, checkpoint=args.reference)[1])
                     for scale in (.5, .75, 1.))
    policy = NewtonKrylovPolicy(
        tolerance=1e-10, maximum_newton_iterations=16, gmres_restart=960,
        gmres_maximum_restart_cycles=1, maximum_line_search_steps=12,
        minimum_gmres_relative_tolerance=1e-8, maximum_gmres_relative_tolerance=.5,
        initial_gmres_relative_tolerance=.5, forcing_strategy='eisenstat_walker',
    )
    return problems, policy, parameters('junction')['beta']


def build_response(static, amplitudes, rank, *, device='gpu'):
    from ..shared.response import response_context, response_operator

    context, sources = response_context(static.oracle, static, amplitudes)
    return response_operator(context, sources, device, rank)


def reduced_response(hessenberg, norm, shifts):
    from ..shared.response import reduced_gmres

    return reduced_gmres(hessenberg, norm, shifts)


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
    prefixes = sorted({min(cap, value) for value in (32, 64, 128, 256, 384, 512, 768, 1024, cap)})
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
        if resident.breakdown or all(axis['converged'] for axis in axes.values()):
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
    omega = np.linspace(-8, 8, 641)
    etas = np.array([.05, .1])
    wn = (2*np.arange(256)+1)*np.pi/beta
    real_shifts = (omega[None, :] + 1j*etas[:, None]).ravel()
    shifts = np.r_[real_shifts, 1j*wn]
    cap = 1024 if args.rank == 'S13' else 1536
    record = dict(model='junction', rank=args.rank, beta=beta, cap=cap, policy=policy.to_dict(),
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

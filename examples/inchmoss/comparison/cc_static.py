"""Validate an old checkpoint, or explicitly solve fresh before measuring."""

import argparse
import json, time, itertools
from dataclasses import asdict
from functools import lru_cache
import numpy as np
from .common import ROOT, ROT
from .. import run as b
from c3l.cc.component_tensor import component_static_residual
from c3l.cc.observables import low_rank_cc_coefficients
from c3l.ci.matrix_free import (
    physical_mode_operator,
    FactorizedModeProduct,
    factorized_mode_product_expectation,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fresh", action="store_true")
parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
args = parser.parse_args()

start = time.monotonic()
static = b.build_static_problem(corrected_convention=True)
p = static.oracle
a = b.load_static_amplitudes(corrected_convention=True)
if args.fresh:
    import jax
    from c3l.cc.component_jax import solve_component_ccsd_jax
    from c3l.cc.newton_krylov import NewtonKrylovPolicy

    jax.config.update("jax_enable_x64", True)
    policy = NewtonKrylovPolicy(
        tolerance=1e-10,
        maximum_newton_iterations=12,
        gmres_restart=160,
        gmres_maximum_restart_cycles=1,
        minimum_gmres_relative_tolerance=1e-8,
        maximum_gmres_relative_tolerance=0.5,
        initial_gmres_relative_tolerance=0.5,
        forcing_strategy="eisenstat_walker",
        maximum_line_search_steps=12,
    )

    def save(iteration, amplitudes, residual):
        np.savez(
            ROOT / "results/cc_latest.npz",
            amplitudes=amplitudes,
            reference_V_up=p.reference.up.V,
            reference_V_down=p.reference.down.V,
            residual=residual,
            iteration=iteration,
        )
        print(
            "STEP",
            iteration,
            "residual",
            residual,
            "seconds",
            time.monotonic() - start,
            flush=True,
        )

    print("SETUP", time.monotonic() - start, flush=True)
    solution = solve_component_ccsd_jax(
        static,
        None,
        policy=policy,
        platform=args.device,
        production=True,
        accepted_step_callback=save,
    )
    (ROOT / "results/cc_solve.json").write_text(
        json.dumps(
            dict(
                converged=bool(solution.converged),
                device=args.device,
                policy=policy.to_dict(),
                history=[asdict(x) for x in solution.history],
                seconds=solution.wall_seconds,
            ),
            indent=2,
        )
        + "\n"
    )
    assert solution.converged
    a = solution.amplitudes
r = component_static_residual(static, a).retained
res = float(np.linalg.norm(p.tilde.pack(r, check=False)))
assert res < 1e-8, res
state = low_rank_cc_coefficients(p.reference, p.cluster_basis, a)
ops = {
    (i, d): physical_mode_operator(p.reference, i // 2, i % 2, "c_dagger" if d else "c")
    for i in range(4)
    for d in (False, True)
}


@lru_cache(None)
def mean(indices):
    prod = FactorizedModeProduct(
        "inchmoss_static", 1.0 + 0j, tuple(ops[x] for x in indices)
    )
    return factorized_mode_product_expectation(
        state.basis, state.coefficients, prod, target_mask=0
    )


def measure(rot):
    def exp(indices):
        value = 0j
        for sites in itertools.product(range(2), repeat=len(indices)):
            weight = np.prod(
                [rot[i // 2, site] for (i, d), site in zip(indices, sites)]
            )
            if weight != 0:
                value += weight * mean(
                    tuple((2 * site + i % 2, d) for (i, d), site in zip(indices, sites))
                )
        return value

    def four(i, j, k, l):
        return exp(((i, True), (j, False), (k, True), (l, False)))

    occ = [exp(((i, True), (i, False))) for i in range(4)]
    zz = (four(0, 0, 2, 2) - four(0, 0, 3, 3) - four(1, 1, 2, 2) + four(1, 1, 3, 3)) / 4
    dot = zz + (four(0, 1, 3, 2) + four(1, 0, 2, 3)) / 2
    doubles = [four(0, 0, 1, 1), four(2, 2, 3, 3)]
    assert max(abs(x.imag) for x in occ + doubles + [zz, dot]) < 1e-8
    return dict(
        occupations=[float((occ[0] + occ[1]).real), float((occ[2] + occ[3]).real)],
        double_occupations=[float(x.real) for x in doubles],
        spin_correlation=float(dot.real),
        spin_zz=float(zz.real),
    )


result = dict(
    method="CCSD",
    beta=0.1,
    convention="decoupling_corrected",
    residual=res,
    seconds=time.monotonic() - start,
    observables=dict(original=measure(np.eye(2)), rotated=measure(ROT)),
)
result["seconds"] = time.monotonic() - start
(ROOT / "results/CCSD.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)

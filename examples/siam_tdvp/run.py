"""Load the zero-temperature N=6 AIM and retained-EOM/TDVP responses."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from ..parameters import parameters

import numpy as np

from c3l.cc.component_response import (
    ComponentSigmaOperator,
    component_retarded_matrix_time,
)
from c3l.cc.component_solver import solve_component_ccsd
from c3l.cc.component_tensor import (
    ComponentEOMProblem,
    ComponentStaticProblem,
    component_eom_context,
    component_eom_sources,
    component_static_residual,
)
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import factorized_ccsd_problem
from c3l.cc.newton_krylov import NewtonKrylovPolicy
from ..shared.bath import CoupledPseudomodeFit, coupled_pseudomode_model


ROOT = Path.cwd() / "siam_tdvp"


@dataclass(frozen=True, slots=True)
class Case:
    N: int
    beta: float
    U: tuple[float, ...]
    M: int


def config() -> dict[str, object]:
    return parameters("siam_tdvp")


def case() -> Case:
    cfg = config()
    return Case(
        N=int(cfg["N_pm_per_correlation"]),
        beta=float("inf"),
        U=tuple(float(value) for value in cfg["U"]),
        M=int(cfg["unitary"]["M_total_bath_spin_orbitals"]),
    )


def load_fit() -> CoupledPseudomodeFit:
    """Reconstruct the frozen zero-temperature physical realization."""

    cfg = config()
    with np.load(ROOT / "data/fit.npz", allow_pickle=False) as data:
        return CoupledPseudomodeFit(
            N=case().N,
            H_lesser=np.asarray(data["H_lesser"]),
            Gamma_lesser=np.asarray(data["Gamma_lesser"]),
            g_lesser=np.asarray(data["g_lesser"]),
            H_greater=np.asarray(data["H_greater"]),
            Gamma_greater=np.asarray(data["Gamma_greater"]),
            g_greater=np.asarray(data["g_greater"]),

        )


def build_model(U: float):
    r"""Assemble the finite AIM at interaction ``U`` and $\epsilon_d=-U/2$."""

    interaction = float(U)
    if interaction not in case().U:
        raise ValueError(f"U must be one of {case().U}")
    return coupled_pseudomode_model(load_fit(), U=interaction)


def build_static_problem(U: float) -> ComponentStaticProblem:
    r"""Compile the finite-model creator-rank two and four equations."""

    cc = factorized_ccsd_problem(
        build_model(U),
        m_up=0.5,
        build_next_rank_masks=False,
        reference_mode="legacy_half_filled",
    )
    return ComponentStaticProblem.compile(cc)


def token(U: float) -> str:
    return str(float(U)).replace(".0", "").replace(".", "p")


def load_static(U: float) -> dict[str, np.ndarray]:
    """Load one provenance-compatible static CCSD root."""

    with np.load(ROOT / f"checkpoints/U{token(U)}.npz", allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def load_retained_response(U: float) -> dict[str, np.ndarray]:
    """Load the accepted retained $Q_1+Q_3$ complex time response."""

    with np.load(ROOT / f"data/U{token(U)}_baseline.npz", allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def static_residual(
    U: float, *, problem: ComponentStaticProblem | None = None
) -> float:
    """Reapply the packed tilde-real CCSD residual to the accepted root."""

    p = build_static_problem(U) if problem is None else problem
    amplitudes = load_static(U)["amplitudes"]
    residual = component_static_residual(p, amplitudes).retained
    return float(np.linalg.norm(p.oracle.tilde.pack(residual, check=False)))


def solve_static(
    U: float,
    initial_amplitudes: np.ndarray | None = None,
    *,
    tolerance: float = 1.0e-8,
):
    """Solve the finite-model CCSD equations from a supplied or saved seed."""

    problem = build_static_problem(U)
    initial = (
        load_static(U)["amplitudes"]
        if initial_amplitudes is None
        else initial_amplitudes
    )
    return solve_component_ccsd(
        problem,
        initial,
        policy=NewtonKrylovPolicy(tolerance=float(tolerance)),
        compute_rank6=False,
    )


def retained_response(
    U: float,
    *,
    time: np.ndarray | None = None,
    validation_omega: np.ndarray = np.asarray([0.0]),
    maximum_krylov_dimension: int = 160,
):
    r"""Recompute the retained $S_1\oplus S_3$ response on a time grid."""

    saved = load_retained_response(U)
    t = saved["time"] if time is None else np.asarray(time, dtype=float)
    problem = build_static_problem(U)
    eom = eom_ccsd_problem(problem.oracle, response_spin=0, build_next_rank_masks=False)
    response = ComponentEOMProblem.compile(eom, problem, charge=1)
    context = component_eom_context(response, load_static(U)["amplitudes"])
    sources = (component_eom_sources(context),)
    shifts = np.asarray(validation_omega, dtype=float) + 0.1j
    return component_retarded_matrix_time(
        ComponentSigmaOperator(context),
        sources,
        t,
        shifts,
        maximum_krylov_dimension=maximum_krylov_dimension,
        residual_tolerance=1.0e-10,
        include_channels=True,
    )


def validate() -> dict[str, float | int]:
    """Check fit symmetry, root bindings, grids, and equal-time normalization."""

    fit = load_fit()
    fit_defect = max(
        float(np.max(abs(fit.H_greater + fit.H_lesser.conj()))),
        float(np.max(abs(fit.Gamma_greater - fit.Gamma_lesser.conj()))),
        float(np.max(abs(fit.g_greater - fit.g_lesser.conj()))),
    )
    equal_time = 0.0
    for U in case().U:
        root = load_static(U)
        if int(root["N"]) != case().N or float(root["U"]) != U:
            raise RuntimeError("static checkpoint parameters do not match the case")
        if float(root["epsilon_d"]) != -0.5 * U:
            raise RuntimeError("static checkpoint is not particle-hole symmetric")
        response = load_retained_response(U)
        equal_time = max(equal_time, float(abs(response["G_retarded"][0] + 1j)))
    return {
        "interactions": len(case().U),
        "fit_particle_hole_defect": fit_defect,
        "maximum_equal_time_defect": equal_time,
        "retained_points": int(response["time"].size),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))

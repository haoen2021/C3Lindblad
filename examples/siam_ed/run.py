"""Load the fixed N=2 AIM and its accepted ED/CCSD-EOM comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from ..parameters import parameters

import numpy as np

from c3l.cc.component_response import (
    ComponentSigmaOperator,
    component_retarded_green,
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
from .physical_sparse import (
    compile_physical_sector_operator,
    physical_density_action,
    physical_response_action,
    physical_response_vectors,
    solve_trace_fixed_static,
)
from .sparse_response import sparse_direct_resolvent


ROOT = Path.cwd() / "siam_ed"


@dataclass(frozen=True, slots=True)
class Case:
    N: int
    beta: float
    U: tuple[float, ...]


def config() -> dict[str, object]:
    """Return the readable frozen configuration."""

    return parameters("siam_ed")


def case() -> Case:
    """Return the three parameters that identify this benchmark family."""

    cfg = config()
    return Case(
        N=int(cfg["N_pm_per_correlation"]),
        beta=float(cfg["beta"]),
        U=tuple(float(value) for value in cfg["U"]),
    )


def load_fit() -> CoupledPseudomodeFit:
    """Reconstruct the accepted physical realization from its compact arrays."""

    cfg = config()
    with np.load(ROOT / "data/fit.npz", allow_pickle=False) as data:
        return CoupledPseudomodeFit(
            N=int(cfg["N_pm_per_correlation"]),
            H_lesser=np.asarray(data["H_lesser"]),
            Gamma_lesser=np.asarray(data["Gamma_lesser"]),
            g_lesser=np.asarray(data["g_lesser"]),
            H_greater=np.asarray(data["H_greater"]),
            Gamma_greater=np.asarray(data["Gamma_greater"]),
            g_greater=np.asarray(data["g_greater"]),

        )


def build_model(U: float):
    r"""Return the finite Lindblad AIM with impurity energy $-U/2$."""

    interaction = float(U)
    if not np.isfinite(interaction) or interaction < 0:
        raise ValueError("U must be finite nonnegative")
    return coupled_pseudomode_model(load_fit(), U=interaction)


def build_static_problem(U: float) -> ComponentStaticProblem:
    r"""Compile the creator-rank two and four equations for the requested finite model."""

    cc = factorized_ccsd_problem(
        build_model(U),
        m_up=0.5,
        build_next_rank_masks=False,
        reference_mode="legacy_half_filled",
    )
    return ComponentStaticProblem.compile(cc)


def load_static(U: float) -> np.ndarray:
    """Load the accepted packed CCSD amplitudes for one interaction."""

    token = f"U{int(float(U))}"
    with np.load(ROOT / f"checkpoints/{token}.npz", allow_pickle=False) as data:
        if float(data["U"]) != float(U):
            raise ValueError("checkpoint interaction does not match its filename")
        return np.asarray(data["amplitudes"], dtype=np.complex128)


def load_response(U: float) -> dict[str, np.ndarray]:
    """Load the accepted complex ED and retained-EOM frequency/time arrays."""

    token = f"U{int(float(U))}"
    with np.load(ROOT / f"data/{token}.npz", allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def static_residual(
    U: float, *, problem: ComponentStaticProblem | None = None
) -> float:
    """Reapply the packed tilde-real CCSD residual to the saved root."""

    p = build_static_problem(U) if problem is None else problem
    residual = component_static_residual(p, load_static(U)).retained
    return float(np.linalg.norm(p.oracle.tilde.pack(residual, check=False)))


def solve_static(
    U: float,
    initial_amplitudes: np.ndarray | None = None,
    *,
    tolerance: float = 1.0e-10,
):
    """Solve the same creator-rank two and four equations, optionally from a supplied seed."""

    problem = build_static_problem(U)
    initial = load_static(U) if initial_amplitudes is None else initial_amplitudes
    policy = NewtonKrylovPolicy(tolerance=float(tolerance))
    return solve_component_ccsd(problem, initial, policy=policy, compute_rank6=False)


def retained_response(
    U: float,
    *,
    omega: np.ndarray | None = None,
    eta: float | None = None,
    problem: ComponentStaticProblem | None = None,
):
    r"""Recompute the retained $S_1\oplus S_3$ frequency response."""

    saved = load_response(U)
    frequency = saved["omega"] if omega is None else np.asarray(omega, dtype=float)
    broadening = float(saved["eta"] if eta is None else eta)
    p = build_static_problem(U) if problem is None else problem
    eom = eom_ccsd_problem(p.oracle, response_spin=0, build_next_rank_masks=False)
    response = ComponentEOMProblem.compile(eom, p, charge=1)
    context = component_eom_context(response, load_static(U))
    return component_retarded_green(
        ComponentSigmaOperator(context),
        component_eom_sources(context),
        frequency,
        broadening,
        rtol=1.0e-10,
        restart=40,
        maxiter=200,
    )


def exact_response(
    U: float,
    *,
    omega: np.ndarray,
    eta: float,
    maximum_bytes: int = 2 << 30,
) -> dict[str, np.ndarray | object]:
    r"""Compute the finite-$N=2$ physical-Lindblad ED response.

    The density and odd response generators are assembled independently in
    conserved physical matrix-unit sectors.  This is intentionally separate
    from the CC normal-mode and EOM code paths.
    """

    frequency = np.asarray(omega, dtype=float)
    broadening = float(eta)
    if frequency.ndim != 1 or frequency.size == 0 or not np.all(np.isfinite(frequency)):
        raise ValueError("omega must be a finite nonempty vector")
    if not np.isfinite(broadening) or broadening <= 0.0:
        raise ValueError("eta must be finite and positive")
    model = build_model(U)
    density = compile_physical_sector_operator(
        physical_density_action(model), maximum_bytes=maximum_bytes
    )
    ness = solve_trace_fixed_static(density)
    vectors = physical_response_vectors(
        density.action.sector,
        ness.coefficients,
        model.orbital_index(model.impurity_site, 0),
    )
    response = compile_physical_sector_operator(
        physical_response_action(model, spin=0, charge=1),
        maximum_bytes=maximum_bytes,
    )
    source = np.column_stack((vectors.creation_left, vectors.creation_right))
    solves = sparse_direct_resolvent(
        response,
        source,
        frequency.astype(np.complex128) + 1j * broadening,
        verification_operator=response.action,
    )
    channels = solves.transfer(vectors.annihilation_sink)[:, 0, :]
    return {
        "omega": frequency,
        "eta": np.asarray(broadening),
        "addition": channels[:, 0],
        "removal": channels[:, 1],
        "retarded": channels.sum(axis=1),
        "ness": ness,
        "response": solves,
    }


def validate() -> dict[str, float | int]:
    """Check model dimensions, particle-hole fit symmetry, and saved residuals."""

    fit = load_fit()
    maximum_shifted_residual = 0.0
    maximum_equal_time_defect = 0.0
    for U in case().U:
        model = build_model(U)
        if model.n_sites != 1 + 2 * fit.N:
            raise RuntimeError("single-impurity finite model has the wrong size")
        data = load_response(U)
        maximum_shifted_residual = max(
            maximum_shifted_residual,
            float(np.max(data["true_shifted_residual"])),
        )
        maximum_equal_time_defect = max(
            maximum_equal_time_defect,
            float(abs(data["G_time_EOM"][0] + 1j)),
        )
        if load_static(U).ndim != 1:
            raise RuntimeError("static amplitudes must be a packed vector")
    ph_fit = max(
        float(np.max(abs(fit.H_greater + fit.H_lesser.conj()))),
        float(np.max(abs(fit.Gamma_greater - fit.Gamma_lesser.conj()))),
        float(np.max(abs(fit.g_greater - fit.g_lesser.conj()))),
    )
    return {
        "interactions": len(case().U),
        "particle_hole_fit_defect": ph_fit,
        "maximum_shifted_residual": maximum_shifted_residual,
        "maximum_equal_time_defect": maximum_equal_time_defect,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="validate", choices=("validate",))
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate(), indent=2))


if __name__ == "__main__":
    main()

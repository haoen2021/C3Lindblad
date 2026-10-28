"""Load the accepted $N_{\rm pm}=4$ Hubbard-dimer comparison."""

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
from c3l.ci.reference import HartreeFockReference
from c3l.conventions import TolerancePolicy
from ..shared.bath import CoupledPseudomodeFit, coupled_pseudomode_dimer_model
from c3l.superfermion.normal_modes import NormalModeReference


ROOT = Path.cwd() / "dimer"


@dataclass(frozen=True, slots=True)
class Case:
    N: int
    beta: float
    U: float
    t12: float
    M_total: int


def config() -> dict[str, object]:
    return parameters("dimer")


def case() -> Case:
    cfg = config()
    return Case(
        N=int(cfg["N_pm_per_correlation_per_impurity"]),
        beta=float("inf"),
        U=float(cfg["U"]),
        t12=float(cfg["t12"]),
        M_total=int(cfg["unitary"]["M_total_bath_spin_orbitals"]),
    )


def load_fit() -> CoupledPseudomodeFit:
    """Reconstruct the frozen scalar bath copied independently to each site."""

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


def build_model():
    r"""Assemble $d_1-d_2$ with two independent copies of the same bath."""

    c = case()
    return coupled_pseudomode_dimer_model(load_fit(), U=c.U, t12=c.t12)


def _load_modes(
    data: np.lib.npyio.NpzFile,
    spin: str,
    policy: TolerancePolicy,
) -> NormalModeReference:
    diagnostics = json.loads(str(data[f"{spin}_diagnostics_json"].item()))
    canonicalization = diagnostics["canonicalization"]
    scalar_eta = diagnostics["scalar_eta"]
    return NormalModeReference(
        policy=policy,
        h=np.asarray(data[f"{spin}_h"]),
        eigenvalues=np.asarray(data[f"{spin}_eigenvalues"]),
        V=np.asarray(data[f"{spin}_V"]),
        V_inv=np.asarray(data[f"{spin}_V_inv"]),
        vacuum_mask=np.asarray(data[f"{spin}_vacuum_mask"]),
        mode_charges=np.asarray(data[f"{spin}_mode_charges"]),
        rapidities=np.asarray(data[f"{spin}_rapidities"]),
        original_order=np.asarray(data[f"{spin}_original_order"]),
        condition_number=np.float64(diagnostics["condition_number"]),
        condition_roundoff=np.float64(diagnostics["condition_roundoff"]),
        condition_limit=np.float64(diagnostics["condition_limit"]),
        occupation_gap=np.float64(diagnostics["occupation_gap"]),
        occupation_threshold=np.float64(diagnostics["occupation_threshold"]),
        eigen_residual=np.float64(diagnostics["eigen_residual"]),
        inverse_residual=np.float64(diagnostics["inverse_residual"]),
        scalar_eta=complex(scalar_eta["real"], scalar_eta["imag"]),
        scalar_cancellation=np.float64(diagnostics["scalar_cancellation"]),
        canonicalization_version=str(canonicalization["version"]),
        canonicalization_applied=bool(canonicalization["applied"]),
        canonicalization_trigger_threshold=np.float64(
            canonicalization["trigger_threshold"]
        ),
        canonicalization_neutral_defect_before=np.float64(
            canonicalization["neutral_defect_before"]
        ),
        canonicalization_neutral_defect_after=np.float64(
            canonicalization["neutral_defect_after"]
        ),
        canonicalization_relative_change=np.float64(
            canonicalization["relative_change"]
        ),
        canonicalization_chart_condition=np.float64(
            canonicalization["chart_condition"]
        ),
        canonicalization_overlap_condition=np.float64(
            canonicalization["overlap_condition"]
        ),
    )


def load_reference(model=None) -> HartreeFockReference:
    """Load the exact normal-mode frame used by the accepted CCSD amplitudes."""

    physical_model = build_model() if model is None else model
    path = ROOT / "checkpoints/U2_normal_modes.npz"
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema_version"].item()) != "cclens.hubbard_dimer.normal-modes.v1":
            raise ValueError("unsupported dimer normal-mode checkpoint")
        if float(data["U"]) != case().U:
            raise ValueError("normal-mode checkpoint interaction does not match")
        policy = TolerancePolicy.from_dict(json.loads(str(data["policy_json"].item())))
        return HartreeFockReference(
            model=physical_model,
            m_up=np.float64(data["m_up"]),
            m_down=np.float64(data["m_down"]),
            E_up=np.asarray(data["E_up"]),
            E_down=np.asarray(data["E_down"]),
            up=_load_modes(data, "up", policy),
            down=_load_modes(data, "down", policy),
            tolerance_policy=policy,
        )


def build_static_problem() -> ComponentStaticProblem:
    r"""Compile the dimer creator-rank two and four equations."""

    model = build_model()
    cc = factorized_ccsd_problem(
        model,
        m_up=0.5,
        build_next_rank_masks=False,
        reference=load_reference(model),
    )
    return ComponentStaticProblem.compile(cc)


def load_static() -> dict[str, np.ndarray]:
    """Load the accepted creator-rank two and four static CCSD root."""

    with np.load(ROOT / "checkpoints/U2.npz", allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def load_retained_response() -> dict[str, np.ndarray]:
    """Load the two-source retained $Q_1+Q_3$ response."""

    with np.load(ROOT / "data/retained_eom.npz", allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def load_comparison() -> dict[str, np.ndarray]:
    """Load PM and $M_{\rm total}=48$ TDVP matrices on the common grid."""

    with np.load(ROOT / "data/comparison.npz", allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def static_residual(*, problem: ComponentStaticProblem | None = None) -> float:
    """Reapply the packed tilde-real CCSD residual to the accepted dimer root."""

    p = build_static_problem() if problem is None else problem
    amplitudes = load_static()["amplitudes"]
    residual = component_static_residual(p, amplitudes).retained
    return float(np.linalg.norm(p.oracle.tilde.pack(residual, check=False)))


def solve_static(
    initial_amplitudes: np.ndarray | None = None,
    *,
    tolerance: float = 1.0e-8,
):
    """Solve the dimer CCSD equations from a supplied or saved seed."""

    problem = build_static_problem()
    initial = (
        load_static()["amplitudes"]
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
    *,
    spin: int = 0,
    time: np.ndarray | None = None,
    validation_omega: np.ndarray = np.asarray([0.0]),
    maximum_krylov_dimension: int = 160,
):
    r"""Recompute the two-source retained $S_1\oplus S_3$ dimer matrix."""

    saved = load_retained_response()
    t = saved["time"] if time is None else np.asarray(time, dtype=float)
    problem = build_static_problem()
    eom = eom_ccsd_problem(
        problem.oracle, response_spin=int(spin), build_next_rank_masks=False
    )
    response = ComponentEOMProblem.compile(eom, problem, charge=1)
    context = component_eom_context(response, load_static()["amplitudes"])
    sources = tuple(component_eom_sources(context, site=i) for i in (0, 1))
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
    """Check the physical topology, provenance scalars, and matrix normalization."""

    fit = load_fit()
    model = build_model()
    root = load_static()
    response = load_retained_response()
    comparison = load_comparison()
    fit_defect = max(
        float(np.max(abs(fit.H_greater + fit.H_lesser.conj()))),
        float(np.max(abs(fit.Gamma_greater - fit.Gamma_lesser.conj()))),
        float(np.max(abs(fit.g_greater - fit.g_lesser.conj()))),
    )
    if tuple(model.interaction_sites) != (0, 1):
        raise RuntimeError("both dimer sites must be interacting")
    if float(root["U"]) != case().U or int(response["sites"].size) != 2:
        raise RuntimeError("saved static/response data do not match the case")
    np.testing.assert_allclose(
        response["time"],
        comparison["time"],
        atol=4.0 * np.finfo(float).eps,
        rtol=0.0,
    )
    equal_time = float(np.max(abs(comparison["G_pseudomode"][0] + 1j * np.eye(2))))
    return {
        "fit_particle_hole_defect": fit_defect,
        "maximum_equal_time_defect_pseudomode": equal_time,
        "tdvp_points": int(comparison["time"].size),
        "spatial_orbitals": int(model.n_sites),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))

"""Tiny CCSD/EOM spectrum compared with exact finite-Lindblad ED.

Energy is measured in Gamma and time in Gamma^{-1}; hbar=1.  The explicit
m_up=0.5 makes this a reproduction-coordinate example for the spin-restricted
half-filled reference used by the SIAM paper examples. For general
multiorbital models, factorized_cc_problem constructs a full HF reference
by default.
"""

import numpy as np

from c3l import FermionicLindbladModel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
)
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.factorized import factorized_cc_problem
from c3l.cc.newton_krylov import NewtonKrylovPolicy
from c3l.cc.s135_full import (
    CompiledFullS135Action,
    FullS135ResponseOperator,
    rapidity_diagonal,
)
from siam_ed.physical_sparse import (
    compile_physical_sector_operator,
    physical_density_action,
    physical_response_action,
    physical_response_vectors,
    solve_trace_fixed_static,
)
from siam_ed.sparse_response import sparse_direct_resolvent


energy_unit = "Gamma"
U = 1.2
bath = np.array([-1.2, 1.2])
E = np.diag(np.concatenate(([-U / 2.0], bath))).astype(np.complex128)
E[0, 1:] = 0.28
E[1:, 0] = 0.28
Gamma_loss = np.diag([0.0, 0.06, 0.48])
Gamma_gain = np.diag([0.0, 0.48, 0.06])
model = FermionicLindbladModel(
    E,
    Gamma_loss,
    Gamma_gain,
    U=U,
    units=energy_unit,
)

# The current CC composition objects are a versioned advanced workflow.
cc = factorized_cc_problem(
    model,
    "CCSD",
    m_up=0.5,
    build_next_rank_masks=False,
)
static = ComponentStaticProblem.compile(cc)
policy = NewtonKrylovPolicy(
    tolerance=1.0e-9,
    gmres_restart=20,
    gmres_maximum_restart_cycles=5,
)
print("Computing static CCSD...", flush=True)
solution = solve_component_ccsd(
    static,
    policy=policy,
    compute_rank6=False,
)

# Retained S1 + S3 EOM-CCSD response.
omega = np.linspace(-4.0, 4.0, 161)
eta = 0.1
eom = eom_ccsd_problem(
    static.oracle,
    response_spin=0,
    build_next_rank_masks=False,
)
response_problem = ComponentEOMProblem.compile(eom, static, charge=1)
context = component_eom_context(response_problem, solution.amplitudes)
sources = component_eom_sources(context)

print("Computing CCSD-EOM (S1 + S3)...", flush=True)
response_s13 = component_retarded_green(
    ComponentSigmaOperator(context),
    sources,
    omega=omega,
    eta=eta,
    rtol=1.0e-9,
    restart=20,
    maxiter=100,
)
G_s13 = response_s13.addition + response_s13.removal
shifted_residual_s13 = max(
    solve.true_relative_residual
    for solve in response_s13.addition_solutions + response_s13.removal_solutions
)

print("Computing CCSD-EOM (S1 + S3 + S5)...", flush=True)
full_s135_operator = FullS135ResponseOperator(CompiledFullS135Action(context))
# The shifted solver uses the same rapidity-sum preconditioner for both
# response spaces; this does not alter the exact full-S135 matrix action.
full_s135_operator.diagonal = rapidity_diagonal(
    static.oracle.rapidities,
    full_s135_operator.coordinates,
)
response_s135 = component_retarded_green(
    full_s135_operator,
    sources,
    omega=omega,
    eta=eta,
    rtol=1.0e-9,
    restart=20,
    maxiter=100,
)
G_s135 = response_s135.addition + response_s135.removal
shifted_residual_s135 = max(
    solve.true_relative_residual
    for solve in response_s135.addition_solutions
    + response_s135.removal_solutions
)

# Exact response of the same finite Lindblad model.  The physical density
# matrix and response live in conserved Liouville sectors; no dense 4096 x
# 4096 Liouvillian is constructed.
print("Computing exact Lindblad ED...", flush=True)
density_operator = compile_physical_sector_operator(
    physical_density_action(model),
    maximum_bytes=1 << 30,
)
ness = solve_trace_fixed_static(density_operator)
response_operator = compile_physical_sector_operator(
    physical_response_action(model, spin=0, charge=1),
    maximum_bytes=1 << 30,
)
vectors = physical_response_vectors(
    density_operator.action.sector,
    ness.coefficients,
    model.orbital_index(model.impurity_site, 0),
)
ed_sources = np.column_stack((vectors.creation_left, vectors.creation_right))
ed_response = sparse_direct_resolvent(
    response_operator,
    ed_sources,
    omega.astype(np.complex128) + 1j * eta,
    verification_operator=response_operator.action,
)
ed_channels = ed_response.transfer(vectors.annihilation_sink)[:, 0, :]
G_ed = ed_channels[:, 0] + ed_channels[:, 1]

A_s13 = -G_s13.imag / np.pi
A_s135 = -G_s135.imag / np.pi
A_ed = -G_ed.imag / np.pi
maximum_error_s13 = float(np.max(np.abs(A_s13 - A_ed)))
maximum_error_s135 = float(np.max(np.abs(A_s135 - A_ed)))
relative_l2_error_s13 = float(
    np.sqrt(
        np.trapezoid(np.abs(A_s13 - A_ed) ** 2, omega)
        / np.trapezoid(np.abs(A_ed) ** 2, omega)
    )
)
relative_l2_error_s135 = float(
    np.sqrt(
        np.trapezoid(np.abs(A_s135 - A_ed) ** 2, omega)
        / np.trapezoid(np.abs(A_ed) ** 2, omega)
    )
)
weight_s13 = float(np.trapezoid(A_s13, omega))
weight_s135 = float(np.trapezoid(A_s135, omega))
weight_ed = float(np.trapezoid(A_ed, omega))

assert solution.converged and solution.packed_residual_norm < 1.0e-9
assert max(shifted_residual_s13, shifted_residual_s135) < 1.0e-8
assert max(ness.trace_defect, ness.hermiticity_defect, ness.scaled_residual) < 1.0e-12
assert ed_response.maximum_relative_residual < 1.0e-10

print(f"C3L example: CCSD/EOM versus exact Lindblad ED [{energy_unit}]")
print(f"final CCSD residual = {solution.packed_residual_norm:.3e}")
print(f"S1 + S3 maximum spectral error = {maximum_error_s13:.6e}")
print(f"S1 + S3 + S5 maximum spectral error = {maximum_error_s135:.6e}")
print(f"S1 + S3 relative L2 spectral error = {relative_l2_error_s13:.6e}")
print(
    "S1 + S3 + S5 relative L2 spectral error = "
    f"{relative_l2_error_s135:.6e}"
)
print(
    "finite-window spectral integrals [-4, 4] = "
    f"ED {weight_ed:.6f}, S1 + S3 {weight_s13:.6f}, "
    f"S1 + S3 + S5 {weight_s135:.6f}"
)

np.savez_compressed("ccsd_eom.npz", omega=omega, A_ed=A_ed, A_s13=A_s13, A_s135=A_s135)
print("Saved ccsd_eom.npz")

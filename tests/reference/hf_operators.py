"""Explicit doubled-Fock HF operators for tiny fermionic-sign tests."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from c3l.ci.reference import HartreeFockReference, _immutable_complex, ComplexMatrix
from .physical_fock import _require_dense_allocation, interaction_hamiltonian
from .mapping import doubled_annihilation_operators, fermionic_trace_bra
from .generator import quadratic_mode_matrix


def global_particle_hole_operators(
    reference: HartreeFockReference,
    max_dense_bytes: int | None = None,
) -> tuple[tuple[ComplexMatrix, ...], tuple[ComplexMatrix, ...]]:
    """Embed all reference ``P,bar_P`` modes in the global doubled JW space.

    The returned order is the M3 order: the ``2n`` up modes followed by the
    ``2n`` down modes when present. All matrices are complex128 and read-only.
    """

    if not isinstance(reference, HartreeFockReference):
        raise TypeError("reference must be a HartreeFockReference")
    n = reference.n_sites
    doubled_dimension = 1 << reference.num_ph_modes
    memory_limit = (
        reference.tolerance_policy.dense_memory_limit_bytes
        if max_dense_bytes is None
        else max_dense_bytes
    )
    if (
        isinstance(memory_limit, bool)
        or not isinstance(memory_limit, int)
        or memory_limit <= 0
    ):
        raise ValueError("max_dense_bytes must be a positive integer")
    if memory_limit > reference.tolerance_policy.dense_memory_limit_bytes:
        raise ValueError("max_dense_bytes may only tighten the reference policy")
    _require_dense_allocation(
        (doubled_dimension, doubled_dimension),
        "global P/bar_P operators and construction workspace",
        memory_limit,
        copies=6 * reference.num_ph_modes + 4,
    )
    normal, tilde = doubled_annihilation_operators(
        reference.model.n_spin_orbitals, max_dense_bytes=memory_limit
    )
    P: list[ComplexMatrix] = []
    bar_P: list[ComplexMatrix] = []
    for spin, spin_reference in enumerate(reference.mode_references):
        sharp = tuple(
            normal[reference.model.orbital_index(site, spin)].conj().T
            for site in range(n)
        ) + tuple(tilde[reference.model.orbital_index(site, spin)] for site in range(n))
        dual = tuple(
            normal[reference.model.orbital_index(site, spin)] for site in range(n)
        ) + tuple(
            tilde[reference.model.orbital_index(site, spin)].conj().T
            for site in range(n)
        )
        for p, occupied in enumerate(spin_reference.vacuum_mask):
            xi = np.zeros_like(normal[0])
            bar_xi = np.zeros_like(normal[0])
            for a in range(2 * n):
                xi += spin_reference.V_inv[p, a] * dual[a]
                bar_xi += sharp[a] * spin_reference.V[a, p]
            if occupied:
                P.append(_immutable_complex(bar_xi))
                bar_P.append(_immutable_complex(xi))
            else:
                P.append(_immutable_complex(xi))
                bar_P.append(_immutable_complex(bar_xi))
    return tuple(P), tuple(bar_P)


@dataclass(frozen=True)
class GlobalGaussianVacuum:
    """Tiny doubled-Fock reference state with its physical trace functional."""

    state: np.ndarray
    trace_bra: np.ndarray


def global_gaussian_vacuum(reference: HartreeFockReference) -> GlobalGaussianVacuum:
    """Solve the common annihilator nullspace independently of CC amplitudes."""

    policy = reference.tolerance_policy
    dimension = 1 << reference.num_ph_modes
    _require_dense_allocation(
        (dimension, dimension),
        "Gaussian-vacuum operators, constraints, and SVD",
        policy.dense_memory_limit_bytes,
        copies=5 * reference.num_ph_modes + 8,
    )
    annihilators, _ = global_particle_hole_operators(reference)
    _, singular_values, right_adjoint = scipy.linalg.svd(
        np.vstack(annihilators), full_matrices=False
    )
    threshold = policy.algebra_abs_rel.threshold(singular_values[0])
    null_indices = np.flatnonzero(singular_values <= threshold)
    if null_indices.size != 1:
        raise ValueError("global Gaussian vacuum must have nullity one")
    state = right_adjoint.conj().T[:, null_indices[0]]
    trace_bra = fermionic_trace_bra(
        reference.model.n_spin_orbitals,
        max_dense_bytes=policy.dense_memory_limit_bytes,
    )
    overlap = trace_bra @ state
    if abs(overlap) <= policy.algebra_abs_rel.threshold(np.linalg.norm(trace_bra)):
        raise ValueError("global Gaussian vacuum has no resolvable trace overlap")
    state /= overlap
    residual = max(np.linalg.norm(operator @ state) for operator in annihilators)
    if residual > policy.algebra_abs_rel.threshold(1.0):
        raise ValueError("global Gaussian annihilation residual exceeds policy")
    return GlobalGaussianVacuum(_immutable_complex(state), _immutable_complex(trace_bra))


def reference_decomposition_paper_K(reference: HartreeFockReference) -> ComplexMatrix:
    """Dense charge-only reference plus physical counterterms and interaction."""

    if reference.model.symmetry != "charge":
        raise ValueError("this dense test decomposition requires charge symmetry")
    annihilators, creators = global_particle_hole_operators(reference)
    generator = np.zeros_like(annihilators[0])
    for rapidity, annihilator, creator in zip(
        reference.up.rapidities, annihilators, creators, strict=True
    ):
        generator += rapidity * (creator @ annihilator)
    normal, tilde = doubled_annihilation_operators(
        reference.model.n_spin_orbitals,
        max_dense_bytes=reference.tolerance_policy.dense_memory_limit_bytes,
    )
    energy_delta = reference.model.E - reference.E_up
    reference_loss = reference.reference_gamma_loss
    reference_gain = reference.reference_gamma_gain
    loss_delta = reference.model.gamma_loss - (
        reference.model.gamma_loss if reference_loss is None else reference_loss
    )
    gain_delta = reference.model.gamma_gain - (
        reference.model.gamma_gain if reference_gain is None else reference_gain
    )
    mode_delta = quadratic_mode_matrix(energy_delta, loss_delta, gain_delta)
    sharp = tuple(operator.conj().T for operator in normal) + tuple(tilde)
    dual = tuple(normal) + tuple(operator.conj().T for operator in tilde)
    for row, sharp_operator in enumerate(sharp):
        for column, dual_operator in enumerate(dual):
            generator += mode_delta[row, column] * (sharp_operator @ dual_operator)
    generator -= np.trace(energy_delta + 1j * (loss_delta + gain_delta)) * np.eye(
        generator.shape[0]
    )
    generator += interaction_hamiltonian(reference.model.interaction, normal)
    for first, second, third, fourth, value in reference.model.interaction.terms:
        generator -= (
            value.conjugate()
            * tilde[first].conj().T @ tilde[second].conj().T
            @ tilde[fourth] @ tilde[third]
        )
    return generator


def creation_matrix(mask: int, num_modes: int) -> ComplexMatrix:
    """Build a canonical creator string directly in the tiny determinant space."""

    if not 0 < mask < (1 << num_modes):
        raise ValueError("creation mask is outside the mode space")
    dimension = 1 << num_modes
    _require_dense_allocation((dimension, dimension), "test creator string", None)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for source in range(dimension):
        if source & mask:
            continue
        target = source
        sign = 1
        for mode in reversed(range(num_modes)):
            if mask & (1 << mode):
                if (target & ((1 << mode) - 1)).bit_count() % 2:
                    sign = -sign
                target |= 1 << mode
        matrix[target, source] = sign
    return matrix

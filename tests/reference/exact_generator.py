"""Independent exact physical generators in plain column-major vectorization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from c3l.models import FermionicLindbladModel, QuadraticOpenSystem

from .physical_fock import (
    _require_dense_allocation,
    annihilation_operators,
    auxiliary_hamiltonian,
    expand_spin_diagonal,
    quadratic_hamiltonian,
)

ComplexMatrix = NDArray[np.complex128]
PhysicalModel = QuadraticOpenSystem | FermionicLindbladModel


def _model_fock_dimensions(model: PhysicalModel) -> tuple[int, int, int]:
    if isinstance(model, QuadraticOpenSystem):
        num_orbitals = model.n_orbitals
    elif isinstance(model, FermionicLindbladModel):
        num_orbitals = model.n_spin_orbitals
    else:
        raise TypeError(
            "model must be a validated QuadraticOpenSystem or FermionicLindbladModel"
        )
    dimension = 1 << num_orbitals
    return num_orbitals, dimension, dimension * dimension


def _require_model_generator_working_set(
    model: PhysicalModel,
    label: str,
    max_dense_bytes: int | None,
    *,
    extra_fock_matrices: int = 0,
) -> None:
    """Gate live Fock inputs together with generator/Kronecker workspaces."""

    num_orbitals, dimension, vector_dimension = _model_fock_dimensions(model)
    elements = (
        4 * vector_dimension * vector_dimension
        + (num_orbitals + 2 + extra_fock_matrices) * dimension * dimension
    )
    _require_dense_allocation((elements,), label, max_dense_bytes)


def _validated_operator_shapes(
    H: ArrayLike,
    annihilation: Sequence[ComplexMatrix],
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
) -> tuple[ComplexMatrix, ComplexMatrix, ComplexMatrix]:
    hamiltonian = np.asarray(H, dtype=np.complex128)
    loss = np.asarray(gamma_loss, dtype=np.complex128)
    gain = np.asarray(gamma_gain, dtype=np.complex128)
    num_orbitals = len(annihilation)
    if num_orbitals == 0:
        raise ValueError("at least one physical annihilator is required")
    dimension = annihilation[0].shape[0]
    if hamiltonian.shape != (dimension, dimension):
        raise ValueError("H shape does not match the physical-Fock dimension")
    if loss.shape != (num_orbitals, num_orbitals):
        raise ValueError("gamma_loss shape does not match the orbital count")
    if gain.shape != (num_orbitals, num_orbitals):
        raise ValueError("gamma_gain shape does not match the orbital count")
    for operator in annihilation:
        if operator.shape != (dimension, dimension):
            raise ValueError("all annihilators must have the same square shape")
    return hamiltonian, loss, gain


def _physical_generator_indexed_arrays(
    H: ArrayLike,
    annihilation: Sequence[ComplexMatrix],
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    *,
    max_dense_bytes: int | None = None,
) -> ComplexMatrix:
    """Assemble the indexed-Gamma generator without factorizing Gamma.

    Loss contributes ``Gamma_ij (2 c_j rho c_i^dagger - {c_i^dagger
    c_j,rho})`` and gain contributes ``Gamma_ij (2 c_i^dagger rho c_j -
    {c_j c_i^dagger,rho})``.  Kronecker factors follow column-major vec.
    """

    hamiltonian, loss, gain = _validated_operator_shapes(
        H, annihilation, gamma_loss, gamma_gain
    )
    dimension = hamiltonian.shape[0]
    _require_dense_allocation(
        (dimension * dimension, dimension * dimension),
        "indexed physical generator and Kronecker workspace",
        max_dense_bytes,
        copies=4,
    )
    identity = np.eye(dimension, dtype=np.complex128)
    physical_generator = -1j * (
        np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity)
    )

    for i, annihilation_i in enumerate(annihilation):
        creation_i = annihilation_i.conj().T
        for j, annihilation_j in enumerate(annihilation):
            loss_number = creation_i @ annihilation_j
            physical_generator += loss[i, j] * (
                2.0 * np.kron(annihilation_i.conj(), annihilation_j)
                - np.kron(identity, loss_number)
                - np.kron(loss_number.T, identity)
            )

            gain_number = annihilation_j @ creation_i
            physical_generator += gain[i, j] * (
                2.0 * np.kron(annihilation_j.T, creation_i)
                - np.kron(identity, gain_number)
                - np.kron(gain_number.T, identity)
            )
    return np.asarray(physical_generator, dtype=np.complex128)


def _direct_indexed_action_arrays(
    rho: ArrayLike,
    H: ArrayLike,
    annihilation: Sequence[ComplexMatrix],
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
) -> ComplexMatrix:
    """Apply the defining indexed master equation by dense matrix products.

    This action does not call either vectorized generator constructor and is a
    separate audit of every matrix index and anticommutator.
    """

    hamiltonian, loss, gain = _validated_operator_shapes(
        H, annihilation, gamma_loss, gamma_gain
    )
    density = np.asarray(rho, dtype=np.complex128)
    if density.shape != hamiltonian.shape:
        raise ValueError("rho shape does not match H")
    action = -1j * (hamiltonian @ density - density @ hamiltonian)
    for i, annihilation_i in enumerate(annihilation):
        creation_i = annihilation_i.conj().T
        for j, annihilation_j in enumerate(annihilation):
            loss_number = creation_i @ annihilation_j
            action += loss[i, j] * (
                2.0 * annihilation_j @ density @ creation_i
                - loss_number @ density
                - density @ loss_number
            )
            gain_number = annihilation_j @ creation_i
            action += gain[i, j] * (
                2.0 * creation_i @ density @ annihilation_j
                - gain_number @ density
                - density @ gain_number
            )
    return np.asarray(action, dtype=np.complex128)


def _psd_factor(gamma: ArrayLike) -> ComplexMatrix:
    """Return ``B = sqrt(w) Q^dagger`` so ``gamma = B^dagger B``.

    The input itself is not modified.  Negative eigenvalues beyond an
    eigensolver-roundoff allowance are rejected; validated model inputs should
    already have recorded and removed any input-level PSD defect.
    """

    matrix = np.asarray(gamma, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gamma must be square")
    scale = max(float(np.linalg.norm(matrix, ord=2)), 1.0)
    roundoff = np.float64(64.0 * np.finfo(np.float64).eps * scale)
    hermiticity_defect = np.linalg.norm(matrix - matrix.conj().T, ord="fro")
    if hermiticity_defect > roundoff:
        raise ValueError("gamma must be Hermitian before jump factorization")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues[0] < -roundoff:
        raise ValueError("gamma must be positive semidefinite before factorization")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return np.asarray(
        np.sqrt(eigenvalues)[:, None] * eigenvectors.conj().T, dtype=np.complex128
    )


def jumps_from_factors(
    B_loss: ArrayLike,
    B_gain: ArrayLike,
    annihilation: Sequence[ComplexMatrix],
) -> tuple[tuple[ComplexMatrix, ...], tuple[ComplexMatrix, ...]]:
    """Build sqrt(2)-scaled loss/gain jumps from declared Lindblad factors.

    Gain uses ``B_gain[alpha,i].conj() * c_i^dagger``; this conjugation fixes
    the coefficient of ``c_i^dagger rho c_j`` for complex Gamma.
    """

    loss_factor = np.asarray(B_loss, dtype=np.complex128)
    gain_factor = np.asarray(B_gain, dtype=np.complex128)
    num_orbitals = len(annihilation)
    if loss_factor.ndim != 2 or loss_factor.shape[1] != num_orbitals:
        raise ValueError("B_loss columns must match the orbital count")
    if gain_factor.ndim != 2 or gain_factor.shape[1] != num_orbitals:
        raise ValueError("B_gain columns must match the orbital count")
    dimension = annihilation[0].shape[0]
    scale = np.sqrt(np.float64(2.0))

    loss_jumps: list[ComplexMatrix] = []
    for row in loss_factor:
        jump = np.zeros((dimension, dimension), dtype=np.complex128)
        for coefficient, operator in zip(row, annihilation, strict=True):
            jump += scale * coefficient * operator
        loss_jumps.append(jump)

    gain_jumps: list[ComplexMatrix] = []
    for row in gain_factor:
        jump = np.zeros((dimension, dimension), dtype=np.complex128)
        for coefficient, operator in zip(row, annihilation, strict=True):
            jump += scale * coefficient.conjugate() * operator.conj().T
        gain_jumps.append(jump)
    return tuple(loss_jumps), tuple(gain_jumps)


def _factorized_jump_operators(
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    annihilation: Sequence[ComplexMatrix],
) -> tuple[tuple[ComplexMatrix, ...], tuple[ComplexMatrix, ...]]:
    """Factor PSD Gamma matrices and return their physical jump lists."""

    return jumps_from_factors(
        _psd_factor(gamma_loss), _psd_factor(gamma_gain), annihilation
    )


def physical_generator_from_jumps(
    H: ArrayLike,
    jumps: Sequence[ComplexMatrix],
    *,
    max_dense_bytes: int | None = None,
) -> ComplexMatrix:
    """Assemble a standard Lindblad generator from physical jump operators."""

    hamiltonian = np.asarray(H, dtype=np.complex128)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("H must be square")
    hermiticity_defect = np.linalg.norm(hamiltonian - hamiltonian.conj().T, ord="fro")
    hermiticity_scale = max(float(np.linalg.norm(hamiltonian, ord="fro")), 1.0)
    if hermiticity_defect > 64.0 * np.finfo(np.float64).eps * hermiticity_scale:
        raise ValueError("H must be Hermitian")
    dimension = hamiltonian.shape[0]
    vector_dimension = dimension * dimension
    fock_matrix_copies = 2 * (len(jumps) + 1) + 1
    _require_dense_allocation(
        (
            4 * vector_dimension * vector_dimension
            + fock_matrix_copies * dimension * dimension,
        ),
        "cumulative jump-factorized inputs and Kronecker workspaces",
        max_dense_bytes,
    )
    identity = np.eye(dimension, dtype=np.complex128)
    physical_generator = -1j * (
        np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity)
    )
    for jump in jumps:
        operator = np.asarray(jump, dtype=np.complex128)
        if operator.shape != hamiltonian.shape:
            raise ValueError("jump shape does not match H")
        jump_number = operator.conj().T @ operator
        physical_generator += (
            np.kron(operator.conj(), operator)
            - 0.5 * np.kron(identity, jump_number)
            - 0.5 * np.kron(jump_number.T, identity)
        )
    return np.asarray(physical_generator, dtype=np.complex128)


def _physical_generator_jump_factorized_arrays(
    H: ArrayLike,
    annihilation: Sequence[ComplexMatrix],
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    *,
    max_dense_bytes: int | None = None,
) -> ComplexMatrix:
    """Build the exact generator through independent PSD jump factors."""

    loss_jumps, gain_jumps = _factorized_jump_operators(
        gamma_loss, gamma_gain, annihilation
    )
    return physical_generator_from_jumps(
        H, loss_jumps + gain_jumps, max_dense_bytes=max_dense_bytes
    )


def paper_K(physical_generator: ArrayLike) -> ComplexMatrix:
    """Convert the stored time generator to the paper convention ``K=iG``."""

    generator = np.asarray(physical_generator, dtype=np.complex128)
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("physical_generator must be square")
    return np.asarray(1j * generator, dtype=np.complex128)


def quadratic_exact_inputs(
    model: QuadraticOpenSystem,
    *,
    max_dense_bytes: int | None = None,
) -> tuple[ComplexMatrix, tuple[ComplexMatrix, ...], ComplexMatrix, ComplexMatrix]:
    """Construct physical-Fock inputs for a validated quadratic model."""

    dimension = 1 << model.n_orbitals
    _require_dense_allocation(
        ((model.n_orbitals + 1) * dimension * dimension,),
        "cumulative quadratic physical-Fock inputs",
        max_dense_bytes,
    )
    annihilation = annihilation_operators(
        model.n_orbitals, max_dense_bytes=max_dense_bytes
    )
    H = quadratic_hamiltonian(model.E, annihilation)
    return H, annihilation, model.gamma_loss, model.gamma_gain


def auxiliary_exact_inputs(
    model: FermionicLindbladModel,
    *,
    max_dense_bytes: int | None = None,
) -> tuple[ComplexMatrix, tuple[ComplexMatrix, ...], ComplexMatrix, ComplexMatrix]:
    """Construct physical-Fock inputs in the model's physical orbital space."""

    dimension = 1 << model.n_spin_orbitals
    _require_dense_allocation(
        (
            (model.n_spin_orbitals + 1) * dimension * dimension
            + 2 * model.n_spin_orbitals * model.n_spin_orbitals,
        ),
        "cumulative auxiliary physical-Fock inputs",
        max_dense_bytes,
    )
    annihilation = annihilation_operators(
        model.n_spin_orbitals, max_dense_bytes=max_dense_bytes
    )
    H = auxiliary_hamiltonian(model, annihilation)
    if model.symmetry == "charge":
        return H, annihilation, model.gamma_loss, model.gamma_gain
    return (
        H,
        annihilation,
        expand_spin_diagonal(model.gamma_loss),
        expand_spin_diagonal(model.gamma_gain),
    )


def _model_exact_inputs(
    model: PhysicalModel, max_dense_bytes: int | None
) -> tuple[ComplexMatrix, tuple[ComplexMatrix, ...], ComplexMatrix, ComplexMatrix]:
    if isinstance(model, QuadraticOpenSystem):
        return quadratic_exact_inputs(model, max_dense_bytes=max_dense_bytes)
    if isinstance(model, FermionicLindbladModel):
        return auxiliary_exact_inputs(model, max_dense_bytes=max_dense_bytes)
    raise TypeError(
        "model must be a validated QuadraticOpenSystem or FermionicLindbladModel"
    )


def physical_generator_indexed(
    model: PhysicalModel, *, max_dense_bytes: int | None = None
) -> ComplexMatrix:
    """Build the direct indexed-Gamma generator from a validated model."""

    _require_model_generator_working_set(
        model,
        "cumulative indexed physical-generator inputs and workspaces",
        max_dense_bytes,
    )
    H, annihilation, gamma_loss, gamma_gain = _model_exact_inputs(
        model, max_dense_bytes
    )
    return _physical_generator_indexed_arrays(
        H,
        annihilation,
        gamma_loss,
        gamma_gain,
        max_dense_bytes=max_dense_bytes,
    )


def direct_indexed_action(
    rho: ArrayLike,
    model: PhysicalModel,
    *,
    max_dense_bytes: int | None = None,
) -> ComplexMatrix:
    """Apply the indexed master equation independently for a validated model."""

    num_orbitals, dimension, _ = _model_fock_dimensions(model)
    _require_dense_allocation(
        ((num_orbitals + 7) * dimension * dimension,),
        "cumulative direct indexed-action inputs and workspaces",
        max_dense_bytes,
    )
    H, annihilation, gamma_loss, gamma_gain = _model_exact_inputs(
        model, max_dense_bytes
    )
    return _direct_indexed_action_arrays(rho, H, annihilation, gamma_loss, gamma_gain)


def physical_generator_jump_factorized(
    model: PhysicalModel, *, max_dense_bytes: int | None = None
) -> ComplexMatrix:
    """Build the independent sqrt(2)-jump generator from a validated model."""

    num_orbitals, _, _ = _model_fock_dimensions(model)
    _require_model_generator_working_set(
        model,
        "cumulative jump-factorized generator inputs and workspaces",
        max_dense_bytes,
        extra_fock_matrices=2 * num_orbitals,
    )
    H, annihilation, gamma_loss, gamma_gain = _model_exact_inputs(
        model, max_dense_bytes
    )
    return _physical_generator_jump_factorized_arrays(
        H,
        annihilation,
        gamma_loss,
        gamma_gain,
        max_dense_bytes=max_dense_bytes,
    )

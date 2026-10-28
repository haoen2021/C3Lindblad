"""Factorized CCSD residual and Jacobian-vector actions.

The model, reference modes, and factorized interaction products define the
CC problem. For checking the tensor contractions, residuals can also be
evaluated as sparse creator polynomials. For each particle-hole linear form

``F = sum_p (f_bar[p] bar_P[p] + f_P[p] P[p])``

it uses the exact pure-creation similarity transform

``exp(-T) F exp(T) = F + sum_p f_P[p] [P[p], T]``.

The sparse factorized two-body products are transformed factor by factor and
applied to the vacuum.  Projection is made only after a complete product.
For CCSD, only creator ranks two and four are needed by the amplitude residual;
creator rank six supplies the immediate next-rank diagnostic.

The public residual uses the physical generator ``L = -1j * paper_K``.  This
makes the residual tilde even, so amplitudes and equations use the same independent real orbit
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as cartesian_product
from math import comb
from time import perf_counter
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from c3l.ci.basis import ness_basis
from c3l.ci.matrix_free import (
    FactorizedModeProduct,
    LinearModeOperator,
    _interaction_products,
    physical_mode_operator,
)
from c3l.ci.reference import (
    HartreeFockReference,
    full_hartree_fock_reference,
    half_filled_hartree_fock_reference,
)
from c3l.ci.sectors import mode_layout_from_references
from c3l.ci.tilde import tilde_map_from_references
from c3l.gaussian_mean_field import solve_ghf
from c3l.models import FermionicLindbladModel
from c3l.superfermion.generator import quadratic_mode_matrix

from .basis import ClusterBasis
from .symmetry import TildeMap, tilde_map


ComplexVector = NDArray[np.complex128]
LinearKey = tuple[tuple[str, int, complex], ...]
Polynomial = dict[int, complex]


def _immutable_complex(values: np.ndarray) -> ComplexVector:
    array = np.ascontiguousarray(values, dtype=np.complex128)
    frozen = np.frombuffer(array.tobytes(), dtype=np.complex128).reshape(array.shape)
    frozen.flags.writeable = False
    return frozen


@dataclass(frozen=True, slots=True)
class FactorizedCCProblem:
    """CCS/CCSD amplitudes and low-rank data without a full sector.

    ``cluster_basis`` contains deterministic retained masks and
    ``next_rank_masks`` only the immediate omitted shell.  The complete
    charge-zero dimension is metadata used to prove that this scalable path
    did not allocate that representation.
    """

    level: str
    model: FermionicLindbladModel
    reference: HartreeFockReference
    cluster_basis: ClusterBasis
    tilde: TildeMap
    rapidities: ComplexVector
    products: tuple[FactorizedModeProduct, ...]
    next_rank_masks: tuple[int, ...]
    next_rank_dimension_count: int
    retained_index: Mapping[int, int]
    next_rank_index: Mapping[int, int]
    complete_sector_dimension: int
    maximum_basis_dimension_constructed: int

    def __post_init__(self) -> None:
        expected = {"CCS": 2, "CCSD": 4}
        if self.level not in expected:
            raise ValueError("factorized level must be CCS or CCSD")
        available_rank = min(expected[self.level], 2 * self.model.n_spin_orbitals)
        if self.cluster_basis.maximum_rank != available_rank:
            raise ValueError("cluster creator rank disagrees with the CC level")
        rapidities = np.asarray(self.rapidities, dtype=np.complex128)
        if rapidities.shape != (self.cluster_basis.num_modes,):
            raise ValueError("rapidity count disagrees with the particle-hole modes")
        if not np.all(np.isfinite(rapidities)):
            raise ValueError("rapidities must be finite")
        if self.tilde.dimension != self.cluster_basis.dimension:
            raise ValueError("tilde map and cluster amplitude dimensions disagree")
        if any(
            mask.bit_count() != self.next_rank_creator_rank
            for mask in self.next_rank_masks
        ):
            raise ValueError("the immediate omitted shell has the wrong creator rank")
        if self.next_rank_dimension_count < len(self.next_rank_masks):
            raise ValueError("next-rank dimension metadata is smaller than stored masks")
        if self.maximum_basis_dimension_constructed > self.complete_sector_dimension:
            raise ValueError("constructed basis dimension exceeds the complete sector")
        object.__setattr__(self, "rapidities", _immutable_complex(rapidities))
        object.__setattr__(self, "retained_index", MappingProxyType(dict(self.retained_index)))
        object.__setattr__(self, "next_rank_index", MappingProxyType(dict(self.next_rank_index)))

    @property
    def num_amplitudes(self) -> int:
        return self.cluster_basis.dimension

    @property
    def num_variables(self) -> int:
        return self.tilde.real_size

    @property
    def next_rank_dimension(self) -> int:
        return self.next_rank_dimension_count

    @property
    def retained_creator_rank(self) -> int:
        return {"CCS": 2, "CCSD": 4}[self.level]

    @property
    def next_rank_creator_rank(self) -> int:
        return self.retained_creator_rank + 2

    def construction_record(self) -> dict[str, int | bool | str]:
        """Machine-readable evidence for the scalable representation boundary."""

        return {
            "level": self.level,
            "operator_convention": "physical_generator_L_equals_minus_i_paper_K",
            "normal_mode_canonicalization_version": (
                self.reference.up.canonicalization_version
            ),
            "normal_mode_canonicalization_applied_up": bool(
                self.reference.up.canonicalization_applied
            ),
            "normal_mode_canonicalization_applied_down": bool(
                self.reference.down is not None and self.reference.down.canonicalization_applied
            ),
            "reference_quadratic_separated": bool(
                not np.array_equal(self.model.E, self.reference.E_up)
                or (self.reference.E_down is not None and not np.array_equal(self.model.E, self.reference.E_down))
            ),
            "num_modes": self.cluster_basis.num_modes,
            "complex_amplitude_dimension": self.num_amplitudes,
            "independent_real_dimension": self.num_variables,
            "next_rank_dimension": self.next_rank_dimension,
            "maximum_basis_dimension_constructed": self.maximum_basis_dimension_constructed,
            "complete_charge_zero_sector_dimension": self.complete_sector_dimension,
            "complete_sector_basis_constructed": self.maximum_basis_dimension_constructed == self.complete_sector_dimension,
            "complete_sector_state_vector_constructed": False,
            "dense_jacobian_constructed": False,
        }


def _quadratic_dissipation_counterterms(
    model: FermionicLindbladModel,
    reference: HartreeFockReference,
) -> tuple[FactorizedModeProduct, ...]:
    """Return the exact physical-minus-reference quadratic generator."""

    if reference.reference_gamma_loss is None or reference.reference_gamma_gain is None:
        raise ValueError("quadratic dissipation counterterms require a regulated reference")
    reference_loss = np.asarray(reference.reference_gamma_loss)
    reference_gain = np.asarray(reference.reference_gamma_gain)
    physical_loss = np.asarray(model.gamma_loss)
    physical_gain = np.asarray(model.gamma_gain)
    n = model.n_sites
    products: list[FactorizedModeProduct] = []
    operators: dict[tuple[int, int, str], LinearModeOperator] = {}

    def operator(site: int, spin: int, channel: str) -> LinearModeOperator:
        key = (site, spin, channel)
        if key not in operators:
            operators[key] = physical_mode_operator(reference, site, spin, channel)
        return operators[key]

    scalar = 0.0 + 0.0j
    for spin, reference_E in enumerate(reference.one_body_blocks):
        delta_h = quadratic_mode_matrix(
            np.asarray(model.E), physical_loss, physical_gain
        ) - quadratic_mode_matrix(reference_E, reference_loss, reference_gain)
        scalar -= np.trace(
            np.asarray(model.E)
            - reference_E
            + 1j
            * (
                physical_loss
                + physical_gain
                - reference_loss
                - reference_gain
            )
        )
        for a in range(2 * n):
            sharp_site = a if a < n else a - n
            sharp_channel = "c_dagger" if a < n else "tilde_c"
            for b in range(2 * n):
                coefficient = complex(delta_h[a, b])
                if coefficient == 0.0:
                    continue
                dual_site = b if b < n else b - n
                dual_channel = "c" if b < n else "tilde_c_dagger"
                products.append(
                    FactorizedModeProduct(
                        f"quadratic_counter:spin_{spin}_{a}_{b}",
                        coefficient,
                        (
                            operator(sharp_site, spin, sharp_channel),
                            operator(dual_site, spin, dual_channel),
                        ),
                    )
                )
    if scalar != 0.0:
        c_dagger = operator(0, 0, "c_dagger")
        c = operator(0, 0, "c")
        products.extend(
            (
                FactorizedModeProduct(
                    "quadratic_counter:scalar_c_dagger_c",
                    complex(scalar),
                    (c_dagger, c),
                ),
                FactorizedModeProduct(
                    "quadratic_counter:scalar_c_c_dagger",
                    complex(scalar),
                    (c, c_dagger),
                ),
            )
        )
    return tuple(products)


def _factorized_cc_problem(
    model: FermionicLindbladModel,
    level: str,
    *,
    m_up: float,
    build_next_rank_masks: bool,
    reference_E: ArrayLike | None,
    reference: HartreeFockReference | None,
    reference_mode: str,
) -> FactorizedCCProblem:
    """Build deterministic retained/next shells without a full CI basis."""

    if not isinstance(model, FermionicLindbladModel):
        raise TypeError("factorized CC requires a FermionicLindbladModel")
    retained_ph_rank = {"CCS": 1, "CCSD": 2}.get(level)
    if retained_ph_rank is None:
        raise ValueError("factorized level must be CCS or CCSD")
    if reference_mode not in ("full_hf", "legacy_half_filled"):
        raise ValueError("reference_mode must be full_hf or legacy_half_filled")
    if model.symmetry == "spin" and any(
        sorted((p % 2, q % 2)) != sorted((r % 2, s % 2))
        for p, q, r, s, _ in model.interaction.terms
    ):
        raise ValueError(
            "the current component coordinates require separate up/down conservation"
        )
    if reference is not None:
        if reference_E is not None:
            raise ValueError("reference and reference_E are mutually exclusive")
        if not isinstance(reference, HartreeFockReference):
            raise TypeError("reference must be a HartreeFockReference")
        if reference.model is not model:
            raise ValueError("explicit reference must belong to the supplied model")
        reference_model = model
    elif reference_E is not None:
        if model.symmetry == "charge":
            raise ValueError("charge-only calculations require a full HF reference")
        if model.two_body_interaction is not None:
            raise ValueError(
                "general interactions require an actual full-HF reference, not reference_E"
            )
        reference_model = FermionicLindbladModel(
            reference_E,
            model.gamma_loss,
            model.gamma_gain,
            U=model.U,
            impurity_site=model.impurity_site,
            units=model.units,
            input_tolerance=model.input_tolerance,
            interaction_sites=model.interaction_sites,
            density_interaction=model.density_interaction,
        )
        reference = half_filled_hartree_fock_reference(reference_model, m_up)
    else:
        reference_model = model
        if reference_mode == "full_hf":
            hf = solve_ghf(
                model.physical_E,
                model.interaction,
                model.physical_gamma_loss,
                model.physical_gamma_gain,
                tolerance=5.0e-13,
                max_iterations=400,
            )
            if (
                not hf.converged
                or hf.self_consistency_residual > 1.0e-10
                or hf.lyapunov_residual > 1.0e-10
                or hf.physicality_defect > 1.0e-10
            ):
                raise RuntimeError("the default general full-HF reference did not converge")
            reference = full_hartree_fock_reference(
                model,
                hf.fock,
                hf.gamma,
            )
        else:
            if model.symmetry == "charge":
                raise ValueError("charge-only calculations require reference_mode='full_hf'")
            reference = half_filled_hartree_fock_reference(reference_model, m_up)
    layout = mode_layout_from_references(reference.up, reference.down)

    # The largest determinant object built here is the immediate omitted shell.
    # In particular, ness_basis(n, 2*n, ...) is never called.
    retained = ness_basis(model.n_sites, retained_ph_rank, layout)
    masks = retained.masks[1:]
    cluster = ClusterBasis(
        layout.num_modes,
        masks,
        tuple(2 * rank for rank in retained.ranks[1:]),
    )
    n = model.n_sites
    next_pair_rank = retained_ph_rank + 1
    next_dimension = comb(n, next_pair_rank) ** 2 if model.symmetry == "charge" else sum(
        comb(n, up_pairs) ** 2 * comb(n, next_pair_rank - up_pairs) ** 2
        for up_pairs in range(next_pair_rank + 1)
    )
    if build_next_rank_masks:
        through_next = ness_basis(model.n_sites, next_pair_rank, layout)
        next_masks = tuple(
            mask
            for mask, rank in zip(through_next.masks, through_next.ranks, strict=True)
            if rank == next_pair_rank
        )
        maximum_basis_dimension = through_next.dimension
    else:
        next_masks = ()
        maximum_basis_dimension = retained.dimension

    # Separate the quadratic rapidities from the general interaction products;
    # the component solver transforms these factors without a CI matrix.
    rapidities = np.asarray(
        tuple(value for block in reference.mode_references for value in block.rapidities),
        dtype=np.complex128,
    )
    determinant_tilde = tilde_map_from_references(reference.up, reference.down)
    symmetry = tilde_map(
        cluster.masks,
        tuple(int(p) for p in determinant_tilde.partners),
        tuple(complex(value) for value in determinant_tilde.factors),
    )
    complete_dimension = comb(2 * n, n) ** (1 if model.symmetry == "charge" else 2)
    products = list(_interaction_products(reference))
    if reference.reference_gamma_loss is not None:
        products.extend(_quadratic_dissipation_counterterms(model, reference))
    else:
        delta_by_spin = tuple(np.asarray(model.E) - block for block in reference.one_body_blocks)
        for i in range(model.n_sites):
            for j in range(model.n_sites):
                for spin in range(len(delta_by_spin)):
                    value = delta_by_spin[spin][i, j]
                    if value == 0.0:
                        continue
                    normal_factors = (
                        physical_mode_operator(reference, i, spin, "c_dagger"),
                        physical_mode_operator(reference, j, spin, "c"),
                    )
                    tilde_factors = (
                        physical_mode_operator(reference, i, spin, "tilde_c_dagger"),
                        physical_mode_operator(reference, j, spin, "tilde_c"),
                    )
                    products.append(
                        FactorizedModeProduct(
                            f"quadratic_counter:c_{i}_{spin}_dagger_c_{j}_{spin}",
                            complex(value),
                            normal_factors,
                        )
                    )
                    products.append(
                        FactorizedModeProduct(
                            f"quadratic_counter:-tilde_c_{i}_{spin}_dagger_tilde_c_{j}_{spin}",
                            -complex(delta_by_spin[spin][j, i]),
                            tilde_factors,
                        )
                    )
    return FactorizedCCProblem(
        level=level,
        model=model,
        reference=reference,
        cluster_basis=cluster,
        tilde=symmetry,
        rapidities=rapidities,
        products=tuple(products),
        next_rank_masks=next_masks,
        next_rank_dimension_count=next_dimension,
        retained_index={mask: i for i, mask in enumerate(cluster.masks)},
        next_rank_index={mask: i for i, mask in enumerate(next_masks)},
        complete_sector_dimension=complete_dimension,
        maximum_basis_dimension_constructed=maximum_basis_dimension,
    )


def factorized_ccsd_problem(
    model: FermionicLindbladModel,
    *,
    m_up: float = 0.5,
    build_next_rank_masks: bool = False,
    reference_E: ArrayLike | None = None,
    reference: HartreeFockReference | None = None,
    reference_mode: str = "full_hf",
) -> FactorizedCCProblem:
    """Build CCSD with full HF by default or a supplied reference.

    Rank-six masks are omitted unless build_next_rank_masks=True. Their
    dimension is recorded without allocating the shell."""

    return _factorized_cc_problem(
        model,
        "CCSD",
        m_up=m_up,
        build_next_rank_masks=build_next_rank_masks,
        reference_E=reference_E,
        reference=reference,
        reference_mode=reference_mode,
    )


def factorized_cc_problem(
    model: FermionicLindbladModel,
    level: str = "CCSD",
    *,
    m_up: float = 0.3,
    build_next_rank_masks: bool = False,
    reference_E: ArrayLike | None = None,
    reference: HartreeFockReference | None = None,
    reference_mode: str = "full_hf",
) -> FactorizedCCProblem:
    """Build controlled factorized CCS/CCSD with a selectable reference mode."""

    return _factorized_cc_problem(
        model,
        level,
        m_up=m_up,
        build_next_rank_masks=build_next_rank_masks,
        reference_E=reference_E,
        reference=reference,
        reference_mode=reference_mode,
    )


@dataclass(frozen=True, slots=True)
class FactorizedActionStats:
    """Bounded work-space instrumentation for one residual or JVP action."""

    largest_intermediate_polynomial: int
    transformed_factor_applications: int
    coefficient_updates: int
    maximum_output_creator_rank: int
    complete_sector_dimension: int
    maximum_basis_dimension_constructed: int
    complete_sector_state_vector_constructed: bool = False
    dense_jacobian_constructed: bool = False


@dataclass(frozen=True, slots=True)
class FactorizedResidual:
    """Retained and optional immediate omitted-rank residual coefficients."""

    retained: ComplexVector
    next_rank: ComplexVector | None
    reference: complex
    stats: FactorizedActionStats
    wall_seconds: float

    @property
    def retained_norm(self) -> float:
        return float(np.linalg.norm(self.retained))

    @property
    def next_rank_norm(self) -> float | None:
        return None if self.next_rank is None else float(np.linalg.norm(self.next_rank))


@dataclass(frozen=True, slots=True)
class FactorizedJVP:
    """Directional derivative of retained and optional next-rank residuals."""

    retained: ComplexVector
    next_rank: ComplexVector | None
    reference: complex
    stats: FactorizedActionStats
    wall_seconds: float


def expand_interaction_products(
    products: Sequence[FactorizedModeProduct],
) -> tuple[FactorizedModeProduct, ...]:
    """Distribute factorized products into single-component oracle monomials.

    This is intentionally an oracle path.  Production residuals retain the six
    low-rank products and never perform this Cartesian expansion.
    """

    expanded: list[FactorizedModeProduct] = []
    for product in products:
        choices = tuple(factor.components for factor in product.factors)
        for term_index, components in enumerate(cartesian_product(*choices)):
            coefficient = product.coefficient * complex(
                np.prod([value for _, _, value in components])
            )
            if coefficient == 0.0:
                continue
            factors = tuple(
                LinearModeOperator(
                    label=f"{product.label}_factor_{j}",
                    components=((kind, mode, 1.0 + 0.0j),),
                )
                for j, (kind, mode, _) in enumerate(components)
            )
            expanded.append(
                FactorizedModeProduct(
                    label=f"expanded_{product.label}_{term_index}",
                    coefficient=coefficient,
                    factors=factors,
                )
            )
    return tuple(expanded)


def _validate_amplitudes(problem: FactorizedCCProblem, values: np.ndarray) -> ComplexVector:
    amplitudes = np.asarray(values, dtype=np.complex128)
    if amplitudes.shape != (problem.num_amplitudes,) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes must be finite and match the CCSD basis")
    return amplitudes


def _add(polynomial: Polynomial, mask: int, value: complex, stats: dict[str, int]) -> None:
    if value == 0.0:
        return
    updated = polynomial.get(mask, 0.0 + 0.0j) + value
    stats["updates"] += 1
    if updated == 0.0:
        polynomial.pop(mask, None)
    else:
        polynomial[mask] = updated


def _derivatives(problem: FactorizedCCProblem, amplitudes: ComplexVector) -> tuple[Polynomial, ...]:
    r"""Return coefficients of $[P_p,T]$ in canonical creator order."""

    values: list[Polynomial] = [dict() for _ in range(problem.cluster_basis.num_modes)]
    for mask, amplitude in zip(problem.cluster_basis.masks, amplitudes, strict=True):
        occupied = mask
        while occupied:
            bit = occupied & -occupied
            p = bit.bit_length() - 1
            sign = -1.0 if (mask & (bit - 1)).bit_count() % 2 else 1.0
            values[p][mask ^ bit] = sign * complex(amplitude)
            occupied ^= bit
    return tuple(values)


def _linear_shifts(
    factors: Sequence[LinearModeOperator],
    derivatives: tuple[Polynomial, ...],
) -> dict[LinearKey, Polynomial]:
    """Consolidate ``sum_p f_P[p] [P_p,T]`` for each distinct linear form."""

    shifts: dict[LinearKey, Polynomial] = {}
    dummy_stats = {"updates": 0}
    for factor in factors:
        key = factor.components
        if key in shifts:
            continue
        polynomial: Polynomial = {}
        for kind, mode, coefficient in factor.components:
            if kind != "annihilate":
                continue
            for mask, value in derivatives[mode].items():
                _add(polynomial, mask, coefficient * value, dummy_stats)
        shifts[key] = polynomial
    return shifts


def _wedge_sign(left_mask: int, right_mask: int) -> float:
    """Sign of a canonical left creator string acting on ``right_mask``."""

    parity = 0
    occupied = left_mask
    while occupied:
        bit = occupied & -occupied
        parity ^= (right_mask & (bit - 1)).bit_count() & 1
        occupied ^= bit
    return -1.0 if parity else 1.0


def _apply_shift(
    shift: Polynomial,
    state: Polynomial,
    rank_limit: int,
    stats: dict[str, int],
) -> Polynomial:
    out: Polynomial = {}
    for left_mask, left_value in shift.items():
        for right_mask, right_value in state.items():
            if left_mask & right_mask:
                continue
            target = left_mask | right_mask
            if target.bit_count() > rank_limit:
                continue
            _add(
                out,
                target,
                left_value * right_value * _wedge_sign(left_mask, right_mask),
                stats,
            )
    return out


def _apply_transformed_linear(
    factor: LinearModeOperator,
    shift: Polynomial,
    state: Polynomial,
    rank_limit: int,
    stats: dict[str, int],
) -> Polynomial:
    r"""Apply $F+[F,T]$ to one sparse, rank-bounded polynomial."""

    out: Polynomial = {}
    for mask, amplitude in state.items():
        for kind, mode, coefficient in factor.components:
            bit = 1 << mode
            occupied = bool(mask & bit)
            if kind == "annihilate":
                if not occupied:
                    continue
                target = mask ^ bit
            else:
                if occupied:
                    continue
                target = mask | bit
            if target.bit_count() > rank_limit:
                continue
            sign = -1.0 if (mask & (bit - 1)).bit_count() % 2 else 1.0
            _add(out, target, amplitude * coefficient * sign, stats)
    shifted = _apply_shift(shift, state, rank_limit, stats)
    for mask, value in shifted.items():
        _add(out, mask, value, stats)
    stats["applications"] += 1
    stats["largest"] = max(stats["largest"], len(out))
    return out


def _all_factors(products: Sequence[FactorizedModeProduct]) -> tuple[LinearModeOperator, ...]:
    return tuple(factor for product in products for factor in product.factors)


def _product_action(
    product: FactorizedModeProduct,
    shifts: Mapping[LinearKey, Polynomial],
    maximum_output_rank: int,
    stats: dict[str, int],
) -> Polynomial:
    state: Polynomial = {0: product.coefficient}
    number = len(product.factors)
    for step, factor in enumerate(reversed(product.factors)):
        remaining = number - step - 1
        # A remaining bare annihilator can lower creator rank by at most one.
        # Anything above this bound cannot return to the requested output.
        rank_limit = maximum_output_rank + remaining
        state = _apply_transformed_linear(
            factor, shifts[factor.components], state, rank_limit, stats
        )
    return state


def _action_stats(
    problem: FactorizedCCProblem,
    maximum_output_rank: int,
    stats: Mapping[str, int],
) -> FactorizedActionStats:
    return FactorizedActionStats(
        largest_intermediate_polynomial=int(stats["largest"]),
        transformed_factor_applications=int(stats["applications"]),
        coefficient_updates=int(stats["updates"]),
        maximum_output_creator_rank=maximum_output_rank,
        complete_sector_dimension=problem.complete_sector_dimension,
        maximum_basis_dimension_constructed=problem.maximum_basis_dimension_constructed,
    )


def factorized_ccsd_residual(
    problem: FactorizedCCProblem,
    amplitudes: np.ndarray,
    *,
    include_next_rank: bool = False,
    products: Sequence[FactorizedModeProduct] | None = None,
) -> FactorizedResidual:
    r"""Return the direct factorized $e^{-T}Le^T|0\rangle$ projections.

    Passing distributed ``products`` is reserved for the small-system oracle.
    The default uses all factorized products of the general interaction.
    Explicit next-rank output requires a problem built with next-rank masks.
    """

    if include_next_rank and len(problem.next_rank_masks) != problem.next_rank_dimension:
        raise ValueError("Explicit next-rank output requires build_next_rank_masks=True")

    started = perf_counter()
    t = _validate_amplitudes(problem, amplitudes)
    interaction = problem.products if products is None else tuple(products)
    maximum_rank = (
        problem.next_rank_creator_rank
        if include_next_rank
        else problem.retained_creator_rank
    )
    derivatives = _derivatives(problem, t)
    shifts = _linear_shifts(_all_factors(interaction), derivatives)
    stats = {"largest": 1, "applications": 0, "updates": 0}

    retained = np.asarray(
        [
            sum(
                (
                    problem.rapidities[p]
                    for p in range(problem.cluster_basis.num_modes)
                    if mask & (1 << p)
                ),
                start=0.0 + 0.0j,
            )
            * value
            for mask, value in zip(problem.cluster_basis.masks, t, strict=True)
        ],
        dtype=np.complex128,
    )
    next_rank = (
        np.zeros(problem.next_rank_dimension, dtype=np.complex128)
        if include_next_rank
        else None
    )
    reference = 0.0 + 0.0j
    for product in interaction:
        result = _product_action(product, shifts, maximum_rank, stats)
        reference += result.get(0, 0.0 + 0.0j)
        for mask, value in result.items():
            i = problem.retained_index.get(mask)
            if i is not None:
                retained[i] += value
            elif next_rank is not None:
                j = problem.next_rank_index.get(mask)
                if j is not None:
                    next_rank[j] += value

    # The accepted factorized products and rapidities store paper_K.
    retained *= -1j
    reference *= -1j
    if next_rank is not None:
        next_rank *= -1j
    return FactorizedResidual(
        retained=_immutable_complex(retained),
        next_rank=None if next_rank is None else _immutable_complex(next_rank),
        reference=complex(reference),
        stats=_action_stats(problem, maximum_rank, stats),
        wall_seconds=float(perf_counter() - started),
    )


def _product_action_jvp(
    product: FactorizedModeProduct,
    shifts: Mapping[LinearKey, Polynomial],
    direction_shifts: Mapping[LinearKey, Polynomial],
    maximum_output_rank: int,
    stats: dict[str, int],
) -> Polynomial:
    state: Polynomial = {0: product.coefficient}
    tangent: Polynomial = {}
    number = len(product.factors)
    for step, factor in enumerate(reversed(product.factors)):
        remaining = number - step - 1
        rank_limit = maximum_output_rank + remaining
        tangent_from_state = _apply_shift(
            direction_shifts[factor.components], state, rank_limit, stats
        )
        tangent = _apply_transformed_linear(
            factor, shifts[factor.components], tangent, rank_limit, stats
        )
        for mask, value in tangent_from_state.items():
            _add(tangent, mask, value, stats)
        state = _apply_transformed_linear(
            factor, shifts[factor.components], state, rank_limit, stats
        )
        stats["largest"] = max(stats["largest"], len(tangent), len(state))
    return tangent


def factorized_ccsd_jvp(
    problem: FactorizedCCProblem,
    amplitudes: np.ndarray,
    direction: np.ndarray,
    *,
    include_next_rank: bool = False,
    products: Sequence[FactorizedModeProduct] | None = None,
) -> FactorizedJVP:
    """Analytic matrix-free directional derivative of the factorized residual."""

    if include_next_rank and len(problem.next_rank_masks) != problem.next_rank_dimension:
        raise ValueError("Explicit next-rank output requires build_next_rank_masks=True")

    started = perf_counter()
    t = _validate_amplitudes(problem, amplitudes)
    dt = _validate_amplitudes(problem, direction)
    interaction = problem.products if products is None else tuple(products)
    maximum_rank = (
        problem.next_rank_creator_rank
        if include_next_rank
        else problem.retained_creator_rank
    )
    factors = _all_factors(interaction)
    shifts = _linear_shifts(factors, _derivatives(problem, t))
    direction_shifts = _linear_shifts(factors, _derivatives(problem, dt))
    stats = {"largest": 1, "applications": 0, "updates": 0}

    retained = np.asarray(
        [
            sum(
                (
                    problem.rapidities[p]
                    for p in range(problem.cluster_basis.num_modes)
                    if mask & (1 << p)
                ),
                start=0.0 + 0.0j,
            )
            * value
            for mask, value in zip(problem.cluster_basis.masks, dt, strict=True)
        ],
        dtype=np.complex128,
    )
    next_rank = (
        np.zeros(problem.next_rank_dimension, dtype=np.complex128)
        if include_next_rank
        else None
    )
    reference = 0.0 + 0.0j
    for product in interaction:
        result = _product_action_jvp(
            product,
            shifts,
            direction_shifts,
            maximum_rank,
            stats,
        )
        reference += result.get(0, 0.0 + 0.0j)
        for mask, value in result.items():
            i = problem.retained_index.get(mask)
            if i is not None:
                retained[i] += value
            elif next_rank is not None:
                j = problem.next_rank_index.get(mask)
                if j is not None:
                    next_rank[j] += value

    retained *= -1j
    reference *= -1j
    if next_rank is not None:
        next_rank *= -1j
    return FactorizedJVP(
        retained=_immutable_complex(retained),
        next_rank=None if next_rank is None else _immutable_complex(next_rank),
        reference=complex(reference),
        stats=_action_stats(problem, maximum_rank, stats),
        wall_seconds=float(perf_counter() - started),
    )


def factorized_residual(
    problem: FactorizedCCProblem,
    amplitudes: np.ndarray,
    *,
    max_rank: int = 4,
    products: Sequence[FactorizedModeProduct] | None = None,
) -> FactorizedResidual:
    """Compact integration API for retained or retained-plus-next residuals."""

    allowed = (problem.retained_creator_rank, problem.next_rank_creator_rank)
    if max_rank not in allowed:
        raise ValueError(f"factorized {problem.level} residual max_rank must be one of {allowed}")
    return factorized_ccsd_residual(
        problem,
        amplitudes,
        include_next_rank=max_rank == problem.next_rank_creator_rank,
        products=products,
    )


def factorized_jvp(
    problem: FactorizedCCProblem,
    amplitudes: np.ndarray,
    direction: np.ndarray,
    *,
    max_rank: int = 4,
    products: Sequence[FactorizedModeProduct] | None = None,
) -> FactorizedJVP:
    """Compact integration API for the analytic matrix-free JVP."""

    allowed = (problem.retained_creator_rank, problem.next_rank_creator_rank)
    if max_rank not in allowed:
        raise ValueError(f"factorized {problem.level} JVP max_rank must be one of {allowed}")
    return factorized_ccsd_jvp(
        problem,
        amplitudes,
        direction,
        include_next_rank=max_rank == problem.next_rank_creator_rank,
        products=products,
    )


def factorized_real_residual(
    problem: FactorizedCCProblem,
    variables: np.ndarray,
) -> NDArray[np.float64]:
    """Retained residual in independent real tilde coordinates."""

    x = np.asarray(variables, dtype=np.float64)
    amplitudes = problem.tilde.unpack(x)
    values = factorized_ccsd_residual(problem, amplitudes).retained
    return problem.tilde.pack(values, check=False)


def factorized_real_jvp(
    problem: FactorizedCCProblem,
    variables: np.ndarray,
    direction: np.ndarray,
) -> NDArray[np.float64]:
    """Matrix-free analytic JVP in independent real tilde coordinates."""

    x = np.asarray(variables, dtype=np.float64)
    dx = np.asarray(direction, dtype=np.float64)
    if x.shape != dx.shape:
        raise ValueError("JVP point and direction must have the same shape")
    amplitudes = problem.tilde.unpack(x)
    tangent = problem.tilde.unpack(dx)
    values = factorized_ccsd_jvp(problem, amplitudes, tangent).retained
    return problem.tilde.pack(values, check=False)


__all__ = [
    "FactorizedActionStats",
    "FactorizedCCProblem",
    "FactorizedJVP",
    "FactorizedResidual",
    "expand_interaction_products",
    "factorized_cc_problem",
    "factorized_ccsd_jvp",
    "factorized_ccsd_problem",
    "factorized_ccsd_residual",
    "factorized_jvp",
    "factorized_real_jvp",
    "factorized_real_residual",
    "factorized_residual",
]

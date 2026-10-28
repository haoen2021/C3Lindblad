"""Factorized physical operators in particle-hole coordinates.

Linear P/bar_P forms and sparse products supply the interaction terms for CC
and the trace contractions for observables. No CI solver is provided here.
When a product acts on a coefficient vector, projection occurs after all its
factors have acted, preserving intermediate states outside the retained basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


from .basis import CIBasis, apply_mode_monomial
from .reference import HartreeFockReference


ComplexVector = NDArray[np.complex128]


def _immutable_complex(value: ArrayLike) -> ComplexVector:
    contiguous = np.ascontiguousarray(value, dtype=np.complex128)
    frozen = np.frombuffer(contiguous.tobytes(), dtype=np.complex128).reshape(
        contiguous.shape
    )
    frozen.flags.writeable = False
    return frozen


def _finite_complex(value: complex, name: str) -> complex:
    scalar = complex(value)
    if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
        raise ValueError(f"{name} must be finite")
    return scalar


CIReference = HartreeFockReference


@dataclass(frozen=True, slots=True)
class LinearModeOperator:
    """One physical or tilde orbital operator as a linear particle-hole form."""

    label: str
    components: tuple[tuple[str, int, complex], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("linear-mode label must be nonempty")
        if not self.components:
            raise ValueError("linear-mode operator must have components")
        normalized: list[tuple[str, int, complex]] = []
        seen: set[tuple[str, int]] = set()
        for kind, mode, coefficient in self.components:
            if kind not in ("create", "annihilate"):
                raise ValueError("linear-mode kind must be create or annihilate")
            if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0:
                raise ValueError("linear-mode index must be nonnegative")
            coefficient = _finite_complex(coefficient, "linear-mode coefficient")
            if coefficient == 0.0:
                continue
            key = (kind, mode)
            if key in seen:
                raise ValueError("linear-mode components must be consolidated")
            seen.add(key)
            normalized.append((kind, mode, coefficient))
        if not normalized:
            raise ValueError("linear-mode operator is identically zero")
        object.__setattr__(self, "components", tuple(normalized))


@dataclass(frozen=True, slots=True)
class FactorizedModeProduct:
    """Coefficient times a left-to-right product of linear mode operators."""

    label: str
    coefficient: complex
    factors: tuple[LinearModeOperator, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("factorized-product label must be nonempty")
        coefficient = _finite_complex(self.coefficient, "product coefficient")
        if coefficient == 0.0:
            raise ValueError("factorized-product coefficient must be nonzero")
        if not self.factors:
            raise ValueError("factorized product must contain at least one factor")
        object.__setattr__(self, "coefficient", coefficient)


def physical_mode_operator(
    reference: CIReference,
    site: int,
    spin: int,
    channel: str,
) -> LinearModeOperator:
    """Transform one site/spin physical or tilde operator to PH modes."""

    if not isinstance(reference, HartreeFockReference):
        raise TypeError("reference must be a validated CI reference")
    if (
        isinstance(site, bool)
        or not isinstance(site, int)
        or not 0 <= site < reference.n_sites
    ):
        raise ValueError("site is outside the reference")
    if spin not in (0, 1):
        raise ValueError("spin must be 0 or 1")
    if channel not in ("c", "c_dagger", "tilde_c", "tilde_c_dagger"):
        raise ValueError("unknown physical/tilde operator channel")
    spin_reference = reference.up if spin == 0 else reference.down
    if spin_reference is None:
        raise ValueError("charge-only references have one full orbital block; use spin=0")
    n = reference.n_sites
    offset = spin * 2 * n
    if channel in ("c", "tilde_c_dagger"):
        sharp = False
        row = site if channel == "c" else n + site
        coefficients = spin_reference.V[row, :]
    else:
        sharp = True
        column = site if channel == "c_dagger" else n + site
        coefficients = spin_reference.V_inv[:, column]
    components: list[tuple[str, int, complex]] = []
    for p, coefficient in enumerate(coefficients):
        occupied = bool(spin_reference.vacuum_mask[p])
        if sharp:
            # sharp = bar_xi V^{-1}; occupied bar_xi is P.
            kind = "annihilate" if occupied else "create"
        else:
            # dual = V xi; occupied xi is bar_P.
            kind = "create" if occupied else "annihilate"
        components.append((kind, offset + p, complex(coefficient)))
    prefix = "up" if spin == 0 else "down"
    return LinearModeOperator(f"{prefix}_site_{site}_{channel}", tuple(components))


def _interaction_products(
    reference: CIReference,
) -> tuple[FactorizedModeProduct, ...]:
    operators: dict[tuple[int, str], LinearModeOperator] = {}

    def operator(orbital: int, channel: str) -> LinearModeOperator:
        key = (orbital, channel)
        if key not in operators:
            operators[key] = physical_mode_operator(
                reference,
                *reference.model.orbital_site_spin(orbital),
                channel,
            )
        return operators[key]

    products: list[FactorizedModeProduct] = []
    for index, (p, q, r, s, value) in enumerate(reference.model.interaction.terms):
        products.append(
            FactorizedModeProduct(
                f"two_body_{index}:physical",
                value,
                (
                    operator(p, "c_dagger"),
                    operator(q, "c_dagger"),
                    operator(s, "c"),
                    operator(r, "c"),
                ),
            )
        )
        products.append(
            FactorizedModeProduct(
                f"two_body_{index}:tilde",
                -value.conjugate(),
                (
                    operator(p, "tilde_c_dagger"),
                    operator(q, "tilde_c_dagger"),
                    operator(s, "tilde_c"),
                    operator(r, "tilde_c"),
                ),
            )
        )
    return tuple(products)


def _apply_linear_form(
    states: Mapping[int, complex],
    operator: LinearModeOperator,
    num_modes: int,
) -> dict[int, complex]:
    result: dict[int, complex] = {}
    for mask, amplitude in states.items():
        for kind, mode, coefficient in operator.components:
            target, sign = apply_mode_monomial(mask, ((kind, mode),), num_modes)
            if target is not None:
                value = amplitude * coefficient * sign
                result[target] = result.get(target, 0.0 + 0.0j) + value
    return {mask: value for mask, value in result.items() if value != 0.0}


def apply_linear_mode_operator(
    states: Mapping[int, complex],
    operator: LinearModeOperator,
    num_modes: int,
) -> dict[int, complex]:
    """Public shared CAR action for one linear PH-mode form."""

    if not isinstance(states, Mapping):
        raise TypeError("states must be a mask-to-amplitude mapping")
    if isinstance(num_modes, bool) or not isinstance(num_modes, int) or num_modes <= 0:
        raise ValueError("num_modes must be a positive integer")
    validated: dict[int, complex] = {}
    for mask, amplitude in states.items():
        if (
            isinstance(mask, bool)
            or not isinstance(mask, int)
            or not 0 <= mask < (1 << num_modes)
        ):
            raise ValueError("state mask is outside the mode space")
        value = _finite_complex(amplitude, "state amplitude")
        if value != 0.0:
            validated[mask] = value
    return _apply_linear_form(validated, operator, num_modes)


def _apply_complete_product(
    mask: int,
    product: FactorizedModeProduct,
    num_modes: int,
) -> dict[int, complex]:
    states: dict[int, complex] = {mask: product.coefficient}
    for factor in reversed(product.factors):
        states = _apply_linear_form(states, factor, num_modes)
        if not states:
            break
    return states


def _adjoint_linear(operator: LinearModeOperator) -> LinearModeOperator:
    return LinearModeOperator(
        label=f"adjoint({operator.label})",
        components=tuple(
            (
                "annihilate" if kind == "create" else "create",
                mode,
                coefficient.conjugate(),
            )
            for kind, mode, coefficient in operator.components
        ),
    )


def _adjoint_product(product: FactorizedModeProduct) -> FactorizedModeProduct:
    return FactorizedModeProduct(
        label=f"adjoint({product.label})",
        coefficient=product.coefficient.conjugate(),
        factors=tuple(_adjoint_linear(factor) for factor in reversed(product.factors)),
    )


def _apply_product_to_superposition(
    basis: CIBasis,
    vector: ComplexVector,
    product: FactorizedModeProduct,
) -> ComplexVector:
    """Apply one complete product globally and project only at its end."""

    states = {
        mask: product.coefficient * complex(amplitude)
        for mask, amplitude in zip(basis.masks, vector, strict=True)
        if amplitude != 0.0
    }
    for factor in reversed(product.factors):
        states = _apply_linear_form(states, factor, basis.layout.num_modes)
        if not states:
            break
    projected = np.zeros(basis.dimension, dtype=np.complex128)
    for mask, amplitude in states.items():
        row = basis.index_of.get(mask)
        if row is not None:
            projected[row] += amplitude
    return projected


def apply_factorized_mode_product(
    basis: CIBasis,
    coefficients: ArrayLike,
    product: FactorizedModeProduct,
) -> ComplexVector:
    """Apply a complete product and project only its final superposition."""

    vector = np.asarray(coefficients, dtype=np.complex128)
    if vector.shape != (basis.dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError("coefficients must match the supplied CI basis")
    return _apply_product_to_superposition(basis, vector, product)


def factorized_mode_product_expectation(
    basis: CIBasis,
    coefficients: ArrayLike,
    product: FactorizedModeProduct,
    *,
    target_mask: int = 0,
) -> complex:
    """Return one final-mask coefficient without an intermediate projection."""

    vector = np.asarray(coefficients, dtype=np.complex128)
    if vector.shape != (basis.dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError("coefficients must match the supplied CI basis")
    if target_mask not in basis.index_of:
        raise ValueError("target_mask is not retained in the supplied CI basis")
    # Evaluate the one requested row as <O^dagger target|coefficients>.  This
    # starts from one mask and visits only masks reachable by the adjoint
    # product (rank 0/1 for a bilinear, rank 0/1/2 for a quartic), instead of
    # applying O to every retained source determinant.
    adjoint_states = _apply_complete_product(
        target_mask, _adjoint_product(product), basis.layout.num_modes
    )
    value = 0.0 + 0.0j
    for mask, amplitude in adjoint_states.items():
        source = basis.index_of.get(mask)
        if source is not None:
            value += amplitude.conjugate() * vector[source]
    return complex(value)

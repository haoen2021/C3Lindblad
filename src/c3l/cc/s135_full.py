"""Production NumPy/CPU response action on the complete S1 + S3 + S5 space.

The CCSD state is fixed; every rank-five response coordinate is retained.
Contractions are evaluated in tiles without assembling the response matrix.
The JAX implementation reuses the contraction plans defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from time import perf_counter
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import component_tensor as _ct
from .component_tensor import (
    BlockCoordinates,
    ComponentCoordinates,
    ComponentEOMContext,
)


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class _RankFiveGroup:
    input_degree: int
    contractions: tuple[Mapping[int, ComplexArray], ...]
    creation: object
    creation_degree: int
    acted_degree: int
    output_degree: int


def _contraction_signature(contractions) -> tuple[tuple[tuple[int, bytes], ...], ...]:
    """Return the exact ordered contraction key used for prefix reuse."""

    return tuple(
        tuple(
            (group, np.ascontiguousarray(vector).tobytes())
            for group, vector in sorted(annihilator.items())
        )
        for annihilator in contractions
    )


def _input_five_plans(context: ComponentEOMContext):
    """Yield exact BCH response plans with a rank-five input placeholder."""

    placeholder = _ct._Placeholder(5)

    def resolved(state):
        positions = [
            position
            for position, form in enumerate(state.factors)
            if isinstance(form, _ct._Placeholder)
        ]
        if len(positions) != 1:
            raise RuntimeError("a rank-five response plan lost its input placeholder")
        position = positions[0]
        current = state.factors[position]
        degree_after = sum(
            form.degrees[0] for form in state.factors[position + 1 :]
        )
        coefficient = state.coefficient * ((-1) ** (current.degree * degree_after))
        creation = tuple(
            form
            for form in state.factors
            if not isinstance(form, _ct._Placeholder)
        )
        output_degree = sum(form.degrees[0] for form in creation) + current.degree
        if output_degree in (1, 3, 5):
            yield coefficient, creation, current.contractions, output_degree

    for record in context.problem.static.oracle.products:
        states = [_ct._Term(record.coefficient, (placeholder,))]
        for factor in reversed(record.factors):
            data = context.problem.static.linear[factor.components]
            shifts = _ct._split_homogeneous(
                _ct.interior(data.annihilator, context.amplitudes)
            )
            states = _ct._apply_factor_plan(states, data, shifts)
        for state in states:
            yield from resolved(state)
    for key, value in context.vacuum_interaction.blocks.items():
        yield from resolved(
            _ct._Term(
                -1.0 + 0.0j,
                (
                    placeholder,
                    _ct.ExteriorForm(context.amplitudes.n, {key: value}),
                ),
            )
        )


class CompiledFullS135Action:
    r"""Compile the exact action blocks among ``S1``, ``S3``, and ``S5``.

    Algebraically identical contraction prefixes are evaluated once.  The
    compiler changes only the contraction graph: it preserves coefficients,
    fermionic order, BCH truncation, and response coordinates.
    """

    def __init__(
        self,
        context: ComponentEOMContext,
        *,
        materialization_batch_size: int = 16,
    ):
        if materialization_batch_size <= 0 or materialization_batch_size > 16:
            raise ValueError("materialization_batch_size must be in [1,16]")
        started = perf_counter()
        grouped: dict[
            tuple[int, tuple[tuple[tuple[int, bytes], ...], ...], int, int],
            list[object],
        ] = {}
        contractions: dict[
            tuple[int, tuple[tuple[tuple[int, bytes], ...], ...], int, int],
            tuple[Mapping[int, ComplexArray], ...],
        ] = {}
        retained = (
            (
                plan.input_degree,
                plan.coefficient,
                plan.creation_factors,
                plan.contractions,
                plan.input_degree
                - len(plan.contractions)
                + sum(form.degrees[0] for form in plan.creation_factors),
            )
            for plan in context.response_plan
        )
        rank_five = (
            (5, coefficient, creation, contraction, output_degree)
            for coefficient, creation, contraction, output_degree in _input_five_plans(
                context
            )
        )
        for input_degree, coefficient, creation, contraction, output_degree in chain(
            retained, rank_five
        ):
            creation_degree = sum(form.degrees[0] for form in creation)
            if output_degree not in (1, 3, 5) or input_degree not in (1, 3, 5):
                continue
            key = (
                input_degree,
                _contraction_signature(contraction),
                creation_degree,
                output_degree,
            )
            grouped.setdefault(key, []).append(_ct._Term(coefficient, creation))
            contractions[key] = contraction

        groups: list[_RankFiveGroup] = []
        for key, terms in grouped.items():
            creation = _ct._materialize_terms_raw(
                terms,
                context.problem.coordinates.n,
                {key[2]},
                batch_size=materialization_batch_size,
            )
            acted_degree = key[0] - len(contractions[key])
            if key[2] + acted_degree != key[3]:
                raise RuntimeError("a compiled full-S135 group has inconsistent degrees")
            groups.append(
                _RankFiveGroup(
                    input_degree=key[0],
                    contractions=contractions[key],
                    creation=creation,
                    creation_degree=key[2],
                    acted_degree=acted_degree,
                    output_degree=key[3],
                )
            )
        self.context = context
        self.groups = tuple(groups)
        self.compile_wall_seconds = float(perf_counter() - started)
        self.storage_bytes = int(
            sum(
                value.nbytes
                for group in self.groups
                for value in group.creation.blocks.values()
            )
        )
        self.group_count = len(self.groups)
        self.creation_degree_counts = {
            degree: sum(group.creation_degree == degree for group in self.groups)
            for degree in range(6)
            if any(group.creation_degree == degree for group in self.groups)
        }
        self.input_degree_counts = {
            degree: sum(group.input_degree == degree for group in self.groups)
            for degree in (1, 3, 5)
            if any(group.input_degree == degree for group in self.groups)
        }
        self.input_five_prefixes = len(
            {
                _contraction_signature(group.contractions)
                for group in self.groups
                if group.input_degree == 5
            }
        )

    def acted(self, vector: ArrayLike, *, coordinates: ComponentCoordinates):
        """Apply each unique contraction prefix to one response vector."""

        if coordinates.n != self.context.problem.coordinates.n:
            raise ValueError("full-S135 coordinates use another mode dimension")
        response = coordinates.unpack(vector)
        degrees = set(response.degrees)
        cache: dict[tuple[int, tuple], object] = {}
        empty = _ct.ExteriorForm(coordinates.n, {})
        out = []
        for group in self.groups:
            if group.input_degree not in degrees:
                out.append(empty)
                continue
            key = (
                group.input_degree,
                _contraction_signature(group.contractions),
            )
            value = cache.get(key)
            if value is None:
                value = response.homogeneous(group.input_degree)
                for annihilator in group.contractions:
                    value = _ct.interior(annihilator, value)
                cache[key] = value
            out.append(value)
        return tuple(out)

    def values_from_acted(
        self,
        acted_groups,
        coordinates: ComponentCoordinates,
    ) -> ComplexArray:
        """Project prepared prefix contractions onto explicit coordinates."""

        if len(acted_groups) != len(self.groups):
            raise ValueError("acted groups do not match the compiled full-S135 action")
        values = np.zeros(coordinates.dimension, dtype=np.complex128)
        for target in coordinates.blocks:
            local = np.zeros(target.vector_positions.size, dtype=np.complex128)
            for group, acted in zip(self.groups, acted_groups, strict=True):
                if not acted.blocks:
                    continue
                for creation_key, creation in group.creation.blocks.items():
                    for acted_key, response in acted.blocks.items():
                        if _ct._add_key(creation_key, acted_key) != target.key:
                            continue
                        if sum(acted_key) == 0:
                            index = tuple(
                                target.indices[:, axis]
                                for axis in range(target.indices.shape[1])
                            )
                            local += complex(response) * creation[index]
                        else:
                            local += _ct._wedge_block_values(
                                creation,
                                creation_key,
                                response,
                                acted_key,
                                target.indices,
                            )
            values[target.vector_positions] = local
        return values


def rapidity_diagonal(
    rapidities: ArrayLike,
    coordinates: ComponentCoordinates,
) -> ComplexArray:
    """Return the quadratic response diagonal on exterior coordinates."""

    values = np.asarray(rapidities, dtype=np.complex128)
    if values.shape not in ((2 * coordinates.n,), (4 * coordinates.n,)) or not np.all(np.isfinite(values)):
        raise ValueError("rapidities must contain two or four finite normal-mode groups")
    if values.size == 2 * coordinates.n and any(block.key[2] or block.key[3] for block in coordinates.blocks):
        raise ValueError("two-group rapidities cannot address spin-resolved coordinates")
    diagonal = np.zeros(coordinates.dimension, dtype=np.complex128)
    for block in coordinates.blocks:
        term = np.zeros(block.vector_positions.size, dtype=np.complex128)
        axis = 0
        for group, count in enumerate(block.key):
            for _ in range(count):
                term += values[group * coordinates.n + block.indices[:, axis]]
                axis += 1
        diagonal[block.vector_positions] = term
    return diagonal


class FullS135ResponseOperator:
    """Exact tiled right action on the complete ``S1 + S3 + S5`` space."""

    def __init__(
        self,
        action: CompiledFullS135Action,
        *,
        s5_tile_size: int = 16384,
    ):
        if s5_tile_size <= 0:
            raise ValueError("s5_tile_size must be positive")
        self.action = action
        self.P = action.context.problem.coordinates
        self.shell = action.context.problem.s5_coordinates
        self.S5 = self.shell.subset(np.arange(self.shell.dimension, dtype=np.int64))
        self.p_dimension = self.P.dimension
        self.s5_dimension = self.S5.dimension
        self.dimension = self.p_dimension + self.s5_dimension
        self.coordinates = ComponentCoordinates(
            self.P.n,
            self.dimension,
            self.P.blocks
            + tuple(
                BlockCoordinates(
                    block.key,
                    block.vector_positions + self.p_dimension,
                    block.indices,
                )
                for block in self.S5.blocks
            ),
        )
        self.s5_tile_size = int(s5_tile_size)

    @staticmethod
    def _tile_coordinates(n: int, tile: BlockCoordinates) -> ComponentCoordinates:
        return ComponentCoordinates(
            n,
            tile.vector_positions.size,
            (
                BlockCoordinates(
                    tile.key,
                    np.arange(tile.vector_positions.size, dtype=np.int64),
                    tile.indices,
                ),
            ),
        )

    def matvec(self, vector: ArrayLike) -> ComplexArray:
        """Apply the exact full response Jacobian without forming a matrix."""

        value = np.asarray(vector, dtype=np.complex128)
        if value.shape != (self.dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("the full-S135 input has the wrong shape or is nonfinite")
        acted = self.action.acted(value, coordinates=self.coordinates)
        out = np.zeros(self.dimension, dtype=np.complex128)
        out[: self.p_dimension] = self.action.values_from_acted(acted, self.P)
        for tile in self.shell.tiles(self.s5_tile_size):
            target = self._tile_coordinates(self.shell.n, tile)
            out[self.p_dimension + tile.vector_positions] = (
                self.action.values_from_acted(acted, target)
            )
        out += rapidity_diagonal(
            self.action.context.problem.static.oracle.rapidities,
            self.coordinates,
        ) * value
        return out


__all__ = [
    "CompiledFullS135Action",
    "FullS135ResponseOperator",
    "rapidity_diagonal",
]

"""Component-tensor backend for spinful Lindblad CCSD and EOM-CCSD.

The four tensor index groups are (up+, up-, down+, down-) in
normal-master-mode order. A block key (a, i, b, j) specifies the
number of indices in each group. Arrays are antisymmetric within
each group and store canonical exterior coefficients.

The backend evaluates factorized operator products using wedge
and interior products, supporting multiorbital interactions
represented by the underlying factorized CC problem.

High-degree products remain factorized until projection, avoiding
explicit materialization of static rank-six and response rank-five
tensors. Matrix-free residual and Jacobian-vector actions support
Newton-Krylov CCSD and EOM response calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations, product
from math import comb, prod
from time import perf_counter
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from c3l.ci.matrix_free import FactorizedModeProduct, LinearModeOperator

from .eom import EOMCCSDProblem
from .factorized import FactorizedCCProblem


ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]
BlockKey = tuple[int, int, int, int]
LinearKey = tuple[tuple[str, int, complex], ...]


def _zero_key() -> BlockKey:
    return (0, 0, 0, 0)


def _add_key(left: BlockKey, right: BlockKey) -> BlockKey:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _degree(key: BlockKey) -> int:
    return sum(key)


def _inversion_sign(sequence: Sequence[int]) -> int:
    inversions = sum(
        sequence[i] > sequence[j]
        for i in range(len(sequence))
        for j in range(i + 1, len(sequence))
    )
    return -1 if inversions % 2 else 1


def _block_shuffles(left: BlockKey, right: BlockKey) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], int]]:
    """Yield axis assignments and signs for one block wedge product."""

    out = _add_key(left, right)
    choices: list[tuple[tuple[int, ...], ...]] = []
    offset = 0
    for nl, nr in zip(left, right, strict=True):
        axes = tuple(range(offset, offset + nl + nr))
        choices.append(tuple(combinations(axes, nl)))
        offset += nl + nr
    all_axes = tuple(range(sum(out)))
    for selected_by_group in product(*choices):
        selected = set(axis for group in selected_by_group for axis in group)
        left_axes = tuple(axis for group in selected_by_group for axis in group)
        right_axes = tuple(axis for axis in all_axes if axis not in selected)
        sign = _inversion_sign(left_axes + right_axes)
        yield left_axes, right_axes, sign


def _wedge_blocks(left: ComplexArray, left_key: BlockKey, right: ComplexArray, right_key: BlockKey) -> ComplexArray:
    """Return one padded antisymmetric block of ``left wedge right``."""

    degree = _degree(left_key) + _degree(right_key)
    if degree == 0:
        return np.asarray(left * right, dtype=np.complex128)
    out = np.zeros((left.shape[0] if left.ndim else right.shape[0],) * degree, dtype=np.complex128)
    labels = list(range(degree))
    for left_axes, right_axes, sign in _block_shuffles(left_key, right_key):
        value = np.einsum(
            left,
            list(left_axes),
            right,
            list(right_axes),
            labels,
            optimize=False,
        )
        if sign == 1:
            out += value
        else:
            out -= value
    return out


def _wedge_blocks_batched(
    left: ComplexArray,
    left_key: BlockKey,
    right: ComplexArray,
    right_key: BlockKey,
    *,
    sum_batch: bool,
) -> ComplexArray:
    """Wedge equally sized structural batches, optionally summing the batch."""

    if left.shape[0] != right.shape[0]:
        raise ValueError("batched wedge factors use different batch sizes")
    degree = _degree(left_key) + _degree(right_key)
    n = left.shape[1] if left.ndim > 1 else right.shape[1]
    shape = (n,) * degree
    if not sum_batch:
        shape = (left.shape[0],) + shape
    out = np.zeros(shape, dtype=np.complex128)
    spatial = list(range(1, degree + 1))
    for left_axes, right_axes, sign in _block_shuffles(
        left_key, right_key
    ):
        value = np.einsum(
            left,
            [0, *(axis + 1 for axis in left_axes)],
            right,
            [0, *(axis + 1 for axis in right_axes)],
            spatial if sum_batch else [0, *spatial],
            optimize=False,
        )
        if sign == 1:
            out += value
        else:
            out -= value
    return out


@dataclass(frozen=True, slots=True)
class ExteriorForm:
    """A homogeneous or inhomogeneous exterior form in padded blocks."""

    n: int
    blocks: Mapping[BlockKey, ComplexArray]

    def __post_init__(self) -> None:
        clean: dict[BlockKey, ComplexArray] = {}
        for raw_key, raw in self.blocks.items():
            key = tuple(int(value) for value in raw_key)
            if len(key) != 4 or any(value < 0 for value in key):
                raise ValueError("an exterior block key must contain four nonnegative counts")
            array = np.asarray(raw, dtype=np.complex128)
            expected = (self.n,) * sum(key)
            if array.shape != expected or not np.all(np.isfinite(array)):
                raise ValueError("an exterior block has the wrong padded shape or is nonfinite")
            if np.any(array):
                clean[key] = array
        object.__setattr__(self, "blocks", MappingProxyType(clean))

    @property
    def degrees(self) -> tuple[int, ...]:
        return tuple(sorted({_degree(key) for key in self.blocks}))

    def homogeneous(self, degree: int) -> "ExteriorForm":
        return ExteriorForm(
            self.n,
            {key: value for key, value in self.blocks.items() if _degree(key) == degree},
        )

    def scaled(self, coefficient: complex) -> "ExteriorForm":
        return ExteriorForm(
            self.n,
            {key: coefficient * value for key, value in self.blocks.items()},
        )

    def norm(self) -> float:
        return float(np.sqrt(sum(np.vdot(value, value).real for value in self.blocks.values())))


def _accumulate_form(
    blocks: dict[BlockKey, ComplexArray],
    form: ExteriorForm,
    n: int,
) -> None:
    if form.n != n:
        raise ValueError("exterior forms use different one-group dimensions")
    for key, value in form.blocks.items():
        if key in blocks:
            blocks[key] += value
        else:
            blocks[key] = np.asarray(value, dtype=np.complex128).copy()


def _sum_forms(forms: Iterable[ExteriorForm], n: int) -> ExteriorForm:
    blocks: dict[BlockKey, ComplexArray] = {}
    for form in forms:
        _accumulate_form(blocks, form, n)
    return ExteriorForm(n, blocks)


def wedge(left: ExteriorForm, right: ExteriorForm, *, maximum_degree: int | None = None) -> ExteriorForm:
    """Antisymmetrized component product without determinant enumeration."""

    if left.n != right.n:
        raise ValueError("wedge factors use different one-group dimensions")
    blocks: dict[BlockKey, ComplexArray] = {}
    for left_key, left_value in left.blocks.items():
        for right_key, right_value in right.blocks.items():
            key = _add_key(left_key, right_key)
            if any(value > left.n for value in key):
                continue
            if maximum_degree is not None and _degree(key) > maximum_degree:
                continue
            value = _wedge_blocks(left_value, left_key, right_value, right_key)
            if key in blocks:
                blocks[key] += value
            else:
                blocks[key] = value
    return ExteriorForm(left.n, blocks)


def interior(vectors: Mapping[int, ComplexArray], form: ExteriorForm) -> ExteriorForm:
    r"""Apply $\sum_p v_p\iota_p$ to a component exterior form."""

    blocks: dict[BlockKey, ComplexArray] = {}
    for key, value in form.blocks.items():
        for group, vector in vectors.items():
            if key[group] == 0:
                continue
            v = np.asarray(vector, dtype=np.complex128)
            if v.shape != (form.n,):
                raise ValueError("an interior-product vector has the wrong shape")
            axis = sum(key[:group])
            target = list(key)
            target[group] -= 1
            target_key = tuple(target)  # type: ignore[assignment]
            contracted = ((-1) ** axis) * np.tensordot(v, value, axes=(0, axis))
            if target_key in blocks:
                blocks[target_key] += contracted
            else:
                blocks[target_key] = np.asarray(contracted, dtype=np.complex128)
    return ExteriorForm(form.n, blocks)


@dataclass(frozen=True, slots=True)
class BlockCoordinates:
    """Canonical coefficient positions for one padded exterior block."""

    key: BlockKey
    vector_positions: IntArray
    indices: IntArray

    def __post_init__(self) -> None:
        positions = np.asarray(self.vector_positions, dtype=np.int64)
        indices = np.asarray(self.indices, dtype=np.int64)
        if positions.ndim != 1 or indices.shape != (positions.size, _degree(self.key)):
            raise ValueError("component coordinate arrays have inconsistent shapes")
        object.__setattr__(self, "vector_positions", positions)
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True, slots=True)
class ComponentCoordinates:
    """One deterministic vector-to-padded-tensor coordinate map."""

    n: int
    dimension: int
    blocks: tuple[BlockCoordinates, ...]

    @classmethod
    def from_masks(
        cls,
        n: int,
        masks: Sequence[int],
        positions: ArrayLike | None = None,
    ) -> "ComponentCoordinates":
        selected = np.arange(len(masks), dtype=np.int64) if positions is None else np.asarray(positions, dtype=np.int64)
        records: dict[BlockKey, list[tuple[int, tuple[int, ...]]]] = {}
        for local_position, source_position in enumerate(selected):
            mask = int(masks[int(source_position)])
            grouped: list[list[int]] = [[], [], [], []]
            for mode in range(4 * n):
                if mask & (1 << mode):
                    grouped[mode // n].append(mode % n)
            key = tuple(len(group) for group in grouped)  # type: ignore[assignment]
            indices = tuple(index for group in grouped for index in group)
            records.setdefault(key, []).append((local_position, indices))
        blocks = tuple(
            BlockCoordinates(
                key,
                np.asarray([record[0] for record in values], dtype=np.int64),
                np.asarray([record[1] for record in values], dtype=np.int64),
            )
            for key, values in sorted(records.items())
        )
        return cls(n=n, dimension=len(selected), blocks=blocks)

    def unpack(self, vector: ArrayLike) -> ExteriorForm:
        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.dimension,) or not np.all(np.isfinite(values)):
            raise ValueError("component vector has the wrong shape or is nonfinite")
        blocks: dict[BlockKey, ComplexArray] = {}
        for block in self.blocks:
            array = np.zeros((self.n,) * _degree(block.key), dtype=np.complex128)
            canonical = values[block.vector_positions]
            group_permutations = [tuple(permutations(range(count))) for count in block.key]
            offsets = np.cumsum((0,) + block.key)
            for permutations_by_group in product(*group_permutations):
                columns: list[int] = []
                sign = 1
                for group, permutation in enumerate(permutations_by_group):
                    start = int(offsets[group])
                    columns.extend(start + position for position in permutation)
                    sign *= _inversion_sign(permutation)
                if columns:
                    target = tuple(block.indices[:, column] for column in columns)
                    array[target] = sign * canonical
                else:
                    array[...] = canonical[0]
            blocks[block.key] = array
        return ExteriorForm(self.n, blocks)

    def pack(self, form: ExteriorForm) -> ComplexArray:
        if form.n != self.n:
            raise ValueError("component form and coordinates use different dimensions")
        out = np.zeros(self.dimension, dtype=np.complex128)
        for block in self.blocks:
            value = form.blocks.get(block.key)
            if value is None:
                continue
            index = tuple(block.indices[:, axis] for axis in range(block.indices.shape[1]))
            out[block.vector_positions] = value[index] if index else value
        return out

    def subset(self, selected: ArrayLike) -> "ComponentCoordinates":
        """Return selected canonical coordinates in the supplied order."""

        raw = np.asarray(selected)
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise TypeError("selected component positions must be one-dimensional integers")
        indices = np.asarray(raw, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.dimension):
            raise ValueError("a selected component position lies outside the coordinate map")
        if np.unique(indices).size != indices.size:
            raise ValueError("selected component positions must be unique")
        new_position = {int(old): new for new, old in enumerate(indices)}
        blocks: list[BlockCoordinates] = []
        for block in self.blocks:
            rows = [row for row, old in enumerate(block.vector_positions) if int(old) in new_position]
            if not rows:
                continue
            rows.sort(key=lambda row: new_position[int(block.vector_positions[row])])
            blocks.append(
                BlockCoordinates(
                    block.key,
                    np.asarray(
                        [new_position[int(block.vector_positions[row])] for row in rows],
                        dtype=np.int64,
                    ),
                    block.indices[np.asarray(rows, dtype=np.int64)],
                )
            )
        return ComponentCoordinates(self.n, indices.size, tuple(blocks))

    def tiles(self, tile_size: int) -> Iterator[BlockCoordinates]:
        """Yield bounded slices while preserving the stored vector positions."""

        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        for block in self.blocks:
            for start in range(0, block.vector_positions.size, tile_size):
                stop = min(start + tile_size, block.vector_positions.size)
                yield BlockCoordinates(
                    block.key,
                    block.vector_positions[start:stop],
                    block.indices[start:stop],
                )


def _charge_shell_keys(
    n: int,
    charge: int,
    creator_rank: int,
    *,
    response_spin: int = 0,
    symmetry: str = "spin",
) -> tuple[BlockKey, ...]:
    if n <= 0 or creator_rank < 0:
        raise ValueError("n must be positive and creator_rank must be nonnegative")
    if response_spin not in (0, 1):
        raise ValueError("response_spin must be 0 or 1")
    if symmetry not in ("spin", "charge"):
        raise ValueError("symmetry must be 'spin' or 'charge'")
    if symmetry == "charge" and response_spin != 0:
        raise ValueError("charge symmetry requires response_spin=0")
    target = (charge, 0) if response_spin == 0 else (0, charge)
    counts = range(min(n, creator_rank) + 1)
    other_counts = counts if symmetry == "spin" else (0,)
    return tuple(
        key
        for key in product(counts, counts, other_counts, other_counts)
        if sum(key) == creator_rank
        and key[0] - key[1] == target[0]
        and key[2] - key[3] == target[1]
    )


def _unrank_combinations(n: int, count: int, ranks: IntArray) -> IntArray:
    """Return lexicographic ``count``-subsets at the requested zero-based ranks."""

    values = np.asarray(ranks, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("combination ranks must be one-dimensional")
    dimension = comb(n, count)
    if np.any(values < 0) or np.any(values >= dimension):
        raise ValueError("a combination rank lies outside its block")
    if count == 0:
        return np.empty((values.size, 0), dtype=np.int64)
    out = np.empty((values.size, count), dtype=np.int64)
    for row, raw_rank in enumerate(values):
        rank = int(raw_rank)
        lower = 0
        for position in range(count):
            remaining = count - position - 1
            for candidate in range(lower, n - remaining):
                suffixes = comb(n - candidate - 1, remaining)
                if rank < suffixes:
                    out[row, position] = candidate
                    lower = candidate + 1
                    break
                rank -= suffixes
    return out


def _block_coordinate_tiles(
    n: int,
    key: BlockKey,
    tile_size: int,
    *,
    position_offset: int = 0,
) -> Iterator[BlockCoordinates]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    choices = tuple(
        np.empty((1, 0), dtype=np.int64)
        if count == 0
        else np.asarray(tuple(combinations(range(n), count)), dtype=np.int64)
        for count in key
    )
    dimensions = tuple(choice.shape[0] for choice in choices)
    size = prod(dimensions)
    for start in range(0, size, tile_size):
        stop = min(start + tile_size, size)
        flat = np.arange(start, stop, dtype=np.int64)
        multi = np.unravel_index(flat, dimensions)
        indices = np.concatenate(
            tuple(
                choice[axis]
                for choice, axis in zip(choices, multi, strict=True)
            ),
            axis=1,
        )
        yield BlockCoordinates(
            key,
            np.arange(position_offset + start, position_offset + stop, dtype=np.int64),
            indices,
        )


@dataclass(frozen=True, slots=True)
class _ShellBlock:
    key: BlockKey
    offset: int
    dimension: int


@dataclass(frozen=True, slots=True)
class ComponentCoordinateShell:
    """Lazy canonical coordinates for one fixed-rank charge shell."""

    n: int
    response_spin: int
    charge: int
    creator_rank: int
    symmetry: str = "spin"
    layouts: tuple[_ShellBlock, ...] = field(init=False)
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        if self.n <= 0 or self.creator_rank < 0:
            raise ValueError("n must be positive and creator_rank must be nonnegative")
        if self.response_spin not in (0, 1):
            raise ValueError("response_spin must be 0 or 1")
        layouts: list[_ShellBlock] = []
        offset = 0
        for key in _charge_shell_keys(
            self.n,
            self.charge,
            self.creator_rank,
            response_spin=self.response_spin,
            symmetry=self.symmetry,
        ):
            size = prod(comb(self.n, count) for count in key)
            if size:
                layouts.append(_ShellBlock(key, offset, size))
                offset += size
        object.__setattr__(self, "layouts", tuple(layouts))
        object.__setattr__(self, "dimension", offset)

    def tiles(self, tile_size: int) -> Iterator[BlockCoordinates]:
        """Generate canonical coordinate tiles without storing the shell."""

        for layout in self.layouts:
            yield from _block_coordinate_tiles(
                self.n,
                layout.key,
                tile_size,
                position_offset=layout.offset,
            )

    def subset(self, selected: ArrayLike) -> ComponentCoordinates:
        """Reconstruct only the selected global shell coordinates."""

        raw = np.asarray(selected)
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise TypeError("selected shell positions must be one-dimensional integers")
        indices = np.asarray(raw, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.dimension):
            raise ValueError("a selected shell position lies outside the coordinate map")
        if np.unique(indices).size != indices.size:
            raise ValueError("selected shell positions must be unique")
        blocks: list[BlockCoordinates] = []
        for layout in self.layouts:
            positions = np.flatnonzero(
                (indices >= layout.offset)
                & (indices < layout.offset + layout.dimension)
            ).astype(np.int64)
            if not positions.size:
                continue
            local = indices[positions] - layout.offset
            dimensions = tuple(comb(self.n, count) for count in layout.key)
            multi = np.unravel_index(local, dimensions)
            coordinates = np.concatenate(
                tuple(
                    _unrank_combinations(
                        self.n,
                        count,
                        np.asarray(axis, dtype=np.int64),
                    )
                    for count, axis in zip(layout.key, multi, strict=True)
                ),
                axis=1,
            )
            blocks.append(BlockCoordinates(layout.key, positions, coordinates))
        return ComponentCoordinates(self.n, indices.size, tuple(blocks))


CoordinateSet = ComponentCoordinates | ComponentCoordinateShell


@dataclass(frozen=True, slots=True)
class _LinearData:
    creator: ExteriorForm
    annihilator: Mapping[int, ComplexArray]


def _linear_data(operator: LinearModeOperator, n: int) -> _LinearData:
    creator_blocks: dict[BlockKey, ComplexArray] = {}
    annihilator: dict[int, ComplexArray] = {}
    for kind, mode, coefficient in operator.components:
        group, index = divmod(mode, n)
        if kind == "create":
            key = tuple(1 if g == group else 0 for g in range(4))
            array = creator_blocks.setdefault(key, np.zeros(n, dtype=np.complex128))
            array[index] += coefficient
        else:
            array = annihilator.setdefault(group, np.zeros(n, dtype=np.complex128))
            array[index] += coefficient
    return _LinearData(ExteriorForm(n, creator_blocks), MappingProxyType(annihilator))


@dataclass(frozen=True, slots=True)
class _Term:
    coefficient: complex
    factors: tuple[ExteriorForm, ...]

    @property
    def degree(self) -> int:
        return sum(form.degrees[0] for form in self.factors) if self.factors else 0


@dataclass(frozen=True, slots=True)
class _StaticLinearizationRecord:
    factor_keys: tuple[LinearKey, ...]
    primal_stages: tuple[tuple[_Term, ...], ...]


@dataclass(frozen=True, slots=True)
class _ProductTrieNode:
    """One applied-order factor in the exact static product trie."""

    parent: int
    factor_key: LinearKey | None
    child_count: int


@dataclass(frozen=True, slots=True)
class _ProductTrieLeaf:
    """One aggregated factor sequence and its exact scalar coefficient."""

    node: int
    coefficient: complex
    path: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ProductTrie:
    """Deterministic shared-prefix representation of static products."""

    nodes: tuple[_ProductTrieNode, ...]
    leaves: tuple[_ProductTrieLeaf, ...]
    input_record_count: int
    input_stage_count: int

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def shared_stage_count(self) -> int:
        return len(self.nodes) - 1


@dataclass(frozen=True, slots=True)
class ComponentProductDiagnostics:
    """Deterministic structural counts for the optional fused evaluator."""

    raw_record_count: int
    aggregated_record_count: int
    coefficient_aggregation_ratio: float
    raw_stage_count: int
    trie_node_count: int
    shared_stage_count: int
    prefix_depth_distribution: Mapping[int, int]
    terminal_depth_distribution: Mapping[int, int]


def _product_diagnostics(
    trie: _ProductTrie,
) -> ComponentProductDiagnostics:
    depths = [0] * len(trie.nodes)
    prefix_depths: dict[int, int] = {}
    for node_index, node in enumerate(trie.nodes[1:], start=1):
        depth = depths[node.parent] + 1
        depths[node_index] = depth
        prefix_depths[depth] = prefix_depths.get(depth, 0) + 1
    terminal_depths: dict[int, int] = {}
    for leaf in trie.leaves:
        depth = depths[leaf.node]
        terminal_depths[depth] = terminal_depths.get(depth, 0) + 1
    aggregated = len(trie.leaves)
    return ComponentProductDiagnostics(
        raw_record_count=trie.input_record_count,
        aggregated_record_count=aggregated,
        coefficient_aggregation_ratio=(
            float(trie.input_record_count / aggregated)
            if aggregated
            else 0.0
        ),
        raw_stage_count=trie.input_stage_count,
        trie_node_count=trie.node_count,
        shared_stage_count=trie.shared_stage_count,
        prefix_depth_distribution=MappingProxyType(prefix_depths),
        terminal_depth_distribution=MappingProxyType(terminal_depths),
    )


def _compile_product_trie(
    products: Sequence[FactorizedModeProduct],
) -> _ProductTrie:
    """Aggregate equal products and share their applied-order prefixes.

    Factorized records remain unchanged in the oracle.  This compiler only
    changes the evaluation graph: factors are keyed by their complete linear
    component tuples, and coefficients of identical ordered products are
    summed in first-occurrence order.
    """

    aggregated: dict[tuple[LinearKey, ...], complex] = {}
    input_stage_count = 0
    for record in products:
        factor_keys = tuple(
            factor.components for factor in reversed(record.factors)
        )
        input_stage_count += len(factor_keys)
        aggregated[factor_keys] = (
            aggregated.get(factor_keys, 0.0 + 0.0j)
            + record.coefficient
        )

    parents = [-1]
    factor_keys: list[LinearKey | None] = [None]
    child_counts = [0]
    edges: dict[tuple[int, LinearKey], int] = {}
    leaves: list[_ProductTrieLeaf] = []
    for sequence, coefficient in aggregated.items():
        if coefficient == 0.0:
            continue
        parent = 0
        path: list[int] = []
        for key in sequence:
            edge = (parent, key)
            node = edges.get(edge)
            if node is None:
                node = len(parents)
                edges[edge] = node
                parents.append(parent)
                factor_keys.append(key)
                child_counts.append(0)
                child_counts[parent] += 1
            parent = node
            path.append(node)
        leaves.append(
            _ProductTrieLeaf(
                node=parent,
                coefficient=coefficient,
                path=tuple(path),
            )
        )
    nodes = tuple(
        _ProductTrieNode(parent, key, children)
        for parent, key, children in zip(
            parents, factor_keys, child_counts, strict=True
        )
    )
    return _ProductTrie(
        nodes=nodes,
        leaves=tuple(leaves),
        input_record_count=len(products),
        input_stage_count=input_stage_count,
    )


def _scaled_terms(
    terms: Iterable[_Term],
    coefficient: complex,
) -> Iterator[_Term]:
    for term in terms:
        yield _Term(coefficient * term.coefficient, term.factors)


def _nonzero(form: ExteriorForm) -> bool:
    return bool(form.blocks)


def _split_homogeneous(form: ExteriorForm) -> tuple[ExteriorForm, ...]:
    return tuple(form.homogeneous(degree) for degree in form.degrees)


def _apply_factor(
    states: Sequence[_Term],
    data: _LinearData,
    shifts: Sequence[ExteriorForm],
) -> list[_Term]:
    out: list[_Term] = []
    wedge_forms = ((data.creator,) if _nonzero(data.creator) else ()) + tuple(shifts)
    for term in states:
        for form in wedge_forms:
            out.append(_Term(term.coefficient, (form,) + term.factors))
        if data.annihilator:
            prefix_degree = 0
            for position, form in enumerate(term.factors):
                contracted = interior(data.annihilator, form)
                if _nonzero(contracted):
                    out.append(
                        _Term(
                            term.coefficient * ((-1) ** prefix_degree),
                            term.factors[:position] + (contracted,) + term.factors[position + 1 :],
                        )
                    )
                prefix_degree += form.degrees[0]
    return out


def _apply_shift_derivative(states: Sequence[_Term], shifts: Sequence[ExteriorForm]) -> list[_Term]:
    return [
        _Term(term.coefficient, (shift,) + term.factors)
        for term in states
        for shift in shifts
    ]


def _product_terms(
    product_record: FactorizedModeProduct,
    linear: Mapping[tuple[tuple[str, int, complex], ...], _LinearData],
    amplitudes: ExteriorForm,
    direction: ExteriorForm | None = None,
    initial: Sequence[_Term] | None = None,
) -> tuple[list[_Term], list[_Term] | None]:
    primal = list(initial) if initial is not None else [_Term(product_record.coefficient, ())]
    tangent: list[_Term] | None = [] if direction is not None else None
    for factor in reversed(product_record.factors):
        data = linear[factor.components]
        shifts = _split_homogeneous(interior(data.annihilator, amplitudes))
        old_primal = primal
        old_tangent = tangent
        primal = _apply_factor(old_primal, data, shifts)
        if direction is not None and old_tangent is not None:
            direction_shifts = _split_homogeneous(interior(data.annihilator, direction))
            tangent = _apply_factor(old_tangent, data, shifts)
            tangent.extend(_apply_shift_derivative(old_primal, direction_shifts))
    return primal, tangent


def _readonly_complex(value: ArrayLike) -> ComplexArray:
    array = np.array(value, dtype=np.complex128, copy=True, order="C")
    array.flags.writeable = False
    return array


def _freeze_form(
    form: ExteriorForm,
    memo: dict[int, tuple[ExteriorForm, ExteriorForm]],
) -> ExteriorForm:
    cached = memo.get(id(form))
    if cached is not None and cached[0] is form:
        return cached[1]
    frozen = ExteriorForm(
        form.n,
        {key: _readonly_complex(value) for key, value in form.blocks.items()},
    )
    memo[id(form)] = (form, frozen)
    memo[id(frozen)] = (frozen, frozen)
    return frozen


def _freeze_terms(
    terms: Iterable[_Term],
    memo: dict[int, tuple[ExteriorForm, ExteriorForm]],
) -> tuple[_Term, ...]:
    return tuple(
        _Term(
            term.coefficient,
            tuple(_freeze_form(form, memo) for form in term.factors),
        )
        for term in terms
    )


def _freeze_linear_data(
    data: _LinearData,
    memo: dict[int, tuple[ExteriorForm, ExteriorForm]],
) -> _LinearData:
    return _LinearData(
        creator=_freeze_form(data.creator, memo),
        annihilator=MappingProxyType(
            {
                group: _readonly_complex(vector)
                for group, vector in data.annihilator.items()
            }
        ),
    )


def _accumulate_materialized_terms(
    blocks: dict[BlockKey, ComplexArray],
    terms: Iterable[_Term],
    n: int,
    degrees: set[int],
) -> None:
    """Materialize and immediately accumulate lazy terms.

    At target sizes a degree-four block is large enough that retaining one
    materialized block per interaction term exhausts memory.  This routine
    preserves the term order but keeps only the running block sums and the
    current term live.
    """

    for term in terms:
        if term.degree not in degrees:
            continue
        value = ExteriorForm(
            n,
            {
                _zero_key(): np.asarray(
                    term.coefficient, dtype=np.complex128
                )
            },
        )
        for factor in term.factors:
            value = wedge(
                value, factor, maximum_degree=term.degree
            )
        _accumulate_form(blocks, value, n)


def _accumulate_materialized_terms_raw_unbatched(
    blocks: dict[BlockKey, ComplexArray],
    terms: Iterable[_Term],
    n: int,
    degrees: set[int],
) -> None:
    """Trusted direct materializer used only by the fused static path."""

    for term in terms:
        if term.degree not in degrees or term.coefficient == 0.0:
            continue
        if not term.factors:
            key = _zero_key()
            value = np.asarray(term.coefficient, dtype=np.complex128)
            if key in blocks:
                blocks[key] += value
            else:
                blocks[key] = value.copy()
            continue

        first = term.factors[0]
        if first.n != n:
            raise ValueError(
                "a materialized term uses the wrong one-group dimension"
            )
        current = {
            key: term.coefficient * value
            for key, value in first.blocks.items()
            if np.any(value)
        }
        for factor in term.factors[1:]:
            if factor.n != n:
                raise ValueError(
                    "a materialized term uses the wrong one-group dimension"
                )
            updated: dict[BlockKey, ComplexArray] = {}
            for left_key, left in current.items():
                for right_key, right in factor.blocks.items():
                    key = _add_key(left_key, right_key)
                    if (
                        any(count > n for count in key)
                        or _degree(key) > term.degree
                    ):
                        continue
                    value = _wedge_blocks(
                        left, left_key, right, right_key
                    )
                    if key in updated:
                        updated[key] += value
                    else:
                        updated[key] = value
            current = {
                key: value
                for key, value in updated.items()
                if np.any(value)
            }
            if not current:
                break
        for key, value in current.items():
            if key in blocks:
                blocks[key] += value
            else:
                blocks[key] = value.copy()


@dataclass(slots=True)
class _RawMaterializationStats:
    contribution_count: int = 0
    signature_group_count: int = 0
    batch_count: int = 0
    contraction_launch_count: int = 0
    unbatched_contraction_launch_count: int = 0


@dataclass(frozen=True, slots=True)
class ComponentRawMaterializationDiagnostics:
    contribution_count: int
    signature_group_count: int
    batch_count: int
    contraction_launch_count: int
    unbatched_contraction_launch_count: int


def _raw_materialization_diagnostics(
    stats: _RawMaterializationStats,
) -> ComponentRawMaterializationDiagnostics:
    return ComponentRawMaterializationDiagnostics(
        contribution_count=stats.contribution_count,
        signature_group_count=stats.signature_group_count,
        batch_count=stats.batch_count,
        contraction_launch_count=stats.contraction_launch_count,
        unbatched_contraction_launch_count=(
            stats.unbatched_contraction_launch_count
        ),
    )


def _raw_term_contributions(
    term: _Term,
    n: int,
) -> Iterator[
    tuple[complex, tuple[BlockKey, ...], tuple[ComplexArray, ...]]
]:
    if not term.factors:
        yield term.coefficient, (), ()
        return
    choices: list[tuple[tuple[BlockKey, ComplexArray], ...]] = []
    for factor in term.factors:
        if factor.n != n:
            raise ValueError(
                "a materialized term uses the wrong one-group dimension"
            )
        choices.append(tuple(factor.blocks.items()))
    for selected in product(*choices):
        coefficient = term.coefficient
        keys: list[BlockKey] = []
        arrays: list[ComplexArray] = []
        for key, value in selected:
            if key == _zero_key():
                coefficient *= complex(value)
            else:
                keys.append(key)
                arrays.append(value)
        if coefficient != 0.0:
            yield coefficient, tuple(keys), tuple(arrays)


def _shuffle_count(left: BlockKey, right: BlockKey) -> int:
    return prod(
        comb(n_left + n_right, n_left)
        for n_left, n_right in zip(left, right, strict=True)
    )


def _accumulate_materialized_terms_raw(
    blocks: dict[BlockKey, ComplexArray],
    terms: Iterable[_Term],
    n: int,
    degrees: set[int],
    *,
    batch_size: int = 16,
    stats: _RawMaterializationStats | None = None,
) -> None:
    """Batch equal block signatures without a batched rank-four output."""

    if batch_size <= 0 or batch_size > 16:
        raise ValueError("raw materialization batch_size must be in [1, 16]")
    grouped: dict[
        tuple[BlockKey, ...],
        list[tuple[complex, tuple[ComplexArray, ...]]],
    ] = {}
    for term in terms:
        if term.degree not in degrees or term.coefficient == 0.0:
            continue
        for coefficient, signature, arrays in _raw_term_contributions(
            term, n
        ):
            grouped.setdefault(signature, []).append(
                (coefficient, arrays)
            )
            if stats is not None:
                stats.contribution_count += 1

    if stats is not None:
        stats.signature_group_count += len(grouped)
    for signature, contributions in grouped.items():
        key = _zero_key()
        for factor_key in signature:
            key = _add_key(key, factor_key)
        if any(count > n for count in key) or _degree(key) not in degrees:
            continue
        for start in range(0, len(contributions), batch_size):
            batch = contributions[start : start + batch_size]
            if stats is not None:
                stats.batch_count += 1
            if not signature:
                value = np.asarray(
                    sum(
                        (coefficient for coefficient, _ in batch),
                        0.0 + 0.0j,
                    ),
                    dtype=np.complex128,
                )
            elif len(signature) == 1 and _degree(key) >= 4:
                value = np.zeros((n,) * _degree(key), dtype=np.complex128)
                for coefficient, arrays in batch:
                    value += coefficient * arrays[0]
            elif len(signature) == 1:
                value = np.sum(
                    np.stack(
                        [
                            coefficient * arrays[0]
                            for coefficient, arrays in batch
                        ]
                    ),
                    axis=0,
                )
            else:
                current = np.stack(
                    [
                        coefficient * arrays[0]
                        for coefficient, arrays in batch
                    ]
                )
                current_key = signature[0]
                for position, right_key in enumerate(
                    signature[1:], start=1
                ):
                    right = np.stack(
                        [arrays[position] for _, arrays in batch]
                    )
                    launches = _shuffle_count(
                        current_key, right_key
                    )
                    if stats is not None:
                        stats.contraction_launch_count += launches
                        stats.unbatched_contraction_launch_count += (
                            len(batch) * launches
                        )
                    final = position + 1 == len(signature)
                    current = _wedge_blocks_batched(
                        current,
                        current_key,
                        right,
                        right_key,
                        sum_batch=final,
                    )
                    current_key = _add_key(
                        current_key, right_key
                    )
                value = current
            if key in blocks:
                blocks[key] += value
            else:
                blocks[key] = np.asarray(
                    value, dtype=np.complex128
                ).copy()


def _materialize_terms(terms: Iterable[_Term], n: int, degrees: set[int]) -> ExteriorForm:
    blocks: dict[BlockKey, ComplexArray] = {}
    _accumulate_materialized_terms(blocks, terms, n, degrees)
    return ExteriorForm(n, blocks)


def _materialize_terms_raw(
    terms: Iterable[_Term],
    n: int,
    degrees: set[int],
    *,
    batch_size: int = 16,
    stats: _RawMaterializationStats | None = None,
) -> ExteriorForm:
    blocks: dict[BlockKey, ComplexArray] = {}
    _accumulate_materialized_terms_raw(
        blocks,
        terms,
        n,
        degrees,
        batch_size=batch_size,
        stats=stats,
    )
    return ExteriorForm(n, blocks)


def _evaluate_product_trie(
    trie: _ProductTrie,
    linear: Mapping[LinearKey, _LinearData],
    shifts: Mapping[LinearKey, Sequence[ExteriorForm]],
    n: int,
    degrees: set[int],
    *,
    direction_shifts: Mapping[
        LinearKey, Sequence[ExteriorForm]
    ] | None = None,
    stats: _RawMaterializationStats | None = None,
) -> dict[BlockKey, ComplexArray]:
    """Evaluate the exact static interaction on one shared-prefix graph."""

    leaf_coefficients = {
        leaf.node: leaf.coefficient for leaf in trie.leaves
    }
    primal_nodes: list[tuple[_Term, ...] | None] = [
        (_Term(1.0 + 0.0j, ()),)
    ]
    tangent_nodes: list[tuple[_Term, ...] | None] | None = (
        [()] if direction_shifts is not None else None
    )
    blocks: dict[BlockKey, ComplexArray] = {}
    for node_index, node in enumerate(trie.nodes[1:], start=1):
        key = node.factor_key
        if key is None:
            raise RuntimeError("a non-root product-trie node has no factor")
        parent_primal = primal_nodes[node.parent]
        if parent_primal is None:
            raise RuntimeError("a product-trie parent state was discarded")
        primal = tuple(
            _apply_factor(parent_primal, linear[key], shifts[key])
        )
        tangent: tuple[_Term, ...] | None = None
        if tangent_nodes is not None and direction_shifts is not None:
            parent_tangent = tangent_nodes[node.parent]
            if parent_tangent is None:
                raise RuntimeError(
                    "a product-trie parent tangent was discarded"
                )
            updated = _apply_factor(
                parent_tangent, linear[key], shifts[key]
            )
            updated.extend(
                _apply_shift_derivative(
                    parent_primal, direction_shifts[key]
                )
            )
            tangent = tuple(updated)
        coefficient = leaf_coefficients.get(node_index)
        if coefficient is not None:
            selected = primal if tangent_nodes is None else tangent
            if selected is None:
                raise RuntimeError("a product-trie tangent is missing")
            _accumulate_materialized_terms_raw(
                blocks,
                _scaled_terms(selected, coefficient),
                n,
                degrees,
                stats=stats,
            )
        primal_nodes.append(primal if node.child_count else None)
        if tangent_nodes is not None:
            tangent_nodes.append(
                tangent if node.child_count else None
            )
    return blocks


def _quadratic_action(form: ExteriorForm, rapidities: ComplexArray) -> ExteriorForm:
    n = form.n
    blocks: dict[BlockKey, ComplexArray] = {}
    for key, value in form.blocks.items():
        multiplier = np.zeros(value.shape, dtype=np.complex128)
        axis = 0
        for group, count in enumerate(key):
            lam = rapidities[group * n : (group + 1) * n]
            for _ in range(count):
                shape = [1] * value.ndim
                shape[axis] = n
                multiplier += lam.reshape(shape)
                axis += 1
        blocks[key] = multiplier * value
    return ExteriorForm(n, blocks)


@dataclass(frozen=True, slots=True)
class ComponentStaticProblem:
    """Padded component representation of one factorized CCSD problem."""

    oracle: FactorizedCCProblem
    coordinates: ComponentCoordinates
    next_coordinates: ComponentCoordinates | None
    linear: Mapping[LinearKey, _LinearData]
    product_trie: _ProductTrie | None = None
    fuse_products: bool = False

    @property
    def compiled_product_trie(self) -> _ProductTrie:
        """Return the compiled trie, with a compatibility path for fixtures."""

        if self.product_trie is not None:
            return self.product_trie
        return _compile_product_trie(self.oracle.products)

    @property
    def product_diagnostics(self) -> ComponentProductDiagnostics:
        return _product_diagnostics(self.compiled_product_trie)

    @classmethod
    def compile(
        cls,
        problem: FactorizedCCProblem,
        *,
        fuse_products: bool = False,
    ) -> "ComponentStaticProblem":
        if problem.level != "CCSD":
            raise ValueError("the component backend currently implements CCSD only")
        n = problem.model.n_sites
        factors = {factor.components: _linear_data(factor, n) for record in problem.products for factor in record.factors}
        return cls(
            oracle=problem,
            coordinates=ComponentCoordinates.from_masks(n, problem.cluster_basis.masks),
            next_coordinates=(
                ComponentCoordinates.from_masks(n, problem.next_rank_masks)
                if problem.next_rank_masks
                else None
            ),
            linear=MappingProxyType(factors),
            product_trie=_compile_product_trie(problem.products),
            fuse_products=bool(fuse_products),
        )


@dataclass(frozen=True, slots=True)
class ComponentStaticAction:
    retained: ComplexArray
    reference: complex
    wall_seconds: float
    materialization: ComponentRawMaterializationDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class ComponentStaticLinearization:
    """Immutable fixed-amplitude data for repeated analytic static JVPs."""

    problem: ComponentStaticProblem
    amplitudes: ComplexArray
    residual: ComponentStaticAction
    amplitude_shifts: Mapping[
        tuple[tuple[str, int, complex], ...],
        tuple[ExteriorForm, ...],
    ]
    _linear: Mapping[
        LinearKey,
        _LinearData,
    ]
    _records: tuple[_StaticLinearizationRecord, ...]
    _primal_nodes: tuple[tuple[_Term, ...] | None, ...]
    owned_bytes: int
    factor_key_count: int
    amplitude_shift_count: int
    primal_stage_count: int
    record_count: int
    aggregated_record_count: int
    trie_node_count: int
    shared_primal_stage_count: int


def _linearization_owned_bytes(
    amplitudes: ComplexArray,
    residual: ComponentStaticAction,
    linear: Mapping[
        LinearKey,
        _LinearData,
    ],
    shifts: Mapping[
        LinearKey,
        tuple[ExteriorForm, ...],
    ],
    primal_nodes: Sequence[tuple[_Term, ...] | None],
) -> int:
    arrays: list[ComplexArray] = [amplitudes, residual.retained]
    for data in linear.values():
        arrays.extend(data.creator.blocks.values())
        arrays.extend(data.annihilator.values())
    for forms in shifts.values():
        for form in forms:
            arrays.extend(form.blocks.values())
    for terms in primal_nodes:
        if terms is None:
            continue
        for term in terms:
            for form in term.factors:
                arrays.extend(form.blocks.values())
    seen: set[int] = set()
    total = 0
    for array in arrays:
        if id(array) not in seen:
            seen.add(id(array))
            total += int(array.nbytes)
    return total


def component_static_linearization(
    problem: ComponentStaticProblem,
    amplitudes: ArrayLike,
) -> ComponentStaticLinearization:
    """Build the residual and fixed-amplitude primal stages for exact JVPs."""

    started = perf_counter()
    frozen_amplitudes = _readonly_complex(amplitudes)
    t = problem.coordinates.unpack(frozen_amplitudes)
    form_memo: dict[int, tuple[ExteriorForm, ExteriorForm]] = {}
    linear = MappingProxyType(
        {
            key: _freeze_linear_data(data, form_memo)
            for key, data in problem.linear.items()
        }
    )
    shifts = MappingProxyType(
        {
            key: tuple(
                _freeze_form(form, form_memo)
                for form in _split_homogeneous(
                    interior(data.annihilator, t)
                )
            )
            for key, data in linear.items()
        }
    )

    trie = problem.compiled_product_trie
    interaction_blocks: dict[BlockKey, ComplexArray] = {}
    records: list[_StaticLinearizationRecord] = []
    primal_nodes: list[tuple[_Term, ...] | None] = []
    raw_stats = _RawMaterializationStats()
    if problem.fuse_products:
        leaf_coefficients = {
            leaf.node: leaf.coefficient for leaf in trie.leaves
        }
        primal_nodes.append((_Term(1.0 + 0.0j, ()),))
        for node_index, node in enumerate(
            trie.nodes[1:], start=1
        ):
            key = node.factor_key
            if key is None:
                raise RuntimeError(
                    "a non-root product-trie node has no factor"
                )
            parent_primal = primal_nodes[node.parent]
            if parent_primal is None:
                raise RuntimeError(
                    "a product-trie parent state was discarded"
                )
            updated = tuple(
                _apply_factor(
                    parent_primal, linear[key], shifts[key]
                )
            )
            coefficient = leaf_coefficients.get(node_index)
            if coefficient is not None:
                _accumulate_materialized_terms_raw(
                    interaction_blocks,
                    _scaled_terms(updated, coefficient),
                    t.n,
                    {0, 2, 4},
                    stats=raw_stats,
                )
            primal_nodes.append(
                _freeze_terms(updated, form_memo)
                if node.child_count
                else None
            )

        for leaf in trie.leaves:
            keys = tuple(
                trie.nodes[node].factor_key for node in leaf.path
            )
            if any(key is None for key in keys):
                raise RuntimeError(
                    "a product-trie path contains the root"
                )
            stage_indices = (0,) + leaf.path[:-1]
            stages = tuple(
                primal_nodes[node] for node in stage_indices
            )
            if any(stage is None for stage in stages):
                raise RuntimeError(
                    "a product-trie path lost an internal primal stage"
                )
            records.append(
                _StaticLinearizationRecord(
                    factor_keys=tuple(keys),  # type: ignore[arg-type]
                    primal_stages=tuple(stages),  # type: ignore[arg-type]
                )
            )
    else:
        for record in problem.oracle.products:
            factor_keys = tuple(
                factor.components
                for factor in reversed(record.factors)
            )
            primal = (_Term(record.coefficient, ()),)
            stages: list[tuple[_Term, ...]] = []
            for position, key in enumerate(factor_keys):
                stages.append(primal)
                updated = _apply_factor(
                    primal, linear[key], shifts[key]
                )
                primal = (
                    _freeze_terms(updated, form_memo)
                    if position + 1 < len(factor_keys)
                    else tuple(updated)
                )
            _accumulate_materialized_terms(
                interaction_blocks,
                primal,
                t.n,
                {0, 2, 4},
            )
            records.append(
                _StaticLinearizationRecord(
                    factor_keys, tuple(stages)
                )
            )

    interaction = ExteriorForm(t.n, interaction_blocks)
    total = _sum_forms(
        (_quadratic_action(t, problem.oracle.rapidities), interaction),
        t.n,
    )
    retained = _readonly_complex(-1j * problem.coordinates.pack(total))
    reference = -1j * complex(
        interaction.blocks.get(
            _zero_key(), np.asarray(0.0 + 0.0j)
        )
    )
    residual = ComponentStaticAction(
        retained=retained,
        reference=reference,
        wall_seconds=float(perf_counter() - started),
        materialization=(
            _raw_materialization_diagnostics(raw_stats)
            if problem.fuse_products
            else None
        ),
    )
    frozen_records = tuple(records)
    frozen_primal_nodes = tuple(primal_nodes)
    cached_primal_states = (
        frozen_primal_nodes
        if problem.fuse_products
        else tuple(
            stage
            for record in frozen_records
            for stage in record.primal_stages
        )
    )
    owned_bytes = _linearization_owned_bytes(
        frozen_amplitudes,
        residual,
        linear,
        shifts,
        cached_primal_states,
    )
    return ComponentStaticLinearization(
        problem=problem,
        amplitudes=frozen_amplitudes,
        residual=residual,
        amplitude_shifts=shifts,
        _linear=linear,
        _records=frozen_records,
        _primal_nodes=frozen_primal_nodes,
        owned_bytes=owned_bytes,
        factor_key_count=len(linear),
        amplitude_shift_count=sum(len(forms) for forms in shifts.values()),
        primal_stage_count=trie.input_stage_count,
        record_count=trie.input_record_count,
        aggregated_record_count=len(trie.leaves),
        trie_node_count=trie.node_count,
        shared_primal_stage_count=trie.shared_stage_count,
    )


def component_static_linearized_jvp(
    context: ComponentStaticLinearization,
    direction: ArrayLike,
) -> ComponentStaticAction:
    """Apply the exact cached JVP at the context's fixed amplitudes."""

    started = perf_counter()
    dt = context.problem.coordinates.unpack(direction)
    direction_shifts = {
        key: _split_homogeneous(interior(data.annihilator, dt))
        for key, data in context._linear.items()
    }
    interaction_blocks: dict[BlockKey, ComplexArray] = {}
    raw_stats = _RawMaterializationStats()
    if context.problem.fuse_products:
        trie = context.problem.compiled_product_trie
        leaf_coefficients = {
            leaf.node: leaf.coefficient for leaf in trie.leaves
        }
        tangent_nodes: list[tuple[_Term, ...] | None] = [()]
        for node_index, node in enumerate(
            trie.nodes[1:], start=1
        ):
            key = node.factor_key
            if key is None:
                raise RuntimeError(
                    "a non-root product-trie node has no factor"
                )
            parent_tangent = tangent_nodes[node.parent]
            parent_primal = context._primal_nodes[node.parent]
            if parent_tangent is None or parent_primal is None:
                raise RuntimeError(
                    "a cached product-trie parent state was discarded"
                )
            tangent = _apply_factor(
                parent_tangent,
                context._linear[key],
                context.amplitude_shifts[key],
            )
            tangent.extend(
                _apply_shift_derivative(
                    parent_primal, direction_shifts[key]
                )
            )
            coefficient = leaf_coefficients.get(node_index)
            if coefficient is not None:
                _accumulate_materialized_terms_raw(
                    interaction_blocks,
                    _scaled_terms(tangent, coefficient),
                    dt.n,
                    {0, 2, 4},
                    stats=raw_stats,
                )
            tangent_nodes.append(
                tuple(tangent) if node.child_count else None
            )
    else:
        for record in context._records:
            tangent: list[_Term] = []
            for stage, key in zip(
                record.primal_stages,
                record.factor_keys,
                strict=True,
            ):
                tangent = _apply_factor(
                    tangent,
                    context._linear[key],
                    context.amplitude_shifts[key],
                )
                tangent.extend(
                    _apply_shift_derivative(
                        stage, direction_shifts[key]
                    )
                )
            _accumulate_materialized_terms(
                interaction_blocks,
                tangent,
                dt.n,
                {0, 2, 4},
            )
    interaction = ExteriorForm(dt.n, interaction_blocks)
    total = _sum_forms(
        (
            _quadratic_action(
                dt, context.problem.oracle.rapidities
            ),
            interaction,
        ),
        dt.n,
    )
    retained = -1j * context.problem.coordinates.pack(total)
    reference = -1j * complex(
        interaction.blocks.get(
            _zero_key(), np.asarray(0.0 + 0.0j)
        )
    )
    return ComponentStaticAction(
        retained=retained,
        reference=reference,
        wall_seconds=float(perf_counter() - started),
        materialization=(
            _raw_materialization_diagnostics(raw_stats)
            if context.problem.fuse_products
            else None
        ),
    )


def _product_root_subtrees(
    trie: _ProductTrie,
) -> tuple[tuple[int, ...], ...]:
    """Partition a prefix trie into independent root-child subtrees."""

    root: list[int | None] = [None] * len(trie.nodes)
    groups: dict[int, list[int]] = {}
    for index, node in enumerate(trie.nodes[1:], start=1):
        ancestor = index if node.parent == 0 else root[node.parent]
        if ancestor is None:
            raise RuntimeError("product-trie nodes are not parent ordered")
        root[index] = ancestor
        groups.setdefault(ancestor, []).append(index)
    return tuple(tuple(groups[index]) for index in sorted(groups))


def _subtree_linearized_blocks(
    context: ComponentStaticLinearization,
    direction_shifts: Mapping[
        LinearKey, Sequence[ExteriorForm]
    ],
    nodes: tuple[int, ...],
    leaf_coefficients: Mapping[int, complex],
) -> dict[BlockKey, ComplexArray]:
    tangent_nodes: dict[int, tuple[_Term, ...]] = {0: ()}
    blocks: dict[BlockKey, ComplexArray] = {}
    stats = _RawMaterializationStats()
    trie = context.problem.compiled_product_trie
    for index in nodes:
        node = trie.nodes[index]
        key = node.factor_key
        if key is None:
            raise RuntimeError(
                "a non-root product-trie node has no factor"
            )
        parent_tangent = tangent_nodes[node.parent]
        parent_primal = context._primal_nodes[node.parent]
        if parent_primal is None:
            raise RuntimeError(
                "a cached product-trie parent state was discarded"
            )
        tangent = _apply_factor(
            parent_tangent,
            context._linear[key],
            context.amplitude_shifts[key],
        )
        tangent.extend(
            _apply_shift_derivative(
                parent_primal, direction_shifts[key]
            )
        )
        coefficient = leaf_coefficients.get(index)
        if coefficient is not None:
            _accumulate_materialized_terms_raw(
                blocks,
                _scaled_terms(tangent, coefficient),
                context.problem.coordinates.n,
                {0, 2, 4},
                stats=stats,
            )
        if node.child_count:
            tangent_nodes[index] = tuple(tangent)
    return blocks


def component_static_linearized_jvp_parallel(
    context: ComponentStaticLinearization,
    direction: ArrayLike,
    *,
    workers: int,
) -> ComponentStaticAction:
    """Apply the exact fused JVP over independent product-trie subtrees."""

    if workers < 1:
        raise ValueError("JVP worker count must be positive")
    if workers == 1:
        return component_static_linearized_jvp(context, direction)
    if not context.problem.fuse_products:
        raise ValueError(
            "parallel subtree evaluation requires the fused product trie"
        )
    from concurrent.futures import ThreadPoolExecutor

    started = perf_counter()
    dt = context.problem.coordinates.unpack(direction)
    direction_shifts = {
        key: _split_homogeneous(interior(data.annihilator, dt))
        for key, data in context._linear.items()
    }
    trie = context.problem.compiled_product_trie
    subtrees = _product_root_subtrees(trie)
    leaf_coefficients = {
        leaf.node: leaf.coefficient for leaf in trie.leaves
    }

    def evaluate(nodes: tuple[int, ...]) -> dict[BlockKey, ComplexArray]:
        return _subtree_linearized_blocks(
            context,
            direction_shifts,
            nodes,
            leaf_coefficients,
        )

    with ThreadPoolExecutor(
        max_workers=min(workers, len(subtrees))
    ) as pool:
        partials = list(pool.map(evaluate, subtrees))
    interaction_blocks: dict[BlockKey, ComplexArray] = {}
    for partial in partials:
        for key, value in partial.items():
            if key in interaction_blocks:
                interaction_blocks[key] += value
            else:
                interaction_blocks[key] = np.asarray(
                    value, dtype=np.complex128
                ).copy()
    interaction = ExteriorForm(dt.n, interaction_blocks)
    total = _sum_forms(
        (
            _quadratic_action(
                dt, context.problem.oracle.rapidities
            ),
            interaction,
        ),
        dt.n,
    )
    retained = -1j * context.problem.coordinates.pack(total)
    reference = -1j * complex(
        interaction.blocks.get(
            _zero_key(), np.asarray(0.0 + 0.0j)
        )
    )
    return ComponentStaticAction(
        retained=retained,
        reference=reference,
        wall_seconds=float(perf_counter() - started),
        materialization=None,
    )


@dataclass(frozen=True, slots=True)
class ComponentRankDiagnostic:
    raw_norm: float
    normalized_norm: float
    normalization_scale: float
    block_norms: Mapping[BlockKey, float]
    tile_size: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class ComponentSampledRankDiagnostic:
    creator_rank: int
    shell_dimension: int
    sample_count: int
    estimated_raw_norm: float
    estimated_normalized_norm: float
    estimated_standard_error: float
    seed: int
    wall_seconds: float


def component_static_residual(problem: ComponentStaticProblem, amplitudes: ArrayLike) -> ComponentStaticAction:
    """Evaluate retained CCSD residual blocks without masks or transitions."""

    started = perf_counter()
    t = problem.coordinates.unpack(amplitudes)
    raw_stats = _RawMaterializationStats()
    if problem.fuse_products:
        shifts = {
            key: _split_homogeneous(
                interior(data.annihilator, t)
            )
            for key, data in problem.linear.items()
        }
        interaction_blocks = _evaluate_product_trie(
            problem.compiled_product_trie,
            problem.linear,
            shifts,
            t.n,
            {0, 2, 4},
            stats=raw_stats,
        )
    else:
        interaction_blocks = {}
        for record in problem.oracle.products:
            primal, _ = _product_terms(
                record, problem.linear, t
            )
            _accumulate_materialized_terms(
                interaction_blocks,
                primal,
                t.n,
                {0, 2, 4},
            )
    interaction = ExteriorForm(t.n, interaction_blocks)
    total = _sum_forms((_quadratic_action(t, problem.oracle.rapidities), interaction), t.n)
    retained = -1j * problem.coordinates.pack(total)
    reference = -1j * complex(interaction.blocks.get(_zero_key(), np.asarray(0.0 + 0.0j)))
    return ComponentStaticAction(
        retained,
        reference,
        float(perf_counter() - started),
        (
            _raw_materialization_diagnostics(raw_stats)
            if problem.fuse_products
            else None
        ),
    )


def component_static_jvp(
    problem: ComponentStaticProblem,
    amplitudes: ArrayLike,
    direction: ArrayLike,
) -> ComponentStaticAction:
    """Forward tangent of the component residual contraction graph."""

    started = perf_counter()
    t = problem.coordinates.unpack(amplitudes)
    dt = problem.coordinates.unpack(direction)
    raw_stats = _RawMaterializationStats()
    if problem.fuse_products:
        shifts = {
            key: _split_homogeneous(
                interior(data.annihilator, t)
            )
            for key, data in problem.linear.items()
        }
        direction_shifts = {
            key: _split_homogeneous(
                interior(data.annihilator, dt)
            )
            for key, data in problem.linear.items()
        }
        interaction_blocks = _evaluate_product_trie(
            problem.compiled_product_trie,
            problem.linear,
            shifts,
            t.n,
            {0, 2, 4},
            direction_shifts=direction_shifts,
            stats=raw_stats,
        )
    else:
        interaction_blocks = {}
        for record in problem.oracle.products:
            _, tangent = _product_terms(
                record, problem.linear, t, dt
            )
            if tangent is not None:
                _accumulate_materialized_terms(
                    interaction_blocks,
                    tangent,
                    t.n,
                    {0, 2, 4},
                )
    interaction = ExteriorForm(t.n, interaction_blocks)
    total = _sum_forms((_quadratic_action(dt, problem.oracle.rapidities), interaction), t.n)
    retained = -1j * problem.coordinates.pack(total)
    reference = -1j * complex(interaction.blocks.get(_zero_key(), np.asarray(0.0 + 0.0j)))
    return ComponentStaticAction(
        retained,
        reference,
        float(perf_counter() - started),
        (
            _raw_materialization_diagnostics(raw_stats)
            if problem.fuse_products
            else None
        ),
    )


def component_static_rank6_diagnostic(
    problem: ComponentStaticProblem,
    amplitudes: ArrayLike,
    *,
    normalization_scale: float,
    tile_size: int = 4096,
) -> ComponentRankDiagnostic:
    """Accumulate the complete rank-six coefficient norm in bounded tiles."""

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    scale = float(normalization_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("normalization_scale must be finite and nonnegative")
    started = perf_counter()
    t = problem.coordinates.unpack(amplitudes)
    terms: list[_Term] = []
    for record in problem.oracle.products:
        primal, _ = _product_terms(record, problem.linear, t)
        terms.extend(primal)
    norms: dict[BlockKey, float] = {}
    total = 0.0
    if problem.next_coordinates is None:
        blocks = (
            (key, _block_coordinate_tiles(t.n, key, tile_size))
            for key in _charge_shell_keys(t.n, 0, 6, symmetry=problem.oracle.model.symmetry)
        )
    else:
        blocks = (
            (
                block.key,
                (
                    BlockCoordinates(
                        block.key,
                        block.vector_positions[start : min(start + tile_size, block.vector_positions.size)],
                        block.indices[start : min(start + tile_size, block.vector_positions.size)],
                    )
                    for start in range(0, block.vector_positions.size, tile_size)
                ),
            )
            for block in problem.next_coordinates.blocks
        )
    for key, tiles in blocks:
        block_sum = 0.0
        for tile in tiles:
            values = np.zeros(tile.vector_positions.size, dtype=np.complex128)
            for term in terms:
                if term.degree == 6:
                    values += _term_values(term, tile)
            block_sum += float(np.vdot(values, values).real)
        norms[key] = float(np.sqrt(block_sum))
        total += block_sum
    raw = float(np.sqrt(total))
    normalized = (
        0.0
        if scale == 0.0 and raw == 0.0
        else raw / max(scale, np.finfo(float).tiny)
    )
    return ComponentRankDiagnostic(
        raw_norm=raw,
        normalized_norm=normalized,
        normalization_scale=scale,
        block_norms=MappingProxyType(norms),
        tile_size=tile_size,
        wall_seconds=float(perf_counter() - started),
    )


@dataclass(frozen=True, slots=True)
class ComponentEOMProblem:
    """One charge block of the rank-one/rank-three EOM component action."""

    oracle: EOMCCSDProblem
    static: ComponentStaticProblem
    charge: int
    positions: IntArray
    coordinates: ComponentCoordinates
    s5_coordinates: CoordinateSet

    @classmethod
    def compile(
        cls,
        problem: EOMCCSDProblem,
        static: ComponentStaticProblem,
        charge: int,
    ) -> "ComponentEOMProblem":
        if problem.cc is not static.oracle:
            raise ValueError("static and EOM component problems do not share a CC problem")
        if charge not in (-1, 1):
            raise ValueError("response charge must be +1 or -1")
        positions = np.flatnonzero(problem.basis.charge_labels[:, problem.response_spin] == charge).astype(np.int64)
        s5_positions = np.asarray(
            [
                i
                for i, mask in enumerate(problem.next_rank_masks)
                if problem.basis.layout.charge(mask)[problem.response_spin] == charge
            ],
            dtype=np.int64,
        )
        s5_coordinates = (
            ComponentCoordinates.from_masks(
                problem.cc.model.n_sites,
                problem.next_rank_masks,
                s5_positions,
            )
            if problem.next_rank_masks
            else ComponentCoordinateShell(
                n=problem.cc.model.n_sites,
                response_spin=problem.response_spin,
                charge=charge,
                creator_rank=5,
                symmetry=problem.basis.layout.symmetry,
            )
        )
        if 2 * s5_coordinates.dimension != problem.next_rank_dimension:
            raise ValueError("one S5 charge shell disagrees with EOM dimension metadata")
        return cls(
            oracle=problem,
            static=static,
            charge=charge,
            positions=positions,
            coordinates=ComponentCoordinates.from_masks(problem.cc.model.n_sites, problem.basis.masks, positions),
            s5_coordinates=s5_coordinates,
        )


@dataclass(frozen=True, slots=True)
class _Placeholder:
    degree: int
    contractions: tuple[Mapping[int, ComplexArray], ...] = ()

    @property
    def degrees(self) -> tuple[int]:
        return (self.degree,)


@dataclass(frozen=True, slots=True)
class _ResponsePlanTerm:
    input_degree: int
    coefficient: complex
    creation_factors: tuple[ExteriorForm, ...]
    contractions: tuple[Mapping[int, ComplexArray], ...]


def _apply_factor_plan(
    states: Sequence[_Term],
    data: _LinearData,
    shifts: Sequence[ExteriorForm],
) -> list[_Term]:
    out: list[_Term] = []
    wedge_forms = ((data.creator,) if _nonzero(data.creator) else ()) + tuple(shifts)
    for term in states:
        for form in wedge_forms:
            out.append(_Term(term.coefficient, (form,) + term.factors))
        if not data.annihilator:
            continue
        prefix_degree = 0
        for position, form in enumerate(term.factors):
            if isinstance(form, _Placeholder):
                contracted = (
                    ()
                    if form.degree == 0
                    else (
                        _Placeholder(
                            form.degree - 1,
                            form.contractions + (data.annihilator,),
                        ),
                    )
                )
            else:
                value = interior(data.annihilator, form)
                contracted = (value,) if _nonzero(value) else ()
            for result in contracted:
                out.append(
                    _Term(
                        term.coefficient * ((-1) ** prefix_degree),
                        term.factors[:position] + (result,) + term.factors[position + 1 :],
                    )
                )
            prefix_degree += form.degrees[0]
    return out


def _compile_response_plan(
    problem: ComponentEOMProblem,
    amplitudes: ExteriorForm,
    vacuum_interaction: ExteriorForm,
) -> tuple[_ResponsePlanTerm, ...]:
    grouped: dict[
        tuple[int, tuple[tuple[tuple[int, bytes], ...], ...], int],
        tuple[
            tuple[Mapping[int, ComplexArray], ...],
            dict[BlockKey, ComplexArray],
        ],
    ] = {}
    uncompressed: list[_ResponsePlanTerm] = []

    def accumulate(state: _Term, input_degree: int) -> None:
        placeholder_positions = [
            position
            for position, form in enumerate(state.factors)
            if isinstance(form, _Placeholder)
        ]
        if len(placeholder_positions) != 1:
            raise RuntimeError("a response plan term lost its unique input placeholder")
        position = placeholder_positions[0]
        current = state.factors[position]
        assert isinstance(current, _Placeholder)
        degree_after = sum(
            form.degrees[0] for form in state.factors[position + 1 :]
        )
        coefficient = state.coefficient * (
            (-1) ** (current.degree * degree_after)
        )
        creation = tuple(
            form for form in state.factors if not isinstance(form, _Placeholder)
        )
        creation_degree = sum(form.degrees[0] for form in creation)
        final_degree = creation_degree + current.degree
        if final_degree not in (1, 3, 5):
            return
        term = _ResponsePlanTerm(
            input_degree=input_degree,
            coefficient=coefficient,
            creation_factors=creation,
            contractions=current.contractions,
        )
        creation_degree = sum(form.degrees[0] for form in term.creation_factors)
        # A rank-four creation factor can only contribute to the projected S5
        # shell.  Keeping it lazy prevents one padded n^4 array per distinct
        # contraction signature from becoming persistent response-plan state.
        if creation_degree >= 4:
            uncompressed.append(term)
            return
        signature = tuple(
            tuple(
                (group, np.ascontiguousarray(vector).tobytes())
                for group, vector in sorted(annihilator.items())
            )
            for annihilator in term.contractions
        )
        key = (term.input_degree, signature, creation_degree)
        scalar = ExteriorForm(
            amplitudes.n,
            {_zero_key(): np.asarray(term.coefficient, dtype=np.complex128)},
        )
        creation = scalar
        for factor in term.creation_factors:
            creation = wedge(creation, factor, maximum_degree=creation_degree)
        if key not in grouped:
            blocks: dict[BlockKey, ComplexArray] = {}
            _accumulate_form(blocks, creation, amplitudes.n)
            grouped[key] = (term.contractions, blocks)
        else:
            _accumulate_form(grouped[key][1], creation, amplitudes.n)

    for input_degree in (1, 3):
        placeholder = _Placeholder(input_degree)
        for record in problem.static.oracle.products:
            states: list[_Term] = [_Term(record.coefficient, (placeholder,))]
            for factor in reversed(record.factors):
                data = problem.static.linear[factor.components]
                shifts = _split_homogeneous(
                    interior(data.annihilator, amplitudes)
                )
                states = _apply_factor_plan(states, data, shifts)
            for state in states:
                accumulate(state, input_degree)
        for key, value in vacuum_interaction.blocks.items():
            accumulate(
                _Term(
                    -1.0 + 0.0j,
                    (placeholder, ExteriorForm(amplitudes.n, {key: value})),
                ),
                input_degree,
            )

    compressed: list[_ResponsePlanTerm] = []
    for key, (contractions, blocks) in grouped.items():
        creation = ExteriorForm(amplitudes.n, blocks)
        if not creation.blocks:
            continue
        compressed.append(
            _ResponsePlanTerm(
                input_degree=key[0],
                coefficient=1.0 + 0.0j,
                creation_factors=(creation,),
                contractions=contractions,
            )
        )
    return tuple(compressed + uncompressed)


@dataclass(frozen=True, slots=True)
class ComponentEOMContext:
    problem: ComponentEOMProblem
    amplitudes: ExteriorForm
    vacuum_interaction: ExteriorForm
    response_plan: tuple[_ResponsePlanTerm, ...]


def component_eom_context(problem: ComponentEOMProblem, amplitudes: ArrayLike) -> ComponentEOMContext:
    t = problem.static.coordinates.unpack(amplitudes)
    vacuum_blocks: dict[BlockKey, ComplexArray] = {}
    for record in problem.static.oracle.products:
        primal, _ = _product_terms(record, problem.static.linear, t)
        _accumulate_materialized_terms(
            vacuum_blocks,
            primal,
            t.n,
            {0, 2, 4},
        )
    vacuum = ExteriorForm(t.n, vacuum_blocks)
    return ComponentEOMContext(
        problem,
        t,
        vacuum,
        _compile_response_plan(problem, t, vacuum),
    )


def _iter_response_terms(
    context: ComponentEOMContext,
    response: ExteriorForm,
    output_degrees: set[int] | None = None,
) -> Iterator[_Term]:
    for plan in context.response_plan:
        if plan.input_degree not in response.degrees:
            continue
        final_degree = (
            sum(form.degrees[0] for form in plan.creation_factors)
            + plan.input_degree
            - len(plan.contractions)
        )
        if output_degrees is not None and final_degree not in output_degrees:
            continue
        acted = response.homogeneous(plan.input_degree)
        for annihilator in plan.contractions:
            acted = interior(annihilator, acted)
        if _nonzero(acted):
            yield _Term(plan.coefficient, plan.creation_factors + (acted,))


def _response_terms(context: ComponentEOMContext, response: ExteriorForm) -> list[_Term]:
    """Materialize lazy response terms only when a tiled projection reuses them."""

    return list(_iter_response_terms(context, response))


@dataclass(frozen=True, slots=True)
class ComponentEOMAction:
    retained: ComplexArray
    wall_seconds: float


def component_eom_sigma(context: ComponentEOMContext, vector: ArrayLike) -> ComponentEOMAction:
    """Apply the retained commutator sigma action in one charge block."""

    started = perf_counter()
    x = context.problem.coordinates.unpack(vector)
    interaction = _materialize_terms(
        _iter_response_terms(context, x, {1, 3}),
        x.n,
        {1, 3},
    )
    total = _sum_forms((_quadratic_action(x, context.problem.static.oracle.rapidities), interaction), x.n)
    return ComponentEOMAction(
        retained=context.problem.coordinates.pack(total),
        wall_seconds=float(perf_counter() - started),
    )


def _wedge_block_values(
    left: ComplexArray,
    left_key: BlockKey,
    right: ComplexArray,
    right_key: BlockKey,
    coordinates: IntArray,
) -> ComplexArray:
    """Evaluate one wedge block only at canonical output coordinates."""

    degree = _degree(left_key) + _degree(right_key)
    if coordinates.ndim != 2 or coordinates.shape[1] != degree:
        raise ValueError("projected wedge coordinates have the wrong shape")
    out = np.zeros(coordinates.shape[0], dtype=np.complex128)
    for left_axes, right_axes, sign in _block_shuffles(left_key, right_key):
        left_index = tuple(coordinates[:, axis] for axis in left_axes)
        right_index = tuple(coordinates[:, axis] for axis in right_axes)
        left_value = left[left_index] if left_index else left
        right_value = right[right_index] if right_index else right
        out += sign * left_value * right_value
    return out


def _term_values(term: _Term, target: BlockCoordinates) -> ComplexArray:
    """Project one lazy wedge term without materializing its final block."""

    if term.degree != _degree(target.key):
        return np.zeros(target.vector_positions.size, dtype=np.complex128)
    if not term.factors:
        if target.key != _zero_key():
            return np.zeros(target.vector_positions.size, dtype=np.complex128)
        return np.full(target.vector_positions.size, term.coefficient, dtype=np.complex128)
    prefix = ExteriorForm(
        term.factors[0].n,
        {_zero_key(): np.asarray(term.coefficient, dtype=np.complex128)},
    )
    for factor in term.factors[:-1]:
        prefix = wedge(prefix, factor, maximum_degree=term.degree)
    last = term.factors[-1]
    out = np.zeros(target.vector_positions.size, dtype=np.complex128)
    for left_key, left in prefix.blocks.items():
        for right_key, right in last.blocks.items():
            if _add_key(left_key, right_key) == target.key:
                out += _wedge_block_values(
                    left, left_key, right, right_key, target.indices
                )
    return out


def _project_terms(
    terms: Sequence[_Term],
    coordinates: CoordinateSet,
    *,
    tile_size: int | None = None,
) -> ComplexArray:
    """Project lazy terms, optionally in bounded coordinate tiles."""

    out = np.zeros(coordinates.dimension, dtype=np.complex128)
    step = max(coordinates.dimension, 1) if tile_size is None else int(tile_size)
    if step <= 0:
        raise ValueError("tile_size must be positive")
    for tile in coordinates.tiles(step):
        values = np.zeros(tile.vector_positions.size, dtype=np.complex128)
        for term in terms:
            if term.degree == _degree(tile.key):
                values += _term_values(term, tile)
        out[tile.vector_positions] = values
    return out


def component_rank5_leakage_oracle(
    context: ComponentEOMContext,
    vector: ArrayLike,
    *,
    tile_size: int = 4096,
) -> ComplexArray:
    """Small-system parity helper returning S5 coefficients tile by tile."""

    x = context.problem.coordinates.unpack(vector)
    return _project_terms(
        _response_terms(context, x),
        context.problem.s5_coordinates,
        tile_size=tile_size,
    )


def component_rank5_leakage_diagnostic(
    context: ComponentEOMContext,
    vector: ArrayLike,
    *,
    normalization_scale: float,
    tile_size: int = 4096,
) -> ComponentRankDiagnostic:
    """Accumulate the complete S5 leakage norm without storing S5 values."""

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    scale = float(normalization_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("normalization_scale must be finite and nonnegative")
    started = perf_counter()
    x = context.problem.coordinates.unpack(vector)
    terms = _response_terms(context, x)
    block_squares: dict[BlockKey, float] = {}
    total = 0.0
    for tile in context.problem.s5_coordinates.tiles(tile_size):
        values = np.zeros(tile.vector_positions.size, dtype=np.complex128)
        for term in terms:
            if term.degree == 5:
                values += _term_values(term, tile)
        squared = float(np.vdot(values, values).real)
        block_squares[tile.key] = block_squares.get(tile.key, 0.0) + squared
        total += squared
    block_norms = {
        key: float(np.sqrt(squared)) for key, squared in block_squares.items()
    }
    raw = float(np.sqrt(total))
    normalized = raw / max(scale, np.finfo(np.float64).tiny)
    return ComponentRankDiagnostic(
        raw_norm=raw,
        normalized_norm=normalized,
        normalization_scale=scale,
        block_norms=MappingProxyType(block_norms),
        tile_size=tile_size,
        wall_seconds=float(perf_counter() - started),
    )


def _bare_diagonal_tile(
    rapidities: ComplexArray,
    n: int,
    key: BlockKey,
    coordinates: IntArray,
) -> ComplexArray:
    out = np.zeros(coordinates.shape[0], dtype=np.complex128)
    axis = 0
    for group, count in enumerate(key):
        for _ in range(count):
            out += rapidities[group * n + coordinates[:, axis]]
            axis += 1
    return out


@dataclass(frozen=True, slots=True)
class _IndexedForm:
    """One canonical exterior monomial for every column of a small batch."""

    n: int
    key: BlockKey
    coordinates: IntArray
    values: ComplexArray

    @property
    def degrees(self) -> tuple[int]:
        return (_degree(self.key),)


def _interior_indexed(
    vectors: Mapping[int, ComplexArray],
    form: _IndexedForm,
) -> tuple[_IndexedForm, ...]:
    out: list[_IndexedForm] = []
    offset = 0
    for group, count in enumerate(form.key):
        vector = vectors.get(group)
        if vector is None:
            offset += count
            continue
        for local_axis in range(count):
            axis = offset + local_axis
            key = list(form.key)
            key[group] -= 1
            out.append(
                _IndexedForm(
                    form.n,
                    tuple(key),  # type: ignore[arg-type]
                    np.delete(form.coordinates, axis, axis=1),
                    form.values * ((-1) ** axis) * vector[form.coordinates[:, axis]],
                )
            )
        offset += count
    return tuple(out)


def _apply_factor_indexed(
    states: Sequence[_Term],
    data: _LinearData,
    shifts: Sequence[ExteriorForm],
) -> list[_Term]:
    out: list[_Term] = []
    wedge_forms = ((data.creator,) if _nonzero(data.creator) else ()) + tuple(shifts)
    for term in states:
        for form in wedge_forms:
            out.append(_Term(term.coefficient, (form,) + term.factors))
        if not data.annihilator:
            continue
        prefix_degree = 0
        for position, form in enumerate(term.factors):
            contracted = (
                _interior_indexed(data.annihilator, form)
                if isinstance(form, _IndexedForm)
                else (interior(data.annihilator, form),)
            )
            for result in contracted:
                if isinstance(result, _IndexedForm) or _nonzero(result):
                    out.append(
                        _Term(
                            term.coefficient * ((-1) ** prefix_degree),
                            term.factors[:position] + (result,) + term.factors[position + 1 :],
                        )
                    )
            prefix_degree += form.degrees[0]
    return out


def _product_indexed_terms(
    record: FactorizedModeProduct,
    context: ComponentEOMContext,
    indexed: _IndexedForm,
) -> list[_Term]:
    states: list[_Term] = [_Term(record.coefficient, (indexed,))]
    for factor in reversed(record.factors):
        data = context.problem.static.linear[factor.components]
        shifts = _split_homogeneous(interior(data.annihilator, context.amplitudes))
        states = _apply_factor_indexed(states, data, shifts)
    return states


def _ordered_axis_partitions(keys: Sequence[BlockKey]) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Yield ordered output-axis subsets for several wedge factors."""

    total = tuple(sum(key[group] for key in keys) for group in range(4))
    per_group: list[list[tuple[tuple[int, ...], ...]]] = []
    offset = 0
    for group, count in enumerate(total):
        axes = tuple(range(offset, offset + count))
        sizes = tuple(key[group] for key in keys)

        def split(remaining: tuple[int, ...], position: int) -> Iterator[tuple[tuple[int, ...], ...]]:
            if position == len(sizes) - 1:
                if len(remaining) == sizes[position]:
                    yield (remaining,)
                return
            for chosen in combinations(remaining, sizes[position]):
                chosen_set = set(chosen)
                rest = tuple(axis for axis in remaining if axis not in chosen_set)
                for suffix in split(rest, position + 1):
                    yield (chosen,) + suffix

        per_group.append(list(split(axes, 0)))
        offset += count
    for grouped in product(*per_group):
        by_factor: list[list[int]] = [[] for _ in keys]
        for group_partition in grouped:
            for factor, axes in enumerate(group_partition):
                by_factor[factor].extend(axes)
        yield tuple(tuple(axes) for axes in by_factor)


def _indexed_term_matrix(term: _Term, target: BlockCoordinates) -> ComplexArray:
    indexed_positions = [i for i, form in enumerate(term.factors) if isinstance(form, _IndexedForm)]
    if len(indexed_positions) != 1:
        raise ValueError("an indexed response term must contain exactly one source form")
    indexed = term.factors[indexed_positions[0]]
    assert isinstance(indexed, _IndexedForm)
    out = np.zeros((target.vector_positions.size, indexed.values.size), dtype=np.complex128)
    block_options: list[tuple[tuple[BlockKey, object], ...]] = []
    for form in term.factors:
        if isinstance(form, _IndexedForm):
            block_options.append(((form.key, form),))
        else:
            block_options.append(tuple(form.blocks.items()))
    for selected in product(*block_options):
        keys = tuple(item[0] for item in selected)
        if tuple(sum(key[group] for key in keys) for group in range(4)) != target.key:
            continue
        for axes_by_factor in _ordered_axis_partitions(keys):
            sequence = tuple(axis for axes in axes_by_factor for axis in axes)
            sign = _inversion_sign(sequence)
            dense = np.full(target.vector_positions.size, term.coefficient * sign, dtype=np.complex128)
            match = None
            for (_, value), axes in zip(selected, axes_by_factor, strict=True):
                if isinstance(value, _IndexedForm):
                    source = value.coordinates
                    if axes:
                        candidate = target.indices[:, np.asarray(axes, dtype=np.int64)]
                        current = np.all(candidate[:, None, :] == source[None, :, :], axis=2)
                    else:
                        current = np.ones((target.vector_positions.size, source.shape[0]), dtype=bool)
                    match = current * value.values[None, :]
                else:
                    index = tuple(target.indices[:, axis] for axis in axes)
                    dense *= value[index] if index else value
            if match is None:
                raise RuntimeError("indexed factor disappeared from its term")
            out += dense[:, None] * match
    return out


def _indexed_response_terms(context: ComponentEOMContext, indexed: _IndexedForm) -> list[_Term]:
    terms: list[_Term] = []
    for record in context.problem.static.oracle.products:
        terms.extend(_product_indexed_terms(record, context, indexed))
    for key, value in context.vacuum_interaction.blocks.items():
        terms.append(_Term(-1.0 + 0.0j, (indexed, ExteriorForm(indexed.n, {key: value}))))
    return terms


def _quadratic_indexed_matrix(
    rapidities: ComplexArray,
    source: _IndexedForm,
    target: BlockCoordinates,
) -> ComplexArray:
    if source.key != target.key:
        return np.zeros((target.vector_positions.size, source.values.size), dtype=np.complex128)
    match = np.all(
        target.indices[:, None, :] == source.coordinates[None, :, :], axis=2
    )
    diagonal = _bare_diagonal_tile(rapidities, source.n, source.key, source.coordinates)
    return match * (diagonal * source.values)[None, :]


def component_block_action(
    context: ComponentEOMContext,
    source: CoordinateSet,
    target: CoordinateSet,
    *,
    batch_size: int = 32,
) -> ComplexArray:
    """Construct one explicit coordinate block with batched exterior tensors."""

    if source.n != target.n or source.n != context.problem.coordinates.n:
        raise ValueError("action coordinates use inconsistent dimensions")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    p = context.problem.coordinates.dimension
    s5 = context.problem.s5_coordinates.dimension
    if (
        (source.dimension == p and target.dimension == s5)
        or (source.dimension == s5 and target.dimension == p)
        or (source.dimension == s5 and target.dimension == s5)
    ):
        raise MemoryError(
            "complete P-S5 and S5-S5 matrices are forbidden; select S5 rows first"
        )
    if not isinstance(source, ComponentCoordinates) or not isinstance(
        target, ComponentCoordinates
    ):
        raise TypeError("block actions require explicit component coordinates")
    out = np.zeros((target.dimension, source.dimension), dtype=np.complex128)
    rapidities = context.problem.static.oracle.rapidities
    for source_block in source.blocks:
        for start in range(0, source_block.vector_positions.size, batch_size):
            stop = min(start + batch_size, source_block.vector_positions.size)
            columns = source_block.vector_positions[start:stop]
            indexed = _IndexedForm(
                source.n,
                source_block.key,
                source_block.indices[start:stop],
                np.ones(stop - start, dtype=np.complex128),
            )
            terms = _indexed_response_terms(context, indexed)
            for target_block in target.blocks:
                if _degree(target_block.key) not in (1, 3, 5, 7, 9):
                    continue
                rows = target_block.vector_positions
                block = _quadratic_indexed_matrix(rapidities, indexed, target_block)
                for term in terms:
                    if term.degree == _degree(target_block.key):
                        block += _indexed_term_matrix(term, target_block)
                out[np.ix_(rows, columns)] = block
    return out


def _sample_charge_shell(
    n: int,
    charge: int,
    creator_rank: int,
    sample_count: int,
    seed: int,
    *,
    response_spin: int = 0,
    symmetry: str = "spin",
) -> tuple[ComponentCoordinates, int]:
    keys = _charge_shell_keys(
        n,
        charge,
        creator_rank,
        response_spin=response_spin,
        symmetry=symmetry,
    )
    weights = tuple(prod(comb(n, value) for value in key) for key in keys)
    shell_dimension = sum(weights)
    if sample_count <= 0 or sample_count > shell_dimension:
        raise ValueError("sample_count lies outside the requested charge shell")
    rng = np.random.default_rng(seed)
    probabilities = np.asarray(weights, dtype=np.float64) / float(shell_dimension)
    records: dict[BlockKey, list[tuple[int, tuple[int, ...]]]] = {}
    seen: set[tuple[int, ...]] = set()
    while len(seen) < sample_count:
        key_index = int(rng.choice(len(keys), p=probabilities))
        key = keys[key_index]
        indices = tuple(
            int(value)
            for count in key
            for value in np.sort(rng.choice(n, size=count, replace=False))
        )
        identity = (key_index,) + indices
        if identity in seen:
            continue
        seen.add(identity)
        position = len(seen) - 1
        records.setdefault(key, []).append((position, indices))
    blocks = tuple(
        BlockCoordinates(
            key,
            np.asarray([item[0] for item in values], dtype=np.int64),
            np.asarray([item[1] for item in values], dtype=np.int64),
        )
        for key, values in sorted(records.items())
    )
    return ComponentCoordinates(n, sample_count, blocks), shell_dimension


def component_sampled_static_rank6_diagnostic(
    problem: ComponentStaticProblem,
    amplitudes: ArrayLike,
    *,
    normalization_scale: float,
    sample_count: int = 32768,
    seed: int = 730613,
) -> ComponentSampledRankDiagnostic:
    """Uniformly sample the complete neutral rank-six residual norm."""

    scale = float(normalization_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("normalization_scale must be finite and nonnegative")
    started = perf_counter()
    t = problem.coordinates.unpack(amplitudes)
    terms: list[_Term] = []
    for record in problem.oracle.products:
        primal, _ = _product_terms(record, problem.linear, t)
        terms.extend(primal)
    sample, shell_dimension = _sample_charge_shell(
        t.n,
        0,
        6,
        sample_count,
        seed,
        symmetry=problem.oracle.model.symmetry,
    )
    values = _project_terms(terms, sample)
    squared = np.abs(values) ** 2
    mean = float(np.mean(squared))
    raw = float(np.sqrt(max(0.0, float(shell_dimension) * mean)))
    if sample_count > 1 and mean > 0.0:
        mean_standard_error = float(np.std(squared, ddof=1) / np.sqrt(sample_count))
        norm_standard_error = (
            0.5 * float(shell_dimension) * mean_standard_error / raw
        )
    else:
        norm_standard_error = 0.0
    return ComponentSampledRankDiagnostic(
        creator_rank=6,
        shell_dimension=shell_dimension,
        sample_count=sample_count,
        estimated_raw_norm=raw,
        estimated_normalized_norm=raw
        / max(scale, np.finfo(np.float64).tiny),
        estimated_standard_error=float(norm_standard_error),
        seed=seed,
        wall_seconds=float(perf_counter() - started),
    )


@dataclass(frozen=True, slots=True)
class ComponentEOMSources:
    addition: ComplexArray
    removal: ComplexArray
    sink: ComplexArray
    charge: int
    site: int


def _component_source(
    context: ComponentEOMContext,
    operator: LinearModeOperator,
) -> ComplexArray:
    data = _linear_data(operator, context.problem.coordinates.n)
    source = _sum_forms(
        (data.creator,) + _split_homogeneous(interior(data.annihilator, context.amplitudes)),
        context.problem.coordinates.n,
    )
    return context.problem.coordinates.pack(source)


def _component_sink(
    problem: ComponentEOMProblem,
    operator: LinearModeOperator,
) -> ComplexArray:
    data = _linear_data(operator, problem.coordinates.n)
    out = np.zeros(problem.coordinates.dimension, dtype=np.complex128)
    for block in problem.coordinates.blocks:
        if _degree(block.key) != 1:
            continue
        group = next(group for group, count in enumerate(block.key) if count)
        vector = data.annihilator.get(group)
        if vector is not None:
            out[block.vector_positions] = vector[block.indices[:, 0]]
    return out


def component_eom_sources(
    context: ComponentEOMContext,
    *,
    site: int | None = None,
) -> ComponentEOMSources:
    """Return the two physical sources and trace sink for one charge block."""

    from c3l.ci.matrix_free import physical_mode_operator

    cc = context.problem.static.oracle
    target = cc.model.impurity_site if site is None else site
    if (
        isinstance(target, (bool, np.bool_))
        or not isinstance(target, (int, np.integer))
        or not 0 <= int(target) < cc.model.n_sites
    ):
        raise ValueError("site is outside the auxiliary model")
    target = int(target)
    spin = context.problem.oracle.response_spin
    reference = cc.reference
    if context.problem.charge == 1:
        addition = _component_source(
            context, physical_mode_operator(reference, target, spin, "c_dagger")
        )
        removal = -1j * _component_source(
            context, physical_mode_operator(reference, target, spin, "tilde_c")
        )
        sink_operator = physical_mode_operator(reference, target, spin, "c")
    else:
        addition = -1j * _component_source(
            context, physical_mode_operator(reference, target, spin, "tilde_c_dagger")
        )
        removal = _component_source(
            context, physical_mode_operator(reference, target, spin, "c")
        )
        sink_operator = physical_mode_operator(reference, target, spin, "c_dagger")
    return ComponentEOMSources(
        addition=addition,
        removal=removal,
        sink=_component_sink(context.problem, sink_operator),
        charge=context.problem.charge,
        site=target,
    )


__all__ = [
    "BlockCoordinates",
    "ComponentCoordinateShell",
    "ComponentCoordinates",
    "ComponentEOMAction",
    "ComponentEOMContext",
    "ComponentEOMProblem",
    "ComponentEOMSources",
    "ComponentProductDiagnostics",
    "ComponentRawMaterializationDiagnostics",
    "ComponentStaticAction",
    "ComponentStaticLinearization",
    "ComponentStaticProblem",
    "ComponentRankDiagnostic",
    "ComponentSampledRankDiagnostic",
    "ExteriorForm",
    "component_eom_context",
    "component_eom_sources",
    "component_eom_sigma",
    "component_rank5_leakage_oracle",
    "component_rank5_leakage_diagnostic",
    "component_sampled_static_rank6_diagnostic",
    "component_block_action",
    "component_static_jvp",
    "component_static_linearization",
    "component_static_linearized_jvp",
    "component_static_linearized_jvp_parallel",
    "component_static_rank6_diagnostic",
    "component_static_residual",
    "interior",
    "wedge",
]

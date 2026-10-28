"""Production complex128 JAX action on the complete S1 + S3 + S5 space.

The CPU implementation in s135_full supplies the shared contraction plans;
this module evaluates them on JAX devices. Both implement the same full-S135
projection. Importing this module does not initialize a JAX device.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from time import perf_counter
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from . import component_tensor as _ct
from .s135_full import _contraction_signature


BlockKey = tuple[int, int, int, int]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class JaxRuntime:
    """Observed JAX runtime properties needed by the production gate."""

    jax_version: str
    jaxlib_version: str
    enable_x64: bool
    platform: str
    device_kind: str


def require_jax_runtime(runtime: JaxRuntime, *, production: bool, platform: str = "gpu") -> None:
    """Require double precision and the explicitly requested production device."""

    if not runtime.enable_x64:
        raise RuntimeError("JAX_ENABLE_X64 must be true for complex128 response")
    if production:
        requested = "gpu" if platform == "cuda" else platform
        observed = "gpu" if runtime.platform == "cuda" else runtime.platform
        if requested not in {"cpu", "gpu"} or observed != requested:
            raise RuntimeError(f"production JAX action requires the requested {requested.upper()} device; observed {observed}")


def probe_jax_runtime(*, production: bool = True, platform: str | None = None) -> JaxRuntime:
    """Inspect the default or requested device and enforce the production gate."""

    import jax
    import jaxlib

    devices = jax.devices(platform)
    if not devices:
        raise RuntimeError("JAX reported no execution devices")
    device = devices[0]
    runtime = JaxRuntime(
        jax_version=str(jax.__version__),
        jaxlib_version=str(jaxlib.__version__),
        enable_x64=bool(jax.config.jax_enable_x64),
        platform=str(device.platform),
        device_kind=str(device.device_kind),
    )
    require_jax_runtime(runtime, production=production, platform=platform or "gpu")
    return runtime


@dataclass(frozen=True, slots=True)
class CoordinateBlockPlan:
    """One canonical source/output block and its static antisymmetry pattern."""

    key: BlockKey
    vector_positions: IntArray
    indices: IntArray
    permutations: tuple[tuple[tuple[int, ...], int], ...]


@dataclass(frozen=True, slots=True)
class ContractionStepPlan:
    """One exterior interior-product step, ordered by component group."""

    vectors: tuple[tuple[int, ComplexArray], ...]


@dataclass(frozen=True, slots=True)
class PrefixPlan:
    """One unique input degree and contraction-prefix sequence."""

    input_degree: int
    contractions: tuple[ContractionStepPlan, ...]


@dataclass(frozen=True, slots=True)
class BatchedContractionStepPlan:
    """Equal-support contraction vectors for several unique prefixes."""

    vectors: tuple[tuple[int, tuple[ComplexArray, ...]], ...]


@dataclass(frozen=True, slots=True)
class PrefixBucketPlan:
    """Unique prefixes sharing input rank and contraction support."""

    input_degree: int
    prefix_indices: IntArray
    contractions: tuple[BatchedContractionStepPlan, ...]
    output_keys: tuple[BlockKey, ...]


@dataclass(frozen=True, slots=True)
class ProjectionTerm:
    """One materialized creation block coupled to a cached acted prefix."""

    prefix_index: int
    creation_key: BlockKey
    creation: ComplexArray


@dataclass(frozen=True, slots=True)
class ProjectionBucket:
    """Creation terms sharing the same response degrees and tensor shape."""

    output_degree: int
    creation_degree: int
    acted_degree: int
    shape: tuple[int, ...]
    terms: tuple[ProjectionTerm, ...]


@dataclass(frozen=True, slots=True)
class ActedBankPlan:
    """Concatenation order for one acted exterior block key."""

    key: BlockKey
    source_buckets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProjectionKernelPlan:
    """Projection terms sharing one target/creation/acted key triple."""

    target_block_index: int
    target_key: BlockKey
    creation_key: BlockKey
    acted_key: BlockKey
    terms: tuple[ProjectionTerm, ...]
    acted_rows: IntArray


@dataclass(frozen=True, slots=True)
class FullS135JaxPlan:
    """Immutable, backend-independent lowering of one exact full-S135 action."""

    n: int
    p_dimension: int
    s5_dimension: int
    coordinates: tuple[CoordinateBlockPlan, ...]
    prefixes: tuple[PrefixPlan, ...]
    projections: tuple[ProjectionBucket, ...]
    prefix_buckets: tuple[PrefixBucketPlan, ...]
    acted_banks: tuple[ActedBankPlan, ...]
    projection_kernels: tuple[ProjectionKernelPlan, ...]
    rapidity_diagonal: ComplexArray
    creation_bytes: int
    projected_creation_bytes: int

    @property
    def dimension(self) -> int:
        return self.p_dimension + self.s5_dimension

    @property
    def coordinate_keys(self) -> tuple[BlockKey, ...]:
        return tuple(block.key for block in self.coordinates)

    @property
    def acted_degrees(self) -> tuple[int, ...]:
        return tuple(sorted({sum(bank.key) for bank in self.acted_banks}))


@dataclass(frozen=True, slots=True)
class DeviceActionBlocks:
    """Four logical blocks of one split full-S135 right action."""

    APP: ComplexArray
    AP5: ComplexArray
    A5P: ComplexArray
    A55: ComplexArray


@dataclass(frozen=True, slots=True)
class JaxActionProfile:
    """Synchronized cold, warm, transfer, and end-to-end action timings."""

    plan_build_seconds: float
    device_upload_seconds: float
    first_device_action_seconds: float
    warm_kernel_seconds: tuple[float, ...]
    h2d_seconds: tuple[float, ...]
    d2h_seconds: tuple[float, ...]
    end_to_end_seconds: tuple[float, ...]
    device_memory: dict[str, int]


def _readonly(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    value = np.asarray(array, dtype=dtype)
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    view = value.view()
    view.flags.writeable = False
    return view


def _inversion_sign(sequence: tuple[int, ...]) -> int:
    inversions = sum(
        sequence[left] > sequence[right]
        for left in range(len(sequence))
        for right in range(left + 1, len(sequence))
    )
    return -1 if inversions % 2 else 1


def _group_permutations(key: BlockKey) -> tuple[tuple[tuple[int, ...], int], ...]:
    offsets = np.cumsum((0,) + key)
    grouped = tuple(tuple(permutations(range(count))) for count in key)
    out: list[tuple[tuple[int, ...], int]] = []
    for selected in product(*grouped):
        columns: list[int] = []
        sign = 1
        for group, permutation in enumerate(selected):
            start = int(offsets[group])
            columns.extend(start + position for position in permutation)
            sign *= _inversion_sign(tuple(permutation))
        out.append((tuple(columns), sign))
    return tuple(out)


def _unpack_block(jnp, values, block: CoordinateBlockPlan, n: int):
    """Expand one canonical vector block into its padded exterior tensor."""

    out = jnp.zeros((n,) * sum(block.key), dtype=jnp.complex128)
    canonical = values[block.vector_positions]
    for columns, sign in block.permutations:
        if columns:
            target = tuple(block.indices[:, column] for column in columns)
            out = out.at[target].set(sign * canonical)
        else:
            out = out.at[...].set(canonical[0])
    return out


def _coordinate_gather_maps(
    block: CoordinateBlockPlan, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the dense value lookup and sign used by one exterior block."""

    shape = (int(n),) * sum(block.key)
    lookup = np.zeros(shape, dtype=np.int32)
    signs = np.zeros(shape, dtype=np.int8)
    positions = np.asarray(block.vector_positions, dtype=np.int64)
    indices = np.asarray(block.indices, dtype=np.int64)
    if positions.size and int(np.max(positions)) > np.iinfo(np.int32).max:
        raise OverflowError("response vector positions exceed the JAX int32 lookup")
    for columns, sign in block.permutations:
        target = tuple(indices[:, column] for column in columns)
        lookup[target] = positions.astype(np.int32, copy=False)
        signs[target] = np.int8(sign)
    return (
        _readonly(lookup, np.dtype(np.int32)),
        _readonly(signs, np.dtype(np.int8)),
    )


def _unpack_from_gather(jnp, values, lookup, signs):
    """Expand one coordinate block without a general device scatter."""

    return values[lookup] * signs


def _assembly_gather_lookup(
    positions: tuple[np.ndarray, ...], dimension: int
) -> np.ndarray:
    """Map global vector order to concatenated projected-block order."""

    lookup = np.empty(int(dimension), dtype=np.int32)
    seen = np.zeros(int(dimension), dtype=bool)
    offset = 0
    for raw in positions:
        selected = np.asarray(raw, dtype=np.int64)
        if np.any(selected < 0) or np.any(selected >= dimension):
            raise ValueError("coordinate assembly position lies outside the vector")
        if np.any(seen[selected]):
            raise ValueError("coordinate assembly positions overlap")
        lookup[selected] = offset + np.arange(selected.size, dtype=np.int32)
        seen[selected] = True
        offset += selected.size
    if offset != dimension or not np.all(seen):
        raise ValueError("coordinate assembly positions do not cover the vector")
    return _readonly(lookup, np.dtype(np.int32))


def _assemble_from_gather(jnp, blocks, lookup, value, diagonal):
    """Restore vector order and add the exact bare diagonal without scatters."""

    return jnp.concatenate(blocks, axis=0)[lookup] + diagonal * value


def _interior_blocks(jnp, vectors: Mapping[int, object], blocks: Mapping[BlockKey, object]):
    """Apply one exact bilinear interior product to device exterior blocks."""

    out: dict[BlockKey, object] = {}
    for key, value in blocks.items():
        for group, vector in vectors.items():
            if key[group] == 0:
                continue
            axis = sum(key[:group])
            target = list(key)
            target[group] -= 1
            target_key = tuple(target)
            contracted = ((-1) ** axis) * jnp.tensordot(
                vector, value, axes=(0, axis)
            )
            if target_key in out:
                out[target_key] = out[target_key] + contracted
            else:
                out[target_key] = contracted
    return out


def _wedge_values(
    jnp,
    left,
    left_key: BlockKey,
    right,
    right_key: BlockKey,
    coordinates,
):
    """Evaluate one exterior wedge on canonical target coordinates."""

    degree = sum(left_key) + sum(right_key)
    if coordinates.ndim != 2 or coordinates.shape[1] != degree:
        raise ValueError("projected JAX wedge coordinates have the wrong shape")
    out = jnp.zeros(coordinates.shape[0], dtype=jnp.complex128)
    for left_axes, right_axes, sign in _ct._block_shuffles(left_key, right_key):
        left_index = tuple(coordinates[:, axis] for axis in left_axes)
        right_index = tuple(coordinates[:, axis] for axis in right_axes)
        left_value = left[left_index] if left_index else left
        right_value = right[right_index] if right_index else right
        out = out + sign * left_value * right_value
    return out


def _batched_wedge_values(
    jnp,
    creation,
    creation_key: BlockKey,
    acted_bank,
    acted_rows,
    acted_key: BlockKey,
    coordinates,
):
    """Evaluate and sum equal-structure creation/acted contributions."""

    degree = sum(creation_key) + sum(acted_key)
    if coordinates.ndim != 2 or coordinates.shape[1] != degree:
        raise ValueError("batched JAX wedge coordinates have the wrong shape")
    count = creation.shape[0]
    if acted_rows.shape != (count,):
        raise ValueError("batched JAX wedge acted rows have the wrong shape")
    selected = acted_bank[acted_rows]
    batch = jnp.arange(count)[:, None]
    out = jnp.zeros(coordinates.shape[0], dtype=jnp.complex128)
    for left_axes, right_axes, sign in _ct._block_shuffles(
        creation_key, acted_key
    ):
        if left_axes:
            left_index = (batch,) + tuple(
                coordinates[None, :, axis] for axis in left_axes
            )
            left_value = creation[left_index]
        else:
            left_value = creation[:, None]
        if right_axes:
            right_index = (batch,) + tuple(
                coordinates[None, :, axis] for axis in right_axes
            )
            right_value = selected[right_index]
        else:
            right_value = selected[:, None]
        out = out + sign * jnp.sum(left_value * right_value, axis=0)
    return out


def _batched_interior_blocks(
    jnp,
    vectors: Mapping[int, object],
    blocks: Mapping[BlockKey, object],
):
    """Apply one interior step to exterior blocks with a leading batch axis."""

    out: dict[BlockKey, object] = {}
    for key, value in blocks.items():
        degree = sum(key)
        for group, vector in vectors.items():
            if key[group] == 0:
                continue
            spatial_axis = sum(key[:group])
            axis = spatial_axis + 1
            shape = [vector.shape[0]] + [1] * degree
            shape[axis] = vector.shape[1]
            contracted = ((-1) ** spatial_axis) * jnp.sum(
                value * jnp.reshape(vector, tuple(shape)), axis=axis
            )
            target = list(key)
            target[group] -= 1
            target_key = tuple(target)
            if target_key in out:
                out[target_key] = out[target_key] + contracted
            else:
                out[target_key] = contracted
    return out


def _lower_coordinates(coordinates: object) -> tuple[CoordinateBlockPlan, ...]:
    blocks: list[CoordinateBlockPlan] = []
    for block in coordinates.blocks:
        key = tuple(int(value) for value in block.key)
        if len(key) != 4 or any(value < 0 for value in key):
            raise ValueError("full-S135 coordinate block has an invalid key")
        positions = _readonly(block.vector_positions, np.dtype(np.int64))
        indices = _readonly(block.indices, np.dtype(np.int64))
        if indices.shape != (positions.size, sum(key)):
            raise ValueError("full-S135 coordinate block has inconsistent arrays")
        blocks.append(
            CoordinateBlockPlan(
                key=key,
                vector_positions=positions,
                indices=indices,
                permutations=_group_permutations(key),
            )
        )
    return tuple(blocks)


def _lower_contractions(
    contractions: tuple[Mapping[int, ComplexArray], ...],
    n: int,
) -> tuple[ContractionStepPlan, ...]:
    steps: list[ContractionStepPlan] = []
    for contraction in contractions:
        vectors: list[tuple[int, ComplexArray]] = []
        for group, raw in sorted(contraction.items()):
            if int(group) not in range(4):
                raise ValueError("full-S135 contraction group lies outside [0,4)")
            vector = _readonly(raw, np.dtype(np.complex128))
            if vector.shape != (n,) or not np.all(np.isfinite(vector)):
                raise ValueError("full-S135 contraction vector is invalid")
            vectors.append((int(group), vector))
        steps.append(ContractionStepPlan(tuple(vectors)))
    return tuple(steps)


def _lower_prefixes(
    action: object,
    n: int,
) -> tuple[tuple[PrefixPlan, ...], dict[tuple[int, tuple], int]]:
    prefixes: list[PrefixPlan] = []
    indices: dict[tuple[int, tuple], int] = {}
    for group in action.groups:
        key = (int(group.input_degree), _contraction_signature(group.contractions))
        if key in indices:
            continue
        indices[key] = len(prefixes)
        prefixes.append(
            PrefixPlan(
                input_degree=key[0],
                contractions=_lower_contractions(group.contractions, n),
            )
        )
    return tuple(prefixes), indices


def _lower_projections(
    action: object,
    prefix_indices: dict[tuple[int, tuple], int],
) -> tuple[ProjectionBucket, ...]:
    grouped: dict[
        tuple[int, int, int, tuple[int, ...]], list[ProjectionTerm]
    ] = {}
    for group in action.groups:
        prefix_key = (
            int(group.input_degree),
            _contraction_signature(group.contractions),
        )
        prefix_index = prefix_indices[prefix_key]
        for raw_key, raw_creation in sorted(group.creation.blocks.items()):
            creation_key = tuple(int(value) for value in raw_key)
            creation = _readonly(raw_creation, np.dtype(np.complex128))
            if not np.all(np.isfinite(creation)):
                raise ValueError("full-S135 creation tensor is nonfinite")
            bucket_key = (
                int(group.output_degree),
                int(group.creation_degree),
                int(group.acted_degree),
                creation.shape,
            )
            grouped.setdefault(bucket_key, []).append(
                ProjectionTerm(prefix_index, creation_key, creation)
            )
    return tuple(
        ProjectionBucket(*key, tuple(terms))
        for key, terms in sorted(grouped.items())
    )


def _prefix_output_keys(
    prefix: PrefixPlan,
    coordinate_keys: tuple[BlockKey, ...],
) -> tuple[BlockKey, ...]:
    keys = {
        key for key in coordinate_keys if sum(key) == prefix.input_degree
    }
    for step in prefix.contractions:
        next_keys: set[BlockKey] = set()
        for key in keys:
            for group, _ in step.vectors:
                if key[group] == 0:
                    continue
                target = list(key)
                target[group] -= 1
                next_keys.add(tuple(target))
        keys = next_keys
    return tuple(sorted(keys))


def _lower_prefix_buckets(
    prefixes: tuple[PrefixPlan, ...],
    coordinate_keys: tuple[BlockKey, ...],
) -> tuple[tuple[PrefixBucketPlan, ...], tuple[tuple[BlockKey, ...], ...]]:
    outputs = tuple(
        _prefix_output_keys(prefix, coordinate_keys) for prefix in prefixes
    )
    grouped: dict[tuple[int, tuple[tuple[int, ...], ...]], list[int]] = {}
    for index, prefix in enumerate(prefixes):
        pattern = (
            prefix.input_degree,
            tuple(tuple(group for group, _ in step.vectors) for step in prefix.contractions),
        )
        grouped.setdefault(pattern, []).append(index)

    buckets: list[PrefixBucketPlan] = []
    for pattern, raw_indices in grouped.items():
        indices = tuple(raw_indices)
        expected = outputs[indices[0]]
        if any(outputs[index] != expected for index in indices[1:]):
            raise RuntimeError("equal-support prefixes produced different block keys")
        contractions: list[BatchedContractionStepPlan] = []
        for step_index, groups in enumerate(pattern[1]):
            vectors: list[tuple[int, tuple[ComplexArray, ...]]] = []
            for group in groups:
                selected: list[ComplexArray] = []
                for prefix_index in indices:
                    step = dict(prefixes[prefix_index].contractions[step_index].vectors)
                    selected.append(step[group])
                vectors.append((group, tuple(selected)))
            contractions.append(BatchedContractionStepPlan(tuple(vectors)))
        buckets.append(
            PrefixBucketPlan(
                input_degree=pattern[0],
                prefix_indices=_readonly(
                    np.asarray(indices, dtype=np.int64), np.dtype(np.int64)
                ),
                contractions=tuple(contractions),
                output_keys=expected,
            )
        )
    return tuple(buckets), outputs


def _lower_acted_banks(
    prefix_buckets: tuple[PrefixBucketPlan, ...],
) -> tuple[
    tuple[ActedBankPlan, ...],
    dict[tuple[int, BlockKey], int],
]:
    keys = sorted(
        {key for bucket in prefix_buckets for key in bucket.output_keys}
    )
    banks: list[ActedBankPlan] = []
    rows: dict[tuple[int, BlockKey], int] = {}
    for key in keys:
        source_buckets: list[int] = []
        row = 0
        for bucket_index, bucket in enumerate(prefix_buckets):
            if key not in bucket.output_keys:
                continue
            source_buckets.append(bucket_index)
            for prefix_index in bucket.prefix_indices:
                rows[(int(prefix_index), key)] = row
                row += 1
        banks.append(ActedBankPlan(key, tuple(source_buckets)))
    return tuple(banks), rows


def _lower_projection_kernels(
    coordinates: tuple[CoordinateBlockPlan, ...],
    projections: tuple[ProjectionBucket, ...],
    prefix_outputs: tuple[tuple[BlockKey, ...], ...],
    acted_rows: dict[tuple[int, BlockKey], int],
) -> tuple[ProjectionKernelPlan, ...]:
    targets = {block.key: index for index, block in enumerate(coordinates)}
    grouped: dict[
        tuple[int, BlockKey, BlockKey, BlockKey],
        list[tuple[ProjectionTerm, int]],
    ] = {}
    for bucket in projections:
        for term in bucket.terms:
            for acted_key in prefix_outputs[term.prefix_index]:
                target_key = _ct._add_key(term.creation_key, acted_key)
                target_index = targets.get(target_key)
                if target_index is None:
                    continue
                key = (
                    target_index,
                    target_key,
                    term.creation_key,
                    acted_key,
                )
                grouped.setdefault(key, []).append(
                    (term, acted_rows[(term.prefix_index, acted_key)])
                )
    return tuple(
        ProjectionKernelPlan(
            target_block_index=key[0],
            target_key=key[1],
            creation_key=key[2],
            acted_key=key[3],
            terms=tuple(item[0] for item in values),
            acted_rows=_readonly(
                np.asarray([item[1] for item in values], dtype=np.int64),
                np.dtype(np.int64),
            ),
        )
        for key, values in sorted(grouped.items())
    )


def build_full_s135_jax_plan(operator: object) -> FullS135JaxPlan:
    """Lower one exact CPU full-S135 operator into immutable backend data."""

    from c3l.cc.s135_full import FullS135ResponseOperator, rapidity_diagonal

    if not isinstance(operator, FullS135ResponseOperator):
        raise TypeError("GPU plan requires FullS135ResponseOperator")
    n = int(operator.P.n)
    coordinates = _lower_coordinates(operator.coordinates)
    prefixes, prefix_indices = _lower_prefixes(operator.action, n)
    projections = _lower_projections(operator.action, prefix_indices)
    prefix_buckets, prefix_outputs = _lower_prefix_buckets(
        prefixes, tuple(block.key for block in coordinates)
    )
    acted_banks, acted_rows = _lower_acted_banks(prefix_buckets)
    projection_kernels = _lower_projection_kernels(
        coordinates,
        projections,
        prefix_outputs,
        acted_rows,
    )
    diagonal = _readonly(
        rapidity_diagonal(
            operator.action.context.problem.static.oracle.rapidities,
            operator.coordinates,
        ),
        np.dtype(np.complex128),
    )
    return FullS135JaxPlan(
        n=n,
        p_dimension=int(operator.p_dimension),
        s5_dimension=int(operator.s5_dimension),
        coordinates=coordinates,
        prefixes=prefixes,
        projections=projections,
        prefix_buckets=prefix_buckets,
        acted_banks=acted_banks,
        projection_kernels=projection_kernels,
        rapidity_diagonal=diagonal,
        creation_bytes=int(operator.action.storage_bytes),
        projected_creation_bytes=int(
            sum(
                term.creation.nbytes
                for kernel in projection_kernels
                for term in kernel.terms
            )
        ),
    )


class JaxFullS135ResponseOperator:
    """Exact JAX right action with the NumPy interface used by Arnoldi."""

    def __init__(
        self,
        plan: FullS135JaxPlan,
        *,
        jax,
        jnp,
        device,
        runtime: JaxRuntime,
        projection_tile_size: int = 16384,
    ):
        upload_started = perf_counter()
        require_jax_runtime(runtime, production=False)
        if projection_tile_size <= 0:
            raise ValueError("JAX projection tile size must be positive")
        self.plan = plan
        self.jax = jax
        self.jnp = jnp
        self.device = device
        self.runtime = runtime
        self.dimension = plan.dimension
        self.p_dimension = plan.p_dimension
        self.s5_dimension = plan.s5_dimension
        self.projection_tile_size = int(projection_tile_size)

        def put(value):
            result = jax.device_put(value, device)
            result.block_until_ready()
            return result

        self._coordinates = tuple(
            CoordinateBlockPlan(
                block.key,
                put(block.vector_positions),
                put(block.indices),
                block.permutations,
            )
            for block in plan.coordinates
        )
        coordinate_keys = tuple(block.key for block in self._coordinates)
        unpack_arrays = tuple(
            tuple(put(array) for array in _coordinate_gather_maps(block, plan.n))
            for block in plan.coordinates
        )

        def unpack(value, arrays):
            return tuple(
                _unpack_from_gather(jnp, value, pair[0], pair[1])
                for pair in arrays
            )

        self._unpack_arrays = unpack_arrays
        self._unpack_kernel = jax.jit(unpack)

        self._prefix_data = []
        self._prefix_kernels = []
        for bucket in plan.prefix_buckets:
            input_keys = tuple(
                key for key in coordinate_keys if sum(key) == bucket.input_degree
            )
            step_groups = tuple(
                tuple(group for group, _ in step.vectors)
                for step in bucket.contractions
            )
            vector_data = tuple(
                tuple(
                    put(np.stack(vectors, axis=0))
                    for _, vectors in step.vectors
                )
                for step in bucket.contractions
            )
            batch_size = int(bucket.prefix_indices.size)
            output_keys = bucket.output_keys

            def apply_prefix(
                source_blocks,
                step_vectors,
                *,
                input_keys=input_keys,
                step_groups=step_groups,
                output_keys=output_keys,
                batch_size=batch_size,
            ):
                blocks = {
                    key: jnp.broadcast_to(
                        value, (batch_size,) + tuple(value.shape)
                    )
                    for key, value in zip(input_keys, source_blocks, strict=True)
                }
                for groups, vectors in zip(
                    step_groups, step_vectors, strict=True
                ):
                    blocks = _batched_interior_blocks(
                        jnp, dict(zip(groups, vectors, strict=True)), blocks
                    )
                return tuple(blocks[key] for key in output_keys)

            self._prefix_data.append(
                (input_keys, output_keys, vector_data)
            )
            self._prefix_kernels.append(jax.jit(apply_prefix))
        self._prefix_data = tuple(self._prefix_data)
        self._prefix_kernels = tuple(self._prefix_kernels)

        acted_bank_index = {
            bank.key: index for index, bank in enumerate(plan.acted_banks)
        }
        kernels_by_target: list[list[ProjectionKernelPlan]] = [
            [] for _ in self._coordinates
        ]
        for kernel in plan.projection_kernels:
            kernels_by_target[kernel.target_block_index].append(kernel)

        self._projection_data = []
        self._projection_kernels = []
        for target_index, target_kernels in enumerate(kernels_by_target):
            specs = tuple(
                (
                    kernel.creation_key,
                    kernel.acted_key,
                    acted_bank_index[kernel.acted_key],
                )
                for kernel in target_kernels
            )
            creations = tuple(
                put(np.stack([term.creation for term in kernel.terms], axis=0))
                for kernel in target_kernels
            )
            acted_rows = tuple(put(kernel.acted_rows) for kernel in target_kernels)
            target_size = int(self._coordinates[target_index].indices.shape[0])
            tile_size = min(self.projection_tile_size, max(1, target_size))
            tile_count = (target_size + tile_size - 1) // tile_size

            def project_target(
                target_indices,
                creation_arrays,
                row_arrays,
                acted_banks,
                *,
                specs=specs,
                target_size=target_size,
                tile_size=tile_size,
                tile_count=tile_count,
            ):
                if target_size == 0:
                    return jnp.zeros(0, dtype=jnp.complex128)
                padded_size = tile_count * tile_size
                coordinates = jnp.pad(
                    target_indices, ((0, padded_size - target_size), (0, 0)),
                    mode="edge",
                ).reshape(tile_count, tile_size, target_indices.shape[1])
                valid = (jnp.arange(padded_size) < target_size).reshape(tile_count, tile_size)

                def project_tile(carry, tile):
                    tile_coordinates, tile_valid = tile
                    local = jnp.zeros(tile_size, dtype=jnp.complex128)
                    for spec, creation, rows in zip(
                        specs, creation_arrays, row_arrays, strict=True
                    ):
                        creation_key, acted_key, bank_index = spec
                        local = local + _batched_wedge_values(
                            jnp,
                            creation,
                            creation_key,
                            acted_banks[bank_index],
                            rows,
                            acted_key,
                            tile_coordinates,
                        )
                    return carry, jnp.where(tile_valid, local, 0.0 + 0.0j)

                _, projected = jax.lax.scan(
                    project_tile, None, (coordinates, valid), unroll=False,
                )
                return projected.reshape(padded_size)[:target_size]

            self._projection_data.append((creations, acted_rows))
            self._projection_kernels.append(jax.jit(project_target))
        self._projection_data = tuple(self._projection_data)
        self._projection_kernels = tuple(self._projection_kernels)

        self._diagonal = put(plan.rapidity_diagonal)
        assembly_lookup = put(
            _assembly_gather_lookup(
                tuple(block.vector_positions for block in plan.coordinates),
                plan.dimension,
            )
        )

        def assemble(blocks, value, diagonal, lookup):
            return _assemble_from_gather(jnp, blocks, lookup, value, diagonal)

        self._assembly_lookup = assembly_lookup
        self._assemble_kernel = jax.jit(assemble)
        self.kernel_counts = {
            "prefix": len(self._prefix_kernels),
            "projection": len(self._projection_kernels),
        }
        self.plan_build_seconds = 0.0
        self.device_upload_seconds = float(perf_counter() - upload_started)

    @classmethod
    def from_cpu(
        cls,
        operator: object,
        *,
        platform: str = "gpu",
        production: bool = True,
    ) -> "JaxFullS135ResponseOperator":
        import jax
        import jax.numpy as jnp
        import jaxlib

        devices = jax.devices(platform)
        if not devices:
            raise RuntimeError(f"JAX reported no {platform} execution device")
        device = devices[0]
        runtime = JaxRuntime(
            jax_version=str(jax.__version__),
            jaxlib_version=str(jaxlib.__version__),
            enable_x64=bool(jax.config.jax_enable_x64),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
        )
        require_jax_runtime(runtime, production=production, platform=platform)
        plan_started = perf_counter()
        plan = build_full_s135_jax_plan(operator)
        plan_seconds = perf_counter() - plan_started
        instance = cls(
            plan,
            jax=jax,
            jnp=jnp,
            device=device,
            runtime=runtime,
            projection_tile_size=int(operator.s5_tile_size),
        )
        instance.plan_build_seconds = float(plan_seconds)
        return instance

    def _device_action(self, value):
        response_values = self._unpack_kernel(value, self._unpack_arrays)
        response = dict(
            zip((block.key for block in self._coordinates), response_values, strict=True)
        )
        prefix_results: list[dict[BlockKey, object]] = []
        for kernel, data in zip(
            self._prefix_kernels, self._prefix_data, strict=True
        ):
            input_keys, output_keys, vectors = data
            values = kernel(tuple(response[key] for key in input_keys), vectors)
            prefix_results.append(dict(zip(output_keys, values, strict=True)))

        acted_banks = tuple(
            self.jnp.concatenate(
                tuple(prefix_results[index][bank.key] for index in bank.source_buckets),
                axis=0,
            )
            for bank in self.plan.acted_banks
        )
        projected = tuple(
            kernel(
                target.indices,
                data[0],
                data[1],
                acted_banks,
            )
            for kernel, data, target in zip(
                self._projection_kernels,
                self._projection_data,
                self._coordinates,
                strict=True,
            )
        )
        return self._assemble_kernel(
            projected, value, self._diagonal, self._assembly_lookup
        )

    def device_matvec(self, vector: object):
        """Return one device-resident exact right action."""

        with self.jax.default_device(self.device):
            value = self.jax.device_put(
                self.jnp.asarray(vector, dtype=self.jnp.complex128), self.device
            )
            if value.shape != (self.dimension,):
                raise ValueError("full-S135 JAX input has the wrong shape")
            return self._device_action(value)

    def matvec(self, vector: object) -> ComplexArray:
        """Apply the exact action with one host input and one host output."""

        value = np.asarray(vector, dtype=np.complex128)
        if value.shape != (self.dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("full-S135 JAX input has the wrong shape or is nonfinite")
        # Arnoldi owns and updates this work vector in place.  JAX's zero-copy
        # NumPy view is intentionally read-only, so cross the API boundary with
        # one explicit writable host copy.
        result = np.array(
            self.device_matvec(value), dtype=np.complex128, copy=True
        )
        if result.shape != value.shape or not np.all(np.isfinite(result)):
            raise RuntimeError("full-S135 JAX action returned an invalid vector")
        return result

    def device_memory_record(self) -> dict[str, int]:
        """Return integer allocator telemetry exposed by this JAX device."""

        raw = self.device.memory_stats()
        if not raw:
            return {}
        return {
            str(key): int(value)
            for key, value in raw.items()
            if isinstance(value, (int, np.integer))
        }

    def profile(
        self,
        vector: object,
        *,
        warm_repeats: int = 5,
    ) -> JaxActionProfile:
        """Synchronize every boundary and separate kernel from transfer cost."""

        if warm_repeats <= 0:
            raise ValueError("warm_repeats must be positive")
        value = np.asarray(vector, dtype=np.complex128)
        if value.shape != (self.dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("full-S135 JAX profile input is invalid")

        device_value = self.jax.device_put(value, self.device)
        device_value.block_until_ready()
        started = perf_counter()
        first = self._device_action(device_value)
        first.block_until_ready()
        first_seconds = perf_counter() - started

        warm: list[float] = []
        for _ in range(warm_repeats):
            started = perf_counter()
            result = self._device_action(device_value)
            result.block_until_ready()
            warm.append(perf_counter() - started)

        h2d: list[float] = []
        for _ in range(warm_repeats):
            started = perf_counter()
            transferred = self.jax.device_put(value, self.device)
            transferred.block_until_ready()
            h2d.append(perf_counter() - started)

        d2h: list[float] = []
        for _ in range(warm_repeats):
            result = self._device_action(device_value)
            result.block_until_ready()
            started = perf_counter()
            host = np.asarray(result, dtype=np.complex128)
            if host.shape != value.shape:
                raise RuntimeError("profiled JAX device-to-host result changed shape")
            d2h.append(perf_counter() - started)

        end_to_end: list[float] = []
        for _ in range(warm_repeats):
            started = perf_counter()
            self.matvec(value)
            end_to_end.append(perf_counter() - started)

        return JaxActionProfile(
            plan_build_seconds=float(self.plan_build_seconds),
            device_upload_seconds=float(self.device_upload_seconds),
            first_device_action_seconds=float(first_seconds),
            warm_kernel_seconds=tuple(float(item) for item in warm),
            h2d_seconds=tuple(float(item) for item in h2d),
            d2h_seconds=tuple(float(item) for item in d2h),
            end_to_end_seconds=tuple(float(item) for item in end_to_end),
            device_memory=self.device_memory_record(),
        )

    def profile_stages(self, vector: object) -> tuple[ComplexArray, dict[str, object]]:
        """Synchronize every action boundary to localize a slow production graph."""

        value = np.asarray(vector, dtype=np.complex128)
        if value.shape != (self.dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("full-S135 JAX stage-profile input is invalid")
        total_started = perf_counter()

        started = perf_counter()
        device_value = self.jax.device_put(value, self.device)
        device_value.block_until_ready()
        h2d_seconds = perf_counter() - started

        started = perf_counter()
        response_values = self._unpack_kernel(
            device_value, self._unpack_arrays
        )
        self.jax.block_until_ready(response_values)
        unpack_seconds = perf_counter() - started
        response = dict(
            zip(
                (block.key for block in self._coordinates),
                response_values,
                strict=True,
            )
        )

        prefix_results: list[dict[BlockKey, object]] = []
        prefix_records: list[dict[str, object]] = []
        for index, (kernel, data, plan_bucket) in enumerate(
            zip(
                self._prefix_kernels,
                self._prefix_data,
                self.plan.prefix_buckets,
                strict=True,
            )
        ):
            input_keys, output_keys, vectors = data
            started = perf_counter()
            values = kernel(tuple(response[key] for key in input_keys), vectors)
            self.jax.block_until_ready(values)
            prefix_records.append(
                {
                    "bucket": index,
                    "input_degree": int(plan_bucket.input_degree),
                    "prefix_count": int(plan_bucket.prefix_indices.size),
                    "output_keys": output_keys,
                    "wall_seconds": float(perf_counter() - started),
                }
            )
            prefix_results.append(dict(zip(output_keys, values, strict=True)))

        started = perf_counter()
        acted_banks = tuple(
            self.jnp.concatenate(
                tuple(
                    prefix_results[index][bank.key]
                    for index in bank.source_buckets
                ),
                axis=0,
            )
            for bank in self.plan.acted_banks
        )
        self.jax.block_until_ready(acted_banks)
        bank_seconds = perf_counter() - started

        projected: list[object] = []
        projection_records: list[dict[str, object]] = []
        for index, (kernel, data, target) in enumerate(
            zip(
                self._projection_kernels,
                self._projection_data,
                self._coordinates,
                strict=True,
            )
        ):
            started = perf_counter()
            result = kernel(target.indices, data[0], data[1], acted_banks)
            result.block_until_ready()
            projection_records.append(
                {
                    "target": index,
                    "target_key": target.key,
                    "target_size": int(target.indices.shape[0]),
                    "wall_seconds": float(perf_counter() - started),
                }
            )
            projected.append(result)

        started = perf_counter()
        assembled = self._assemble_kernel(
            tuple(projected),
            device_value,
            self._diagonal,
            self._assembly_lookup,
        )
        assembled.block_until_ready()
        assembly_seconds = perf_counter() - started

        started = perf_counter()
        host = np.asarray(assembled, dtype=np.complex128)
        d2h_seconds = perf_counter() - started
        return host, {
            "h2d_wall_seconds": float(h2d_seconds),
            "unpack_wall_seconds": float(unpack_seconds),
            "prefix_buckets": prefix_records,
            "acted_bank_wall_seconds": float(bank_seconds),
            "projection_targets": projection_records,
            "assembly_wall_seconds": float(assembly_seconds),
            "d2h_wall_seconds": float(d2h_seconds),
            "total_wall_seconds": float(perf_counter() - total_started),
            "device_memory": self.device_memory_record(),
        }

    def block_action(self, vector: object) -> DeviceActionBlocks:
        """Evaluate four logical blocks separately for parity/profiling."""

        value = np.asarray(vector, dtype=np.complex128)
        if value.shape != (self.dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("full-S135 JAX input has the wrong shape or is nonfinite")
        retained = np.zeros_like(value)
        retained[: self.p_dimension] = value[: self.p_dimension]
        rank_five = np.zeros_like(value)
        rank_five[self.p_dimension :] = value[self.p_dimension :]
        from_p = np.asarray(self.device_matvec(retained), dtype=np.complex128)
        from_five = np.asarray(self.device_matvec(rank_five), dtype=np.complex128)
        return DeviceActionBlocks(
            APP=from_p[: self.p_dimension],
            AP5=from_five[: self.p_dimension],
            A5P=from_p[self.p_dimension :],
            A55=from_five[self.p_dimension :],
        )


__all__ = [
    "CoordinateBlockPlan",
    "DeviceActionBlocks",
    "FullS135JaxPlan",
    "JaxActionProfile",
    "JaxFullS135ResponseOperator",
    "JaxRuntime",
    "PrefixBucketPlan",
    "PrefixPlan",
    "ProjectionBucket",
    "ProjectionKernelPlan",
    "build_full_s135_jax_plan",
    "probe_jax_runtime",
    "require_jax_runtime",
]

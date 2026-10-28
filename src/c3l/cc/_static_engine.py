"""Compiled CCSD residual and its Jacobian-vector product for JAX static solves.

JaxStaticCCSDOperator requests an engine for its fixed array layout. Numerical
Hamiltonian, bath, reference, and amplitude values are passed as array inputs;
the cached JIT functions can be reused when that layout is unchanged. JAX
obtains the exact directional derivative of the residual with jax.jvp.
The NumPy static solver uses its own explicit contraction path.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp

from c3l.cc import component_tensor as tensor
from c3l.cc.s135_jax import _interior_blocks, _unpack_from_gather

class Engine:
    def __init__(self, layout):
        self.layout = layout
        self.trace_counts = dict(residual=0, jvp=0, gmres=0)
        mode_count = layout.mode_count

        def degree(form):
            return sum(next(iter(form)))

        def split(form):
            groups = {}
            for key, value in form.items():
                groups.setdefault(sum(key), {})[key] = value
            return tuple(groups[key] for key in sorted(groups))

        def wedge(left, right, maximum):
            result = {}
            for left_key, left_value in left.items():
                for right_key, right_value in right.items():
                    key = tuple(first+second for first, second in zip(left_key, right_key))
                    rank = sum(key)
                    if rank > maximum or any(count > mode_count for count in key):
                        continue
                    value = jnp.zeros((mode_count,)*rank, dtype=jnp.complex128)
                    for left_axes, right_axes, sign in tensor._block_shuffles(left_key, right_key):
                        value = value + sign*jnp.einsum(left_value, list(left_axes), right_value,
                                                        list(right_axes), list(range(rank)), optimize=False)
                    result[key] = result.get(key, 0.) + value
            return result

        def apply(states, creator, annihilator, shifts):
            result = []
            forms = ((creator,) if creator else ()) + tuple(shifts)
            for coefficient, factors in states:
                for form in forms:
                    result.append((coefficient, (form,)+factors))
                preceding = 0
                for position, form in enumerate(factors):
                    contracted = _interior_blocks(jnp, annihilator, form)
                    if contracted:
                        result.append((coefficient*((-1)**preceding), factors[:position]+(contracted,)+factors[position+1:]))
                    preceding += degree(form)
            return result

        def product_action(signature, coefficient, factor_values, amplitudes, data):
            states = [(coefficient, ())]
            for position, (creator_keys, annihilator_groups) in enumerate(signature):
                creator = {key: factor_values[position, 0, key.index(1)] for key in creator_keys}
                annihilator = {group: factor_values[position, 1, group] for group in annihilator_groups}
                shifts = split(_interior_blocks(jnp, annihilator, amplitudes))
                states = apply(states, creator, annihilator, shifts)
            interaction = {}
            for coefficient, factors in states:
                rank = sum(degree(form) for form in factors)
                if rank not in (0, 2, 4):
                    continue
                current = {(0, 0, 0, 0): jnp.asarray(coefficient, dtype=jnp.complex128)}
                for form in factors:
                    current = wedge(current, form, rank)
                for key, value in current.items():
                    interaction[key] = interaction.get(key, 0.) + value
            projected = []
            for key, size, indices in zip(layout.coordinate_keys, layout.coordinate_sizes, data['indices']):
                if key not in interaction:
                    projected.append(jnp.zeros(size, dtype=jnp.complex128))
                else:
                    index = tuple(indices[:, axis] for axis in range(indices.shape[1]))
                    projected.append(interaction[key][index] if index else jnp.reshape(interaction[key], (1,)))
            return jnp.concatenate(projected)[data['assembly']]

        def residual(values, data):
            self.trace_counts['residual'] += 1
            blocks = tuple(_unpack_from_gather(jnp, values, pair[0], pair[1]) for pair in data['gather'])
            amplitudes = dict(zip(layout.coordinate_keys, blocks))
            result = jnp.zeros(layout.dimension, dtype=jnp.complex128)
            for signature, group in zip(layout.signatures, data['groups']):
                coefficients, factor_indices, active_tiles = group
                def accumulate(index, total):
                    weights = jax.lax.dynamic_index_in_dim(coefficients, index, keepdims=False)
                    indices = jax.lax.dynamic_index_in_dim(factor_indices, index, keepdims=False)
                    factors = data['factor_bank'][indices]
                    contributions = jax.vmap(lambda coefficient, vectors: product_action(
                        signature, coefficient, vectors, amplitudes, data))(weights, factors)
                    return total + jnp.sum(contributions, axis=0)
                result = jax.lax.fori_loop(0, active_tiles, accumulate, result)
            return -1j*result+data['diagonal']*values

        def jvp(values, direction, data):
            self.trace_counts['jvp'] += 1
            return jax.jvp(lambda amplitudes: residual(amplitudes, data), (values,), (direction,))[1]

        self.raw_residual = residual
        self.raw_jvp = jvp
        self.residual = jax.jit(residual)
        self.jvp = jax.jit(jvp)

@lru_cache(maxsize=32)
def get_engine(layout):
    return Engine(layout)


@jax.jit
def unpack(values, maps):
    return maps[2] * values[maps[0]] + maps[3] * values[maps[1]]


@jax.jit
def pack(values, maps):
    return jnp.real(maps[5] * values[maps[4]])

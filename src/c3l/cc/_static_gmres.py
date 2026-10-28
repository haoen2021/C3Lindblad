"""Cached matrix-free GMRES cycles with runtime model and preconditioner arrays."""

from functools import lru_cache

import jax
import jax.numpy as jnp

from ._static_engine import get_engine, pack, unpack


@lru_cache(maxsize=32)
def get_cycle(layout, restart):
    engine = get_engine(layout)
    def gmres_cycle(amplitudes, right_hand_side, initial, data, diagonal):
        engine.trace_counts['gmres'] += 1
        def precondition(values):
            return pack(unpack(values, data['tilde']) / diagonal, data['tilde'])
        def linear_action(values, direction):
            return pack(engine.raw_jvp(values, unpack(direction, data['tilde']), data), data['tilde'])
        preconditioned_residual = precondition(
            right_hand_side - linear_action(amplitudes, initial)
        )
        beta = jnp.linalg.norm(preconditioned_residual)
        tiny = jnp.finfo(jnp.float64).tiny
        basis = jnp.zeros(
            (layout.real_dimension, restart + 1), dtype=jnp.float64
        )
        basis = basis.at[:, 0].set(
            preconditioned_residual / jnp.maximum(beta, tiny)
        )
        hessenberg = jnp.zeros((restart + 1, restart), dtype=jnp.float64)

        def arnoldi_step(index, state):
            vectors, matrix = state
            work = precondition(
                linear_action(amplitudes, vectors[:, index])
            )
            first = vectors.T @ work
            work = work - vectors @ first
            second = vectors.T @ work
            work = work - vectors @ second
            coefficients = first + second
            norm = jnp.linalg.norm(work)
            matrix = matrix.at[:, index].set(coefficients)
            matrix = matrix.at[index + 1, index].set(norm)
            vectors = vectors.at[:, index + 1].set(
                work / jnp.maximum(norm, tiny)
            )
            return vectors, matrix

        basis, hessenberg = jax.lax.fori_loop(
            0,
            restart,
            arnoldi_step,
            (basis, hessenberg),
        )
        rhs = jnp.zeros(restart + 1, dtype=jnp.float64).at[0].set(beta)
        coefficients = jnp.linalg.lstsq(
            hessenberg,
            rhs,
            rcond=None,
        )[0]
        return initial + basis[:, :restart] @ coefficients


    return jax.jit(gmres_cycle)

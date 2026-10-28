"""Response construction and reduced GMRES for paper calculations."""

import numpy as np
from types import SimpleNamespace
from numba import njit
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.component_tensor import (
    ComponentEOMProblem,
    component_eom_context,
    component_eom_sources,
)


def resident_curve(chain, source_norm, grid, kind, *, eta=None):
    """Evaluate the small Galerkin response without moving the device basis."""
    import scipy.linalg
    from scipy.sparse.linalg import expm_multiply

    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or not grid.size or not np.all(np.isfinite(grid)):
        raise ValueError("response grid must be a finite nonempty vector")
    dimension = chain.current
    hessenberg = chain.hessenberg[:dimension, :dimension]
    rhs = np.zeros(dimension, dtype=np.complex128)
    rhs[0] = source_norm
    if kind == "frequency":
        if eta is None or not np.isfinite(eta):
            raise ValueError("frequency response requires finite eta")
        triangular, unitary = scipy.linalg.schur(hessenberg, output="complex")
        rotated_rhs = unitary.conj().T @ rhs
        identity = np.eye(dimension)
        coefficients = np.asarray([
            unitary @ scipy.linalg.solve_triangular(
                (point + 1j * eta) * identity - triangular, rotated_rhs
            )
            for point in grid
        ])
    elif kind == "time":
        generator = -1j * hessenberg
        if grid.size > 1 and np.allclose(
            np.diff(grid), grid[1] - grid[0], atol=1e-14, rtol=1e-14
        ):
            coefficients = expm_multiply(
                generator, rhs, start=float(grid[0]), stop=float(grid[-1]),
                num=grid.size, endpoint=True,
            )
        else:
            coefficients = np.asarray([
                expm_multiply(float(point) * generator, rhs) for point in grid
            ])
        coefficients *= -1j
    else:
        raise ValueError("response kind must be frequency or time")
    return coefficients @ chain.projected[:, :dimension].T


def response_context(oracle, static, amplitudes):
    """Build the physical spin-up addition-charge block and all four sites."""
    eom = eom_ccsd_problem(oracle, response_spin=0, build_next_rank_masks=False)
    problem = ComponentEOMProblem.compile(eom, static, charge=1)
    context = component_eom_context(problem, amplitudes)
    sources = tuple(component_eom_sources(context, site=i) for i in range(4))
    return context, sources


@njit(cache=False)
def reduced_gmres(h, beta, shifts):
    """Givens QR of z*[I;0]-Hbar; no normal equations or large-space matvecs."""
    m = h.shape[1]
    ys = np.empty((len(shifts), m), np.complex128)
    res = np.empty(len(shifts))
    for wi in range(len(shifts)):
        a = -h.copy()
        b = np.zeros(m + 1, np.complex128)
        b[0] = beta
        for j in range(m):
            a[j, j] += shifts[wi]
        for j in range(m):
            x = a[j, j]
            y = a[j + 1, j]
            norm = np.hypot(abs(x), abs(y))
            if norm == 0:
                continue
            phase = x / abs(x) if abs(x) > 0 else 1.0 + 0j
            c = abs(x) / norm
            s = phase * np.conj(y) / norm
            for k in range(j, m):
                u = a[j, k]
                v = a[j + 1, k]
                a[j, k] = c * u + s * v
                a[j + 1, k] = -np.conj(s) * u + c * v
            u = b[j]
            v = b[j + 1]
            b[j] = c * u + s * v
            b[j + 1] = -np.conj(s) * u + c * v
        for j in range(m - 1, -1, -1):
            value = b[j]
            for k in range(j + 1, m):
                value -= a[j, k] * ys[wi, k]
            ys[wi, j] = value / a[j, j]
        res[wi] = abs(b[m]) / beta
    return ys, res


def response_operator(context, sources, platform, response_rank):
    """Use the core full-Q5 action with physical sources in its retained block."""
    if response_rank == "S13":
        from c3l.cc.component_jax import JaxRetainedEOMOperator

        operator = JaxRetainedEOMOperator.from_cpu(
            context, platform=platform, production=True
        )
        return operator, sources
    if response_rank != "S135":
        raise ValueError("Expected S13 or S135")
    from c3l.cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator

    print("Compiling core full-Q5 action", flush=True)
    operator = FullS135ResponseOperator(CompiledFullS135Action(context))
    print(
        f"Full-Q5 action: P={operator.p_dimension}, Q5={operator.s5_dimension}",
        flush=True,
    )
    embedded = []
    for source in sources:
        values = {}
        for name in ("addition", "removal", "sink"):
            values[name] = np.zeros(operator.dimension, dtype=np.complex128)
            values[name][: operator.p_dimension] = getattr(source, name)
        embedded.append(SimpleNamespace(**values))
    from c3l.cc.s135_jax import JaxFullS135ResponseOperator

    operator = JaxFullS135ResponseOperator.from_cpu(
        operator, platform=platform, production=True
    )
    return operator, tuple(embedded)

"""Device-resident two-pass Arnoldi with bounded reconstruction workspace.

The caller supplies a device-native matrix-free action. Only projected arrays
return to the host; this class has no disk storage or streaming fallback.
"""

from functools import lru_cache
import os
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np


def _host_available_memory_bytes():
    """Best-effort Linux RAM budget, bounded by Slurm and cgroup limits."""
    budgets = []
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        budgets.append(int(info["MemAvailable"].split()[0]) * 1024)
    except (OSError, KeyError, ValueError):
        pass
    try:
        status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
        resident = int(status["VmRSS"].split()[0]) * 1024
        megabytes = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
        if not megabytes:
            megabytes = int(os.environ.get("SLURM_MEM_PER_CPU", "0")) * int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        if megabytes:
            budgets.append(max(0, megabytes * 2**20 - resident))
    except (OSError, KeyError, ValueError):
        pass
    try:
        groups = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        groups = []
    for group in groups:
        _, controllers, relative = group.split(":", 2)
        if controllers and "memory" not in controllers.split(","):
            continue
        root = Path("/sys/fs/cgroup/memory" if controllers else "/sys/fs/cgroup")
        folder = root / relative.lstrip("/")
        limit_name, usage_name = ("memory.limit_in_bytes", "memory.usage_in_bytes") if controllers else ("memory.max", "memory.current")
        for location in (folder, *folder.parents):
            if location != root and root not in location.parents:
                break
            try:
                limit = int((location / limit_name).read_text())
                usage = int((location / usage_name).read_text())
                budgets.append(max(0, limit - usage))
            except (OSError, ValueError):
                continue
    return min(budgets) if budgets else None


def check_resident_memory(device, required_bytes, workspace_bytes=8 * 2**30):
    """Reject an oversized resident basis without changing storage or device."""
    memory = device.memory_stats()
    available = None
    if memory is not None and memory.get("bytes_limit", 0):
        available = memory["bytes_limit"] - memory["bytes_in_use"]
    elif getattr(device, "platform", None) == "cpu":
        available = _host_available_memory_bytes()
    if available is not None and required_bytes + workspace_bytes > available:
        raise MemoryError("Resident basis plus workspace exceeds available memory; reduce the Krylov cap or use a device with more memory. No streaming fallback.")
    return available


@lru_cache(None)
def resident_kernels(dimension, rows, panel):
    @jax.jit
    def orthogonalize(basis, work, width):
        coefficients = jnp.zeros(rows, dtype=work.dtype)
        for _ in range(2):
            reference = work

            def project(index, state):
                remainder, correction = state
                block = jax.lax.dynamic_slice(basis, (index * panel, jnp.asarray(0, dtype=index.dtype)), (panel, dimension))
                values = block.conj() @ reference
                remainder = remainder - values @ block
                correction = jax.lax.dynamic_update_slice(correction, values, (index * panel,))
                return remainder, correction

            work, correction = jax.lax.fori_loop(0, (width + panel - 1) // panel, project, (work, jnp.zeros_like(coefficients)))
            coefficients = coefficients + correction
        norm = jnp.linalg.norm(work)
        return work / jnp.where(norm > 0, norm, 1), coefficients, norm

    def append(basis, vector, index):
        return jax.lax.dynamic_update_slice(basis, vector[None, :], (index, jnp.asarray(0, dtype=index.dtype)))

    @jax.jit
    def reconstruct(basis, coefficients, width):
        def accumulate(index, vector):
            block = jax.lax.dynamic_slice(basis, (index * panel, jnp.asarray(0, dtype=index.dtype)), (panel, dimension))
            weights = jax.lax.dynamic_slice(coefficients, (index * panel,), (panel,))
            return vector + weights @ block

        return jax.lax.fori_loop(0, (width + panel - 1) // panel, accumulate, jnp.zeros(dimension, dtype=basis.dtype))

    return orthogonalize, jax.jit(append, donate_argnums=(0,)), reconstruct


class ResidentArnoldi:
    def __init__(self, action, rhs, sinks, cap, panel=32, *, device=None, workspace_bytes=8 * 2**30):
        if not jax.config.jax_enable_x64:
            raise ValueError("Resident Arnoldi requires JAX double precision")
        rhs = np.asarray(rhs, dtype=np.complex128)
        sinks = np.asarray(sinks, dtype=np.complex128)
        if rhs.ndim != 1 or not np.all(np.isfinite(rhs)) or np.linalg.norm(rhs) == 0:
            raise ValueError("rhs must be a finite nonzero vector")
        if sinks.ndim != 2 or sinks.shape[1] != len(rhs) or not np.all(np.isfinite(sinks)):
            raise ValueError("sinks must be a finite matrix with one column per coordinate")
        if not isinstance(cap, int) or not 1 <= cap <= len(rhs) or not isinstance(panel, int) or panel < 1:
            raise ValueError("Invalid Krylov capacity or panel size")
        self.device = jax.devices()[0] if device is None else device
        required = ((cap + panel) // panel) * panel * len(rhs) * 16
        check_resident_memory(self.device, required, workspace_bytes)
        self.action = action
        self.cap = cap
        self.dimension = len(rhs)
        self.rows = ((cap + 1 + panel - 1) // panel) * panel
        self.orthogonalize, self.append, self.combine = resident_kernels(self.dimension, self.rows, panel)
        with jax.default_device(self.device):
            self.basis = jnp.zeros((self.rows, self.dimension), dtype=jnp.complex128)
            self.vector = jax.device_put(rhs / np.linalg.norm(rhs), self.device)
            self.basis = self.append(self.basis, self.vector, jnp.asarray(0, dtype=jnp.int32))
            self.sinks = jax.device_put(sinks, self.device)
        self.hessenberg = np.zeros((cap + 1, cap), dtype=np.complex128)
        self.projected = np.zeros((len(sinks), cap + 1), dtype=np.complex128)
        self.projected[:, 0] = np.asarray(self.sinks @ self.vector)
        self.current = 0
        self.breakdown = False
        self.action_seconds = []
        self.orthogonalization_seconds = []
        self.basis.block_until_ready()

    def advance(self, target):
        if not isinstance(target, int) or not self.current <= target <= self.cap:
            raise ValueError("target must lie between the current dimension and capacity")
        while self.current < min(target, self.cap) and not self.breakdown:
            started = perf_counter()
            work = self.action(self.vector)
            work.block_until_ready()
            self.action_seconds.append(perf_counter() - started)
            started = perf_counter()
            width = jax.device_put(np.int32(self.current + 1), self.device)
            self.vector, coefficients, norm = self.orthogonalize(self.basis, work, width)
            coefficients, norm = jax.device_get((coefficients, norm))
            if not np.isfinite(norm) or not np.all(np.isfinite(coefficients)):
                raise FloatingPointError('Nonfinite Arnoldi step')
            self.hessenberg[:self.current + 1, self.current] = coefficients[:self.current + 1]
            self.hessenberg[self.current + 1, self.current] = norm
            self.current += 1
            self.breakdown = bool(norm <= 1e-14)
            self.basis = self.append(self.basis, self.vector, width)
            self.projected[:, self.current] = np.asarray(self.sinks @ self.vector)
            self.basis.block_until_ready()
            self.orthogonalization_seconds.append(perf_counter() - started)

    def reconstruct(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if coefficients.ndim != 1 or len(coefficients) > self.current or not np.all(np.isfinite(coefficients)):
            raise ValueError("Invalid reduced coordinates")
        padded = np.zeros(self.rows, dtype=np.complex128)
        padded[:len(coefficients)] = coefficients
        return self.combine(
            self.basis, jax.device_put(padded, self.device),
            jax.device_put(np.int32(len(coefficients)), self.device),
        )

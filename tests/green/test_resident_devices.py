from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


def test_explicit_cpu_production_keeps_dtype_and_device_guards():
    from c3l.cc.s135_jax import JaxRuntime, require_jax_runtime, probe_jax_runtime

    runtime = JaxRuntime("test", "test", True, "cpu", "cpu")
    require_jax_runtime(runtime, production=True, platform="cpu")
    assert probe_jax_runtime(production=True, platform="cpu").platform == "cpu"
    assert probe_jax_runtime(production=False).platform == jax.devices()[0].platform
    with pytest.raises(RuntimeError):
        require_jax_runtime(runtime, production=True, platform="gpu")
    with pytest.raises(RuntimeError, match="X64"):
        require_jax_runtime(replace(runtime, enable_x64=False), production=True, platform="cpu")


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
def test_resident_basis_and_reconstruction_use_requested_device(platform):
    from c3l.green.resident_arnoldi import ResidentArnoldi
    from examples.shared.response import reduced_gmres

    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"No {platform} device")
    matrix = np.diag(np.arange(1., 7.) - .4j) + .1j * np.triu(np.ones((6, 6)), 1)
    rhs = np.arange(1., 7.).astype(complex)
    sinks = np.array([rhs.conj(), np.ones(6)])
    device_matrix = jax.device_put(matrix, device)
    chain = ResidentArnoldi(lambda vector: device_matrix @ vector, rhs, sinks, 6, device=device)
    chain.advance(6)
    for array in (chain.basis, chain.vector, chain.sinks):
        assert array.devices() == {device}
    shifts = np.array([.3 + .2j, 1j])
    coefficients, residuals = reduced_gmres(chain.hessenberg, np.linalg.norm(rhs), shifts)
    assert max(residuals) < 1e-12
    for shift, coordinates in zip(shifts, coefficients):
        vector = chain.reconstruct(coordinates)
        assert vector.devices() == {device}
        expected = np.linalg.solve(shift * np.eye(6) - matrix, rhs)
        np.testing.assert_allclose(vector, expected, atol=1e-12, rtol=1e-12)


def test_cpu_memory_guard_rejects_before_allocating(monkeypatch):
    from c3l.green import resident_arnoldi

    monkeypatch.setattr(resident_arnoldi, "_host_available_memory_bytes", lambda: 1024)
    device = SimpleNamespace(platform="cpu", memory_stats=lambda: None)
    with pytest.raises(MemoryError, match="No streaming fallback"):
        resident_arnoldi.check_resident_memory(device, 1025, workspace_bytes=0)
    assert resident_arnoldi.check_resident_memory(device, 512, workspace_bytes=512) == 1024


def test_cpu_memory_guard_respects_job_and_cgroup_limits(monkeypatch):
    from pathlib import Path
    from c3l.green.resident_arnoldi import _host_available_memory_bytes

    files = {
        "/proc/meminfo": "MemAvailable: 100000000 kB\n",
        "/proc/self/status": "VmRSS: 1024 kB\n",
        "/proc/self/cgroup": "0::/slurm/job123\n",
        "/sys/fs/cgroup/slurm/job123/memory.max": str(12 * 2**30),
        "/sys/fs/cgroup/slurm/job123/memory.current": str(3 * 2**30),
    }

    def read(path, *args, **kwargs):
        if str(path) not in files:
            raise FileNotFoundError(path)
        return files[str(path)]

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16384")
    assert _host_available_memory_bytes() == 9 * 2**30
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "4096")
    assert _host_available_memory_bytes() == 4 * 2**30 - 2**20

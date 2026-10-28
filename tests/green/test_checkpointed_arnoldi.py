from __future__ import annotations

import numpy as np
import pytest

from c3l.green.checkpointed_arnoldi import (
    checkpointed_arnoldi_resolvent,
    require_resources,
)
from c3l.green.krylov import arnoldi_resolvent


def test_eom_identity_preserves_the_existing_checkpoint_encoding():
    from types import SimpleNamespace
    from c3l.cc.component_tensor import ComponentCoordinateShell
    from c3l.green.checkpointed_arnoldi import eom_checkpoint_identity

    block = SimpleNamespace(
        key=(1, 0, 0, 0),
        vector_positions=np.array([0, 1], dtype=np.int64),
        indices=np.array([[0], [1]], dtype=np.int64),
    )
    coordinates = SimpleNamespace(n=2, dimension=2, blocks=(block,))
    shell = ComponentCoordinateShell(n=2, response_spin=0, charge=1, creator_rank=5)
    product = SimpleNamespace(
        coefficient=0.3 + 0.1j,
        factors=(SimpleNamespace(components=((0, 1), (2, 3))),),
    )
    oracle = SimpleNamespace(
        rapidities=np.array([0.25 - 0.5j, -0.125 - 0.25j], dtype=np.complex128),
        products=(product,),
    )
    context = SimpleNamespace(
        problem=SimpleNamespace(
            static=SimpleNamespace(oracle=oracle),
            coordinates=coordinates,
            s5_coordinates=shell,
        ),
        amplitudes=SimpleNamespace(
            blocks={(1, 1, 0, 0): np.array([[0.125 + 0.25j]], dtype=np.complex128)}
        ),
    )
    assert eom_checkpoint_identity(context) == (
        "4999c3562ad9bd20527274b6265353f9cb4f3e01c95890b08edd449008264c5d"
    )


class DenseAction:
    def __init__(self, A: np.ndarray):
        self.A = np.asarray(A, dtype=np.complex128)
        self.dimension = self.A.shape[0]

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x


def fixture() -> tuple[DenseAction, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260722)
    A = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    A += np.diag(np.arange(6))
    b = rng.normal(size=6) + 1j * rng.normal(size=6)
    sinks = rng.normal(size=(3, 6)) + 1j * rng.normal(size=(3, 6))
    z = np.asarray((-1.0 + 0.1j, 0.2 + 0.1j, 1.4 + 0.1j))
    return DenseAction(A), b, sinks, z


def test_checkpointed_arnoldi_matches_in_memory_and_resumes(tmp_path) -> None:
    """Catch a resumed chain changing its Krylov basis or sink projections."""

    A, b, sinks, z = fixture()
    directory = tmp_path / "run"
    first = checkpointed_arnoldi_resolvent(
        A,
        b,
        sinks,
        z,
        directory,
        maximum_dimension=6,
        checkpoint_dimensions=(3, 6),
        residual_tolerance=1.0e-30,
        maximum_new_steps=3,
        provenance="small-nonnormal-v1",
    )
    resumed = checkpointed_arnoldi_resolvent(
        A,
        b,
        sinks,
        z,
        directory,
        maximum_dimension=6,
        checkpoint_dimensions=(3, 6),
        residual_tolerance=1.0e-30,
        provenance="small-nonnormal-v1",
    )
    reference = arnoldi_resolvent(
        A,
        b,
        z,
        maximum_dimension=6,
        minimum_dimension=6,
        check_interval=6,
        residual_tolerance=1.0e-30,
    )

    assert first.projection.dimension == 3
    assert resumed.projection.dimension == 6
    np.testing.assert_allclose(
        resumed.projection.hessenberg,
        reference.hessenberg,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        resumed.projection.basis,
        reference.basis,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        resumed.sink_projections,
        sinks @ reference.basis[:, : reference.dimension],
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        resumed.transfer(z),
        np.stack(
            [sinks @ np.linalg.solve(shift * np.eye(6) - A.A, b) for shift in z]
        ),
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    assert np.load(directory / "basis.npy", mmap_mode="r").flags.f_contiguous


def test_zero_step_resume_validates_binding_without_an_action(tmp_path) -> None:
    A, b, sinks, z = fixture()
    directory = tmp_path / "validate-only"
    first = checkpointed_arnoldi_resolvent(
        A,
        b,
        sinks,
        z,
        directory,
        maximum_dimension=6,
        checkpoint_dimensions=(3, 6),
        residual_tolerance=1.0e-30,
        maximum_new_steps=3,
        provenance="validate-only-v1",
    )

    class NoAction(DenseAction):
        def matvec(self, x: np.ndarray) -> np.ndarray:
            raise AssertionError("a validation-only resume executed an action")

    checked = checkpointed_arnoldi_resolvent(
        NoAction(A.A),
        b,
        sinks,
        z,
        directory,
        maximum_dimension=6,
        checkpoint_dimensions=(3, 6),
        residual_tolerance=1.0e-30,
        maximum_new_steps=0,
        provenance="validate-only-v1",
    )
    assert first.projection.dimension == checked.projection.dimension == 3
    with pytest.raises(ValueError, match="checkpoint inputs"):
        checkpointed_arnoldi_resolvent(
            NoAction(A.A),
            b,
            sinks,
            z,
            directory,
            maximum_dimension=6,
            checkpoint_dimensions=(3, 6),
            residual_tolerance=1.0e-30,
            maximum_new_steps=0,
            provenance="changed-provenance",
        )


@pytest.mark.parametrize("binding", ["changed-cache-settings", None])
def test_external_cgs2_orthogonalizer_is_checkpointed_and_resumed(tmp_path, binding) -> None:
    """Cache settings and optional utility metadata do not define the problem."""

    class Step:
        def __init__(self, correction: np.ndarray, remainder: np.ndarray):
            self.correction = correction
            self.remainder = remainder
            self.profile = {
                name: 0.0
                for name in (
                    "work_upload_seconds",
                    "basis_upload_seconds",
                    "cache_fill_upload_seconds",
                    "stream_upload_seconds",
                    "kernel_seconds",
                    "result_download_seconds",
                    "panel_dispatches",
                    "work_upload_bytes",
                    "cache_fill_bytes",
                    "stream_upload_bytes",
                    "cache_hit_bytes",
                    "transient_panel_reuse_bytes",
                    "logical_basis_bytes",
                    "result_download_bytes",
                    "cached_basis_bytes",
                    "wall_seconds",
                )
            }

    class LiteralCGS2:
        backend = "test_cgs2"
        binding = "test_cgs2:v1"
        setup_wall_seconds = 0.0

        @staticmethod
        def orthogonalize(basis: np.ndarray, vector: np.ndarray) -> Step:
            work = vector.copy()
            correction = np.zeros(basis.shape[1], dtype=np.complex128)
            for _ in range(2):
                delta = basis.conj().T @ work
                correction += delta
                work -= basis @ delta
            return Step(correction, work)

    A, b, sinks, z = fixture()
    directory = tmp_path / "external"
    first = checkpointed_arnoldi_resolvent(
        A,
        b,
        sinks,
        z,
        directory,
        maximum_dimension=6,
        checkpoint_dimensions=(3, 6),
        residual_tolerance=1.0e-30,
        maximum_new_steps=3,
        provenance="external-cgs2-v1",
        orthogonalizer=LiteralCGS2(),
    )
    if binding is None:
        del LiteralCGS2.binding
    else:
        LiteralCGS2.binding = binding
    resumed = checkpointed_arnoldi_resolvent(
        A,
        b,
        sinks,
        z,
        directory,
        maximum_dimension=6,
        checkpoint_dimensions=(3, 6),
        residual_tolerance=1.0e-30,
        provenance="external-cgs2-v1",
        orthogonalizer=LiteralCGS2(),
    )
    reference = arnoldi_resolvent(
        A,
        b,
        z,
        maximum_dimension=6,
        minimum_dimension=6,
        check_interval=6,
        residual_tolerance=1.0e-30,
    )

    assert first.projection.dimension == 3
    assert resumed.orthogonalization_backend == "test_cgs2"
    np.testing.assert_allclose(
        resumed.projection.hessenberg,
        reference.hessenberg,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    with np.load(directory / "state.npz", allow_pickle=False) as saved:
        assert str(saved["orthogonalization_backend"].item()) == "test_cgs2"


def test_resume_rejects_changed_source_or_provenance(tmp_path) -> None:
    """Catch a checkpoint being resumed for a different linear problem."""

    A, b, sinks, z = fixture()
    directory = tmp_path / "run"
    checkpointed_arnoldi_resolvent(
        A,
        b,
        sinks,
        z,
        directory,
        maximum_dimension=6,
        checkpoint_dimensions=(2, 6),
        residual_tolerance=1.0e-30,
        maximum_new_steps=2,
        provenance="model-a",
    )
    with pytest.raises(ValueError, match="checkpoint inputs do not match"):
        checkpointed_arnoldi_resolvent(
            A,
            b + np.asarray((1.0e-4, 0, 0, 0, 0, 0)),
            sinks,
            z,
            directory,
            maximum_dimension=6,
            checkpoint_dimensions=(2, 6),
            residual_tolerance=1.0e-30,
            provenance="model-a",
        )
    with pytest.raises(ValueError, match="checkpoint inputs do not match"):
        checkpointed_arnoldi_resolvent(
            A,
            b,
            sinks,
            z,
            directory,
            maximum_dimension=6,
            checkpoint_dimensions=(2, 6),
            residual_tolerance=1.0e-30,
            provenance="model-b",
        )


def test_resource_stops_are_literal() -> None:
    """Catch resource checks silently accepting insufficient RAM or disk."""

    require_resources(
        required_memory_bytes=100,
        required_disk_bytes=200,
        available_memory_bytes=100,
        free_disk_bytes=200,
    )
    with pytest.raises(MemoryError, match="memory"):
        require_resources(
            required_memory_bytes=100,
            required_disk_bytes=200,
            available_memory_bytes=99,
            free_disk_bytes=200,
        )
    with pytest.raises(OSError, match="disk"):
        require_resources(
            required_memory_bytes=100,
            required_disk_bytes=200,
            available_memory_bytes=100,
            free_disk_bytes=199,
        )

"""Bounded complex128 JAX panels for disk-backed Arnoldi reorthogonalization."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from c3l.green.arnoldi_memory import select_basis_cache


ComplexVector = NDArray[np.complex128]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JaxOrthogonalizationStep:
    """One two-pass correction, remainder, and synchronized cost record."""

    correction: ComplexVector
    remainder: ComplexVector
    profile: dict[str, float]


def _panel_cgs_update(jnp: Any, reference: Any, updated: Any, panel_rows: Any):
    """Project against one transposed basis panel and accumulate its update."""

    coefficients = jnp.conj(panel_rows) @ reference
    return updated - panel_rows.T @ coefficients, coefficients


class JaxPanelArnoldiOrthogonalizer:
    """Apply two-pass classical Gram--Schmidt using bounded JAX panels.

    Complete panels in the declared cache prefix remain device-resident.
    Partial panels are padded to one fixed JIT shape and streamed from the
    authoritative Fortran-order host basis.
    """

    backend = "jax_panel_cgs2"

    def __init__(
        self,
        *,
        jax: Any,
        jnp: Any,
        device: Any,
        vector_dimension: int,
        panel_columns: int = 32,
        cache_columns: int | None = 384,
        maximum_columns: int | None = None,
        basis_budget_bytes: int | None = None,
    ) -> None:
        self._memory_policy = None
        self.maximum_columns = maximum_columns
        if cache_columns is None:
            if maximum_columns is None or basis_budget_bytes is None:
                raise ValueError("budget selection requires maximum_columns and basis_budget_bytes")
            self._memory_policy = select_basis_cache(
                vector_dimension=vector_dimension, maximum_columns=maximum_columns,
                panel_columns=panel_columns, budget_bytes=basis_budget_bytes,
            )
            cache_columns = self._memory_policy.cache_columns
        elif maximum_columns is not None or basis_budget_bytes is not None:
            raise ValueError("use cache_columns=None to select a net basis memory budget")
        for name, value, positive in (
            ("vector_dimension", vector_dimension, True),
            ("panel_columns", panel_columns, True),
            ("cache_columns", cache_columns, False),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if (positive and value <= 0) or (not positive and value < 0):
                raise ValueError(f"{name} has an invalid value")
        if cache_columns % panel_columns:
            raise ValueError("cache_columns must contain complete panels")
        if not bool(jax.config.read("jax_enable_x64")):
            raise RuntimeError("JAX_ENABLE_X64=true is required for Arnoldi")

        self.jax = jax
        self.jnp = jnp
        self.device = device
        self._platform = device.platform
        self.vector_dimension = int(vector_dimension)
        self.panel_columns = int(panel_columns)
        self.cache_columns = int(cache_columns)
        self._cache: dict[int, Any] = {}
        self._basis_identity: tuple[object, ...] | None = None
        self._basis_owner: np.ndarray | None = None
        self._tail_host: np.ndarray | None = None
        self._tail_valid_rows = 0
        self._streaming_reported = False
        self._active_columns = 0

        def kernel(reference: Any, updated: Any, panel_rows: Any):
            return _panel_cgs_update(jnp, reference, updated, panel_rows)

        setup_started = perf_counter()
        vector_shape = jax.ShapeDtypeStruct(
            (self.vector_dimension,), np.dtype(np.complex128)
        )
        panel_shape = jax.ShapeDtypeStruct(
            (self.panel_columns, self.vector_dimension),
            np.dtype(np.complex128),
        )
        with jax.default_device(device):
            self._kernel = jax.jit(kernel).lower(
                vector_shape, vector_shape, panel_shape
            ).compile()
        self.setup_wall_seconds = float(perf_counter() - setup_started)

    @property
    def memory_metadata(self) -> dict[str, object]:
        """Describe the configured capacity without changing checkpoint arrays."""

        policy = self._memory_policy
        required_bytes = (
            policy.required_bytes if policy is not None
            else ((self._active_columns + self.panel_columns - 1) // self.panel_columns)
            * self.panel_columns * self.vector_dimension * 16
        )
        cache_bytes = self.cache_columns * self.vector_dimension * 16
        return {
            "mode": policy.mode if policy is not None else (
                "resident_capacity" if required_bytes <= cache_bytes else "streaming"
            ),
            "reason": "explicit_basis_budget" if policy is not None else "explicit_cache_columns",
            "cache_columns": self.cache_columns,
            "cache_bytes": cache_bytes,
            "required_bytes": required_bytes,
            "budget_bytes": policy.budget_bytes if policy is not None else cache_bytes,
        }

    @property
    def cached_basis_bytes(self) -> int:
        """Return the currently resident logical basis bytes."""

        return int(
            len(self._cache)
            * self.panel_columns
            * self.vector_dimension
            * np.dtype(np.complex128).itemsize
        )

    def _tail_panel(self, panel_rows: np.ndarray) -> np.ndarray:
        width = int(panel_rows.shape[0])
        if self._tail_host is None:
            self._tail_host = np.zeros(
                (self.panel_columns, self.vector_dimension),
                dtype=np.complex128,
                order="C",
            )
        self._tail_host[:width] = panel_rows
        if width < self._tail_valid_rows:
            self._tail_host[width : self._tail_valid_rows].fill(0.0)
        self._tail_valid_rows = width
        return self._tail_host

    def _put(self, host: np.ndarray) -> tuple[Any, float]:
        started = perf_counter()
        value = self.jax.device_put(host, self.device)
        value.block_until_ready()
        return value, float(perf_counter() - started)

    @staticmethod
    def _identity(values: np.ndarray) -> tuple[object, ...]:
        root = values
        while isinstance(root.base, np.ndarray):
            root = root.base
        return (
            str(Path(values.filename).resolve()) if isinstance(values, np.memmap) and values.filename is not None else "array",
            id(root),
            int(values.__array_interface__["data"][0]),
            int(values.shape[0]),
            int(values.strides[0]),
            int(values.strides[1]),
        )

    def reset(self) -> None:
        """Invalidate cached columns before mutating or replacing a basis prefix."""

        self._cache.clear()
        self._basis_identity = None
        self._basis_owner = None
        self._active_columns = 0
        self._streaming_reported = False
        self._tail_host = None
        self._tail_valid_rows = 0

    def orthogonalize(
        self, basis: np.ndarray, vector: np.ndarray
    ) -> JaxOrthogonalizationStep:
        """Return the exact two-pass panel correction and writable remainder."""

        started = perf_counter()
        values = np.asanyarray(basis)
        work_host = np.asarray(vector)
        if (
            values.ndim != 2
            or values.shape[0] != self.vector_dimension
            or values.shape[1] == 0
            or values.dtype != np.complex128
        ):
            raise ValueError("basis must be a nonempty complex128 matrix")
        if (
            work_host.shape != (self.vector_dimension,)
            or work_host.dtype != np.complex128
            or not np.all(np.isfinite(work_host))
        ):
            raise ValueError("work vector must be finite complex128")

        if self.maximum_columns is not None and values.shape[1] > self.maximum_columns:
            raise ValueError("basis exceeds declared maximum_columns")
        identity = self._identity(values)
        if self._basis_identity != identity or values.shape[1] < self._active_columns:
            self.reset()
            self._basis_identity = identity
            root = values
            while isinstance(root.base, np.ndarray):
                root = root.base
            self._basis_owner = root

        work_device, work_upload_seconds = self._put(work_host)
        width_total = int(values.shape[1])
        self._active_columns = width_total
        complete_columns = (width_total // self.panel_columns) * self.panel_columns
        if (
            self._platform in ("gpu", "cuda", "rocm")
            and complete_columns > self.cache_columns
            and not self._streaming_reported
        ):
            logger.warning(
                "Arnoldi basis streaming active: basis panels are transferred repeatedly "
                "between CPU memory and GPU memory because the configured GPU cache "
                "is insufficient. This can substantially slow orthogonalization. "
                "Increase the cache budget only if sufficient GPU memory remains after "
                "operator/workspace allocation. dimension=%d active_columns=%d "
                "cache_columns=%d reason=%s required_bytes=%d cache_bytes=%d",
                self.vector_dimension, width_total, self.cache_columns,
                self.memory_metadata["reason"], self.memory_metadata["required_bytes"],
                self.memory_metadata["cache_bytes"],
            )
            self._streaming_reported = True
        padded_width = (
            (width_total + self.panel_columns - 1) // self.panel_columns
        ) * self.panel_columns
        scalar_bytes = np.dtype(np.complex128).itemsize
        profile = {
            "work_upload_seconds": work_upload_seconds,
            "basis_upload_seconds": 0.0,
            "cache_fill_upload_seconds": 0.0,
            "stream_upload_seconds": 0.0,
            "kernel_seconds": 0.0,
            "result_download_seconds": 0.0,
            "panel_dispatches": 0.0,
            "work_upload_bytes": float(self.vector_dimension * scalar_bytes),
            "cache_fill_bytes": 0.0,
            "stream_upload_bytes": 0.0,
            "cache_hit_bytes": 0.0,
            "transient_panel_reuse_bytes": 0.0,
            "logical_basis_bytes": float(
                2 * self.vector_dimension * width_total * scalar_bytes
            ),
            "result_download_bytes": float(
                (self.vector_dimension + padded_width) * scalar_bytes
            ),
            "cached_basis_bytes": 0.0,
            "wall_seconds": 0.0,
        }
        pass_corrections: list[Any] = []
        transient_tail: tuple[int, int, Any] | None = None

        for _ in range(2):
            reference = work_device
            updated = work_device
            coefficient_parts: list[Any] = []
            for start in range(0, width_total, self.panel_columns):
                stop = min(start + self.panel_columns, width_total)
                width = stop - start
                cacheable = (
                    width == self.panel_columns and stop <= self.cache_columns
                )
                panel_device = self._cache.get(start) if cacheable else None
                persistent_cache_hit = panel_device is not None
                logical_bytes = width * self.vector_dimension * scalar_bytes
                physical_bytes = (
                    self.panel_columns * self.vector_dimension * scalar_bytes
                )
                if (
                    panel_device is None
                    and width != self.panel_columns
                    and transient_tail is not None
                    and transient_tail[:2] == (start, width)
                ):
                    panel_device = transient_tail[2]
                    profile["transient_panel_reuse_bytes"] += float(
                        physical_bytes
                    )
                if panel_device is None:
                    panel_rows = values[:, start:stop].T
                    if not panel_rows.flags.c_contiguous:
                        panel_rows = np.ascontiguousarray(panel_rows)
                    if width != self.panel_columns:
                        panel_rows = self._tail_panel(panel_rows)
                    panel_device, upload_seconds = self._put(panel_rows)
                    profile["basis_upload_seconds"] += upload_seconds
                    if cacheable:
                        self._cache[start] = panel_device
                        profile["cache_fill_upload_seconds"] += upload_seconds
                        profile["cache_fill_bytes"] += float(physical_bytes)
                    else:
                        profile["stream_upload_seconds"] += upload_seconds
                        profile["stream_upload_bytes"] += float(physical_bytes)
                        if width != self.panel_columns:
                            transient_tail = (start, width, panel_device)
                elif persistent_cache_hit:
                    profile["cache_hit_bytes"] += float(logical_bytes)

                kernel_started = perf_counter()
                updated, coefficients = self._kernel(
                    reference, updated, panel_device
                )
                updated.block_until_ready()
                profile["kernel_seconds"] += float(
                    perf_counter() - kernel_started
                )
                profile["panel_dispatches"] += 1.0
                coefficient_parts.append(coefficients)
            work_device = updated
            pass_corrections.append(self.jnp.concatenate(coefficient_parts))

        correction_device = pass_corrections[0] + pass_corrections[1]
        download_started = perf_counter()
        correction_value, remainder_value = self.jax.device_get(
            (correction_device, work_device)
        )
        profile["result_download_seconds"] = float(
            perf_counter() - download_started
        )
        correction = np.array(
            correction_value[:width_total], dtype=np.complex128, copy=True
        )
        remainder = np.array(remainder_value, dtype=np.complex128, copy=True)
        if (
            correction.shape != (width_total,)
            or remainder.shape != (self.vector_dimension,)
            or not np.all(np.isfinite(correction))
            or not np.all(np.isfinite(remainder))
        ):
            raise RuntimeError("JAX Arnoldi orthogonalization returned invalid data")
        profile["cached_basis_bytes"] = float(self.cached_basis_bytes)
        profile["wall_seconds"] = float(perf_counter() - started)
        return JaxOrthogonalizationStep(
            correction=correction,
            remainder=remainder,
            profile=profile,
        )


__all__ = [
    "JaxOrthogonalizationStep",
    "JaxPanelArnoldiOrthogonalizer",
]

"""Absolute/relative error budgets for reference modes and numerical checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class AbsRelTolerance:
    """Absolute/relative tolerance pair, stored as float64 scalars."""

    absolute: np.float64
    relative: np.float64

    def __post_init__(self) -> None:
        absolute = np.float64(self.absolute)
        relative = np.float64(self.relative)
        if not np.isfinite(absolute) or not np.isfinite(relative):
            raise ValueError("tolerances must be finite")
        if absolute < 0.0 or relative < 0.0:
            raise ValueError("tolerances must be nonnegative")
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "relative", relative)

    def threshold(self, scale: float | np.floating[Any]) -> np.float64:
        """Return ``absolute + relative * abs(scale)`` in float64."""

        return np.float64(self.absolute + self.relative * abs(np.float64(scale)))

    def to_dict(self) -> dict[str, float]:
        return {"absolute": float(self.absolute), "relative": float(self.relative)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AbsRelTolerance:
        return cls(
            absolute=np.float64(data["absolute"]), relative=np.float64(data["relative"])
        )


@dataclass(frozen=True, slots=True)
class TolerancePolicy:
    """Serialized numerical acceptance policy with distinct error budgets."""

    input_abs_rel: AbsRelTolerance
    algebra_abs_rel: AbsRelTolerance
    eigensolver_abs_rel: AbsRelTolerance
    krylov_abs_rel: AbsRelTolerance
    nonlinear_abs_rel: AbsRelTolerance
    grid_and_sum_rule: AbsRelTolerance
    benchmark_regression: AbsRelTolerance
    dense_scale_floor: np.float64 = np.float64(1.0e-13)
    dense_memory_limit_bytes: int = 1_073_741_824
    normal_mode_condition_budget_fraction: np.float64 = np.float64(0.1)

    def __post_init__(self) -> None:
        tolerance_fields = (
            self.input_abs_rel,
            self.algebra_abs_rel,
            self.eigensolver_abs_rel,
            self.krylov_abs_rel,
            self.nonlinear_abs_rel,
            self.grid_and_sum_rule,
            self.benchmark_regression,
        )
        if not all(isinstance(value, AbsRelTolerance) for value in tolerance_fields):
            raise TypeError("every tolerance bucket must be an AbsRelTolerance")
        floor = np.float64(self.dense_scale_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("dense_scale_floor must be finite and positive")
        if (
            isinstance(self.dense_memory_limit_bytes, bool)
            or not isinstance(self.dense_memory_limit_bytes, int)
            or self.dense_memory_limit_bytes <= 0
        ):
            raise ValueError("dense_memory_limit_bytes must be a positive integer")
        condition_fraction = np.float64(self.normal_mode_condition_budget_fraction)
        if not np.isfinite(condition_fraction) or not 0.0 < condition_fraction <= 1.0:
            raise ValueError("normal_mode_condition_budget_fraction must lie in (0, 1]")
        object.__setattr__(self, "dense_scale_floor", floor)
        object.__setattr__(
            self, "normal_mode_condition_budget_fraction", condition_fraction
        )

    @classmethod
    def first(cls) -> TolerancePolicy:
        """Return the default numerical error budgets and dense-oracle limits."""

        return cls(
            input_abs_rel=AbsRelTolerance(1.0e-13, 1.0e-12),
            algebra_abs_rel=AbsRelTolerance(1.0e-13, 1.0e-12),
            eigensolver_abs_rel=AbsRelTolerance(1.0e-12, 1.0e-10),
            krylov_abs_rel=AbsRelTolerance(1.0e-12, 1.0e-10),
            nonlinear_abs_rel=AbsRelTolerance(1.0e-10, 1.0e-10),
            grid_and_sum_rule=AbsRelTolerance(1.0e-6, 1.0e-4),
            benchmark_regression=AbsRelTolerance(1.0e-8, 1.0e-6),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_abs_rel": self.input_abs_rel.to_dict(),
            "algebra_abs_rel": self.algebra_abs_rel.to_dict(),
            "eigensolver_abs_rel": self.eigensolver_abs_rel.to_dict(),
            "krylov_abs_rel": self.krylov_abs_rel.to_dict(),
            "nonlinear_abs_rel": self.nonlinear_abs_rel.to_dict(),
            "grid_and_sum_rule": self.grid_and_sum_rule.to_dict(),
            "benchmark_regression": self.benchmark_regression.to_dict(),
            "dense_scale_floor": float(self.dense_scale_floor),
            "dense_memory_limit_bytes": self.dense_memory_limit_bytes,
            "normal_mode_condition_budget_fraction": float(
                self.normal_mode_condition_budget_fraction
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TolerancePolicy:
        keys = (
            "input_abs_rel",
            "algebra_abs_rel",
            "eigensolver_abs_rel",
            "krylov_abs_rel",
            "nonlinear_abs_rel",
            "grid_and_sum_rule",
            "benchmark_regression",
        )
        tolerances = {key: AbsRelTolerance.from_dict(data[key]) for key in keys}
        return cls(
            **tolerances,
            dense_scale_floor=np.float64(data["dense_scale_floor"]),
            dense_memory_limit_bytes=int(data["dense_memory_limit_bytes"]),
            normal_mode_condition_budget_fraction=np.float64(
                data["normal_mode_condition_budget_fraction"]
            ),
        )

"""Quadratic normal modes for the C3L particle-hole reference."""

from .normal_modes import (
    NormalModeReference,
    match_normal_modes,
    near_degenerate_groups,
    normal_mode_reference,
    normal_mode_reference_from_arrays,
)

__all__ = [
    "NormalModeReference",
    "match_normal_modes",
    "near_degenerate_groups",
    "normal_mode_reference",
    "normal_mode_reference_from_arrays",
]

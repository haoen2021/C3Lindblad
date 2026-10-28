"""Explicit net device-memory budgets for Arnoldi basis panels."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BasisCachePolicy:
    """Complete-panel capacity after the caller reserves operator workspace."""

    cache_columns: int
    required_bytes: int
    budget_bytes: int
    mode: str


def select_basis_cache(
    *, vector_dimension: int, maximum_columns: int, panel_columns: int,
    budget_bytes: int,
) -> BasisCachePolicy:
    """Round a net basis budget down and requested capacity up to panels."""

    for name, value in (
        ("vector_dimension", vector_dimension),
        ("maximum_columns", maximum_columns),
        ("panel_columns", panel_columns),
        ("budget_bytes", budget_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < (0 if name == "budget_bytes" else 1):
            raise ValueError(f"{name} has an invalid value")
    panels = (maximum_columns + panel_columns - 1) // panel_columns
    panel_bytes = vector_dimension * panel_columns * 16
    cached_panels = min(panels, budget_bytes // panel_bytes)
    return BasisCachePolicy(
        cache_columns=cached_panels * panel_columns,
        required_bytes=panels * panel_bytes,
        budget_bytes=budget_bytes,
        mode="resident_capacity" if cached_panels == panels else "streaming",
    )

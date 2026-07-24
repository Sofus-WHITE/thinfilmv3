"""Placeholder for deposition-time-to-thickness models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepositionRate:
    """Simple future-ready deposition rate representation."""

    material_name: str
    nm_per_min: float

    def thickness_from_minutes(self, minutes: float) -> float:
        """Estimate thickness from deposition duration."""

        return self.nm_per_min * minutes

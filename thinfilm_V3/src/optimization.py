"""Placeholder for future optimization workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationSettings:
    """Minimal settings object reserved for future optimizers."""

    max_iterations: int = 200
    tolerance: float = 1e-6

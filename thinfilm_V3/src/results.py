"""Simulation result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulationResult:
    """Reflectance spectrum and metadata returned by an optical model."""

    wavelengths_nm: NDArray[np.float64]
    reflectance: NDArray[np.float64]
    angle_deg: float
    stack_name: str
    stack_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelengths = np.asarray(self.wavelengths_nm, dtype=float)
        reflectance = np.asarray(self.reflectance, dtype=float)
        if wavelengths.shape != reflectance.shape:
            raise ValueError("Wavelength and reflectance arrays must have matching shapes.")
        object.__setattr__(self, "wavelengths_nm", wavelengths)
        object.__setattr__(self, "reflectance", reflectance)

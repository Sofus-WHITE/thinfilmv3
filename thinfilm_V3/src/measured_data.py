"""Placeholder module for measured spectra and sample metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MeasuredSpectrum:
    """Measured reflectance data prepared for future fitting workflows."""

    wavelengths_nm: NDArray[np.float64]
    reflectance: NDArray[np.float64]
    label: str = "measured spectrum"

    def __post_init__(self) -> None:
        wavelengths = np.asarray(self.wavelengths_nm, dtype=float)
        reflectance = np.asarray(self.reflectance, dtype=float)
        if wavelengths.shape != reflectance.shape:
            raise ValueError("Measured wavelengths and reflectance must have matching shapes.")
        object.__setattr__(self, "wavelengths_nm", wavelengths)
        object.__setattr__(self, "reflectance", reflectance)

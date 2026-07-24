"""Placeholder objective functions for future inverse design and fitting."""

from __future__ import annotations

import numpy as np

from .results import SimulationResult
from .measured_data import MeasuredSpectrum


def mean_squared_reflectance_error(
    simulation: SimulationResult,
    measurement: MeasuredSpectrum,
) -> float:
    """Compute a basic MSE for spectra already sampled on the same wavelength grid."""

    if simulation.wavelengths_nm.shape != measurement.wavelengths_nm.shape:
        raise ValueError("Simulation and measurement must be sampled on the same grid.")
    return float(np.mean((simulation.reflectance - measurement.reflectance) ** 2))

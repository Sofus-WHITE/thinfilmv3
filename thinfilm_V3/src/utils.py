"""Small shared helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wavelength_grid(
    start_nm: float = 400.0,
    stop_nm: float = 700.0,
    points: int = 301,
) -> NDArray[np.float64]:
    """Create a visible-range wavelength grid in nanometers."""

    if points < 2:
        raise ValueError("At least two wavelength points are required.")
    return np.linspace(start_nm, stop_nm, points, dtype=float)

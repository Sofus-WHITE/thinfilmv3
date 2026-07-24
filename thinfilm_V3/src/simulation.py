"""Small orchestration helpers for running simulations."""

from __future__ import annotations

from numpy.typing import ArrayLike

from .optical_model import OpticalModel
from .results import SimulationResult
from .stack import ThinFilmStack


def run_simulation(
    model: OpticalModel,
    stack: ThinFilmStack,
    wavelengths_nm: ArrayLike,
    angle_deg: float,
) -> SimulationResult:
    """Run any optical model through the common interface."""

    return model.simulate(stack=stack, wavelengths_nm=wavelengths_nm, angle_deg=angle_deg)

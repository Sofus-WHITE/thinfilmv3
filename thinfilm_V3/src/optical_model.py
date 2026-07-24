"""Replaceable optical-model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from .results import SimulationResult
from .stack import ThinFilmStack


class OpticalModel(ABC):
    """Interface shared by TMM, roughness, scattering, and future optical models."""

    @abstractmethod
    def simulate(
        self,
        stack: ThinFilmStack,
        wavelengths_nm: ArrayLike,
        angle_deg: float,
    ) -> SimulationResult:
        """Simulate the stack and return a reusable result object."""

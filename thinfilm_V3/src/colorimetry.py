"""Perceived-color utilities for simulated reflectance spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import reflectance_to_srgb, reflectance_to_xyz
from .results import SimulationResult


@dataclass(frozen=True)
class PerceivedColor:
    """Display-ready color derived from a reflectance spectrum under an illuminant."""

    srgb: tuple[float, float, float]
    srgb_255: tuple[int, int, int]
    xyz: tuple[float, float, float]

    @property
    def hex(self) -> str:
        """Return the color as a CSS-style hex string."""

        return "#{:02x}{:02x}{:02x}".format(*self.srgb_255)


def perceived_color_from_result(result: SimulationResult) -> PerceivedColor:
    """Estimate perceived sRGB color of a simulated reflectance spectrum under D65."""

    xyz = reflectance_to_xyz(result.reflectance, wavelengths_nm=result.wavelengths_nm)
    srgb = reflectance_to_srgb(result.reflectance, wavelengths_nm=result.wavelengths_nm)

    srgb_255 = tuple(int(round(channel * 255)) for channel in srgb)
    return PerceivedColor(
        srgb=tuple(float(channel) for channel in srgb),
        srgb_255=srgb_255,
        xyz=tuple(float(channel) for channel in xyz),
    )

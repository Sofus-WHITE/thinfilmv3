"""Efficient angle sweeps using prepared TMM stacks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .color import prepare_color_conversion, reflectance_to_srgb
from .stack import ThinFilmStack
from .thickness_sweep import _resolve_wavelengths, _quality_settings
from .tmm_model import TMMModel


@dataclass(frozen=True)
class AngleSweepResult:
    """Colour result from varying angle of incidence."""

    angle_values_deg: NDArray[np.float64]
    rgb_values: NDArray[np.float64]
    stack_label: str
    wavelengths_nm: NDArray[np.float64]
    reflectance_spectra: NDArray[np.float64] | None = None


def run_angle_sweep(
    stack: ThinFilmStack,
    model: TMMModel,
    angle_min_deg: float,
    angle_max_deg: float,
    wavelengths_nm: ArrayLike | None = None,
    num_points: int | None = None,
    quality: str = "normal",
    save_reflectance: bool = False,
) -> AngleSweepResult:
    """Vary incident angle and return predicted colour at each angle."""

    wavelengths = _resolve_wavelengths(wavelengths_nm, quality)
    point_count = num_points or int(_quality_settings(quality)["points_1d"])
    angle_values = np.linspace(angle_min_deg, angle_max_deg, point_count, dtype=float)

    prepared = model.prepare_stack(stack, wavelengths)
    color_cache = prepare_color_conversion(wavelengths)
    rgb_values = np.empty((point_count, 3), dtype=float)
    reflectance_spectra = (
        np.empty((point_count, wavelengths.size), dtype=float) if save_reflectance else None
    )

    for index, angle_deg in enumerate(angle_values):
        reflectance = model.reflectance_from_prepared(
            prepared,
            prepared.base_d_list,
            angle_deg,
        )
        rgb_values[index] = reflectance_to_srgb(reflectance, cache=color_cache)
        if reflectance_spectra is not None:
            reflectance_spectra[index] = reflectance

    return AngleSweepResult(
        angle_values_deg=angle_values,
        rgb_values=rgb_values,
        stack_label=prepared.display_summary.replace(" / ", " | "),
        wavelengths_nm=wavelengths,
        reflectance_spectra=reflectance_spectra,
    )
